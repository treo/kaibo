package xaibo.primitives

import xaibo.*
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * The harness-grade agent loop, as an ordinary module: what
 * [SimpleToolOrchestrator] does, plus the affordances a live frontend needs.
 *
 * - **streaming rounds** (`stream: true`): every model call runs through
 *   `llm.generateStream`, so each `StreamFrame` (text, *thinking*, tool-call,
 *   usage) crosses the module proxy as a YIELD event — a frontend can render
 *   the whole turn live from the event bus alone. The settled response is
 *   rebuilt by [StreamAssembler]; thinking stays separate from content.
 * - **persistence** (`persist: true`): user, assistant and tool-result
 *   messages are written to the injected history, making the conversation log
 *   the transcript — a history that stores to disk is then the source of truth.
 * - **steering** ([SteeringProtocol]): `steer(text)` from another coroutine
 *   queues a user message DURING a running turn. It lands at the earliest
 *   point the exchange grammar allows — after a tool batch is fully answered,
 *   or as a follow-up round after a final answer — and is never lost or split.
 * - **compaction**: if the history module implements [CompactionProtocol] it
 *   is asked to compact before each turn; no loop instrumentation needed.
 * - **chunked delivery**: the settled answer reaches the response lane in
 *   `stream_chunk_chars` slices, so response-consuming adapters see deltas.
 *
 * Config: `system_prompt`, `max_thoughts` (24), `temperature` (0.0),
 * `reasoning_effort` (name of [xaibo.ReasoningEffort], applied to every round),
 * `stream` (false), `stream_chunk_chars` (240), `persist` (false).
 */
class StreamingToolOrchestrator(
    val response: ResponseProtocol,
    val llm: LLMProtocol,
    val toolProvider: ToolProviderProtocol,
    val history: ConversationHistoryProtocol,
    val config: Map<String, Any?> = emptyMap(),
) : TextMessageHandlerProtocol, SteeringProtocol {

    private val systemPrompt = config["system_prompt"] as? String ?: ""
    private val maxThoughts = (config["max_thoughts"] as? Number)?.toInt() ?: 24
    private val stream = config["stream"] as? Boolean ?: false
    private val streamChunkChars = (config["stream_chunk_chars"] as? Number)?.toInt() ?: 240
    private val persist = config["persist"] as? Boolean ?: false
    private val temperature = (config["temperature"] as? Number)?.toDouble() ?: 0.0
    private val reasoningEffort = (config["reasoning_effort"] as? String)
        ?.let { runCatching { ReasoningEffort.valueOf(it.uppercase()) }.getOrNull() }

    private val steering = ConcurrentLinkedQueue<String>()

    /** Queue a user message into the turn that is running right now. */
    override fun steer(text: String) {
        text.trim().takeIf { it.isNotEmpty() }?.let(steering::add)
    }

    private suspend fun buildConversation(): MutableList<LLMMessage> {
        val conversation = history.getHistory().toMutableList()
        if (systemPrompt.isNotEmpty()) conversation.add(0, LLMMessage.system(systemPrompt))
        return conversation
    }

    private suspend fun persist(message: LLMMessage) {
        if (persist) history.addMessage(message)
    }

    /** Append + journal queued steering messages; true if any landed. */
    private suspend fun drainSteering(conversation: MutableList<LLMMessage>): Boolean {
        var any = false
        while (true) {
            val text = steering.poll() ?: break
            val message = LLMMessage.user(text)
            conversation.add(message)
            persist(message)
            any = true
        }
        return any
    }

    private suspend fun generate(conversation: List<LLMMessage>, options: LLMOptions): LLMResponse =
        if (stream && llm.streams) {
            val assembler = StreamAssembler()
            llm.generateStream(conversation, options).collect(assembler::feed)
            assembler.response()
        } else llm.generate(conversation, options)

    override suspend fun handleText(text: String) {
        // compaction is a property of the history module, not of this loop
        (history as? CompactionProtocol)?.compactIfNeeded(llm)

        val userMessage = LLMMessage.user(text)
        val conversation = buildConversation()
        if (persist) {
            history.addMessage(userMessage)
            conversation.add(userMessage)
        } else conversation.add(userMessage)

        // anything steered while the loop sat idle (or an abort stranded)
        // opens this turn rather than vanishing into the next one
        drainSteering(conversation)

        val tools = toolProvider.listTools()
        var thoughts = 0
        var finalText = ""

        while (thoughts < maxThoughts) {
            thoughts++
            val lastTurn = thoughts == maxThoughts
            if (lastTurn)
                conversation.add(LLMMessage.system("Maximum tool usage reached. Tools unavailable — answer now."))

            // the reasoning level rides on every round: a model that reasons
            // hard on round 1 and coasts on round 9 is not a session setting
            val options = LLMOptions(
                temperature = temperature,
                functions = if (!lastTurn) tools else null,
                reasoningEffort = reasoningEffort,
            )
            val llmResponse = generate(conversation, options)
            llmResponse.usage?.let { response.respondEvent(UsageEvent(it)) }

            val assistant = LLMMessage(
                role = if (llmResponse.toolCalls != null) LLMRole.FUNCTION else LLMRole.ASSISTANT,
                content = listOf(LLMMessageContent(LLMMessageContentType.TEXT, text = llmResponse.content)),
                toolCalls = llmResponse.toolCalls,
            )
            conversation.add(assistant)
            persist(assistant)

            val calls = llmResponse.toolCalls.orEmpty()
            if (calls.isNotEmpty() && !lastTurn) {
                val (results, _) = executeToolBatch(toolProvider, response, calls)
                results.forEach { conversation.add(it); persist(it) }
                // every call has its result: the exchange is legal again, so
                // this is the earliest point a steered message may be added
                if (drainSteering(conversation)) thoughts = 0 // a steered message earns its own rounds
                continue
            }

            // the round answered without tools, so the turn would end — unless
            // the user spoke during that answer; then it runs as a follow-up
            // round, re-read from the log, which also drops the tools-off guard
            if (calls.isEmpty() && drainSteering(conversation)) {
                conversation.clear()
                conversation.addAll(buildConversation())
                finalText = ""
                thoughts = 0
                continue
            }

            finalText = llmResponse.content
            break
        }

        if (finalText.isEmpty()) finalText = "(no response)"

        // chunked delivery: response-consuming adapters see deltas too
        if (streamChunkChars > 0 && finalText.length > streamChunkChars)
            finalText.chunked(streamChunkChars).forEach { response.respondText(it) }
        else response.respondText(finalText)
    }
}
