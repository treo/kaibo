package kaibo.primitives

import kaibo.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Conversation history kept in memory, optionally seeded from a list of
 * OpenAI-format messages. Trims to `max_history` (default 100).
 */
class SimpleConversation(val config: Map<String, Any?> = emptyMap()) : ConversationHistoryProtocol {
    private val maxHistory = (config["max_history"] as? Number)?.toInt() ?: 100
    private var history = (config["initial_messages"] as? List<Map<String, Any?>> ?: emptyList())
        .map { fromOpenaiMessage(it) }

    override suspend fun getHistory(): List<LLMMessage> = history.toList()

    override suspend fun addMessage(message: LLMMessage) {
        history = (history + message).takeLast(maxHistory)
    }

    override suspend fun clearHistory() {
        history = emptyList()
    }

    companion object {
        /** Convert an OpenAI-format message ({role, content, name}) to an LLMMessage */
        @Suppress("UNCHECKED_CAST")
        fun fromOpenaiMessage(msg: Map<String, Any?>): LLMMessage {
            val role = LLMRole.of(msg["role"] as? String ?: "user")
            val name = msg["name"] as? String
            return when (val content = msg["content"]) {
                is List<*> -> LLMMessage(role, content.filterIsInstance<Map<String, Any?>>().map { item ->
                    when (item["type"]) {
                        "image_url" -> LLMMessageContent(
                            LLMMessageContentType.IMAGE,
                            image = ((item["image_url"] as? Map<String, Any?>)?.get("url") as? String) ?: ""
                        )
                        else -> LLMMessageContent(LLMMessageContentType.TEXT, text = (item["text"] as? String) ?: "")
                    }
                }.ifEmpty { listOf(LLMMessageContent(LLMMessageContentType.TEXT, text = "")) }, name)
                else -> LLMMessage(role, listOf(LLMMessageContent(LLMMessageContentType.TEXT, text = content as? String ?: "")), name)
            }
        }

        fun fromOpenaiMessages(messages: List<Map<String, Any?>>, config: Map<String, Any?> = emptyMap()) =
            SimpleConversation(config + mapOf("initial_messages" to messages))
    }
}

/**
 * Accumulates the response for the current turn. `getResponse` hands off the
 * response and resets turn-scoped events.
 */
class ResponseHandler(@Suppress("unused_constructor_parameter") val config: Map<String, Any?> = emptyMap())
    : ResponseProtocol {

    private val accumulated = Response()

    override suspend fun getResponse() =
        Response(accumulated.text, accumulated.attachments.toMutableList(), accumulated.events.toMutableList())
            .also { accumulated.events.clear() }

    override suspend fun respondText(text: String) {
        accumulated.text = (accumulated.text ?: "") + text
    }

    override suspend fun respondImage(bytes: ByteArray) {
        accumulated.attachments.add(FileAttachment(bytes, FileType.IMAGE))
    }

    override suspend fun respondAudio(bytes: ByteArray) {
        accumulated.attachments.add(FileAttachment(bytes, FileType.AUDIO))
    }

    override suspend fun respondFile(bytes: ByteArray) {
        accumulated.attachments.add(FileAttachment(bytes, FileType.FILE))
    }

    override suspend fun respond(response: Response) {
        response.text?.let { accumulated.text = (accumulated.text ?: "") + it }
        accumulated.attachments.addAll(response.attachments)
        accumulated.events.addAll(response.events)
    }

    override suspend fun respondEvent(event: Any) {
        accumulated.events.add(event)
    }
}

/**
 * LLM for testing: answers with exactly the configured `responses` (a list of
 * LLMResponse-shaped maps), cycling when exhausted. `streaming_delay` is in
 * milliseconds, `streaming_chunk_size` in characters.
 */
class MockLLM(val config: Map<String, Any?> = emptyMap()) : LLMProtocol {
    private val streamingDelay = (config["streaming_delay"] as? Number)?.toInt() ?: 0
    private val chunkSize = (config["streaming_chunk_size"] as? Number)?.toInt() ?: 3
    private val responses = (config["responses"] as? List<Map<String, Any?>>).orEmpty()
        .map { LLMResponse.fromMap(it) }
        .ifEmpty { error("Invalid MockLLM Configuration. No responses.") }
    private var cur = 0

    override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?): LLMResponse =
        responses[cur].also { cur = (cur + 1) % responses.size }

    override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?): Flow<StreamFrame> = flow {
        val response = generate(messages, options)
        response.thinking?.let { emit(StreamFrame.Thinking(it)) }
        val text = response.content
        var i = 0
        while (i < text.length) {
            if (streamingDelay > 0) delay(streamingDelay.toLong())
            emit(StreamFrame.Text(text.substring(i, minOf(i + chunkSize, text.length))))
            i += chunkSize
        }
    }
}

/** Fans out to several [ToolProviderProtocol]s, routing by tool name. */
class ToolCollector(
    val toolProviders: List<ToolProviderProtocol>,
    @Suppress("unused_constructor_parameter") val config: Map<String, Any?> = emptyMap(),
) : ToolProviderProtocol {

    private var cache: Map<String, ToolProviderProtocol> = emptyMap()

    override suspend fun listTools(): List<Tool> {
        cache = emptyMap()
        val out = mutableListOf<Tool>()
        for (provider in toolProviders) {
            val tools = provider.listTools()
            out += tools
            tools.forEach { cache = cache + (it.name to provider) }
        }
        return out
    }

    override suspend fun executeTool(toolName: String, parameters: Map<String, Any?>): ToolResult {
        val provider = cache[toolName] ?: run { listTools(); cache[toolName] }
        return provider?.executeTool(toolName, parameters)
            ?: ToolResult(success = false, error = "Could not find $toolName")
    }
}

/** Logs every event; handy with `KAIBO_DEBUG=1`. */
class DebugEventListener {
    fun handleEvent(event: Event) {
        when (event.eventType) {
            EventType.CALL -> println("CALL [${event.callId}] ${event.moduleClass}.${event.methodName}() - Args: ${event.arguments}")
            EventType.RESULT -> println("RESULT [${event.callId}] ${event.moduleClass}.${event.methodName}() -> ${event.result}")
            else -> println("EVENT ${event.eventName} - $event")
        }
    }
}

fun registerDebugListener(registry: Registry, prefix: String = "", agentId: String? = null) {
    val listener = DebugEventListener()
    registry.registerEventListener(prefix, listener::handleEvent, agentId)
}
