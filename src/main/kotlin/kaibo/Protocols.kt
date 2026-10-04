package kaibo

import kotlinx.coroutines.flow.Flow

/** Protocol for interacting with LLM models */
interface LLMProtocol {
    /** Wrappers that cannot stream should publish `streams = false` so
     * callers fall back to [generate] instead of attempting the flow. */
    val streams: Boolean get() = true

    suspend fun generate(messages: List<LLMMessage>, options: LLMOptions? = null): LLMResponse

    /** Generate a streaming response as [StreamFrame]s (each frame is also
     * observable on the event bus as a YIELD through the module proxy). */
    fun generateStream(messages: List<LLMMessage>, options: LLMOptions? = null): Flow<StreamFrame>
}

/** Protocol for accessing conversation history, oldest message first */
interface ConversationHistoryProtocol {
    suspend fun getHistory(): List<LLMMessage>
    suspend fun addMessage(message: LLMMessage)
    suspend fun clearHistory()
}

/** Protocol for providing and executing tools */
interface ToolProviderProtocol {
    suspend fun listTools(): List<Tool>
    suspend fun executeTool(toolName: String, parameters: Map<String, Any?>): ToolResult
}

/**
 * Protocol for sending responses. `respondText` appends; `getResponse` hands
 * off the accumulated response and resets turn-scoped events.
 */
interface ResponseProtocol {
    suspend fun getResponse(): Response
    suspend fun respondText(response: String)
    suspend fun respondImage(bytes: ByteArray)
    suspend fun respondAudio(bytes: ByteArray)
    suspend fun respondFile(bytes: ByteArray)
    suspend fun respond(response: Response)

    /** Report a structured event produced while generating the response */
    suspend fun respondEvent(event: Any)
}

// Message handler protocols -----------------------------------------------------

interface TextMessageHandlerProtocol {
    suspend fun handleText(text: String)
}

/**
 * A message handler that accepts steering: user messages queued WHILE a turn
 * is running. The running loop drains them at the earliest point the
 * conversation grammar allows (after a tool-call batch is fully answered);
 * a steered message can never be lost or half-taken.
 */
interface SteeringProtocol {
    fun steer(text: String)
}

/**
 * A conversation history that can compact itself (typically by asking an LLM
 * to summarize older turns) before a turn begins. Implement it on a history
 * module to get compaction for free from orchestrators that check for it.
 */
interface CompactionProtocol {
    suspend fun compactIfNeeded(llm: LLMProtocol)
}

interface ImageMessageHandlerProtocol {
    suspend fun handleImage(image: ByteArray)
}

interface AudioMessageHandlerProtocol {
    suspend fun handleAudio(audio: ByteArray)
}

interface VideoMessageHandlerProtocol {
    suspend fun handleVideo(video: ByteArray)
}
