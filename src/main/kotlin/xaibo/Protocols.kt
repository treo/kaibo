package xaibo

import kotlinx.coroutines.flow.Flow

/** Protocol for interacting with LLM models */
interface LLMProtocol {
    suspend fun generate(messages: List<LLMMessage>, options: LLMOptions? = null): LLMResponse

    /** Generate a streaming response; chunks arrive as text pieces */
    fun generateStream(messages: List<LLMMessage>, options: LLMOptions? = null): Flow<String>
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

interface ImageMessageHandlerProtocol {
    suspend fun handleImage(image: ByteArray)
}

interface AudioMessageHandlerProtocol {
    suspend fun handleAudio(audio: ByteArray)
}

interface VideoMessageHandlerProtocol {
    suspend fun handleVideo(video: ByteArray)
}
