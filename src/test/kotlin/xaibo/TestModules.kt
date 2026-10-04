package xaibo

import xaibo.primitives.*

/** Entry that just echoes what the LLM said — the smallest fully-wired module set. */
class EchoEntry(
    val llm: LLMProtocol,
    val response: ResponseProtocol,
    val config: Map<String, Any?> = emptyMap(),
) : TextMessageHandlerProtocol {
    override suspend fun handleText(text: String) {
        response.respondText("echo:${llm.generate(listOf(LLMMessage.user(text))).content}")
    }
}

/** Entry that consumes the LLM stream — exercises Flow/YIELD events. */
class StreamEchoEntry(
    val llm: LLMProtocol,
    val response: ResponseProtocol,
    val config: Map<String, Any?> = emptyMap(),
) : TextMessageHandlerProtocol {
    override suspend fun handleText(text: String) {
        val sb = StringBuilder()
        llm.generateStream(listOf(LLMMessage.user(text))).collect { frame ->
            (frame as? StreamFrame.Text)?.let { sb.append(it.text) }
        }
        response.respondText(sb.toString())
    }
}

class FakeTools(val config: Map<String, Any?> = emptyMap()) : ToolProviderProtocol {
    override suspend fun listTools() = listOf(
        Tool("add", "adds two integers", mapOf(
            "a" to ToolParameter("integer", required = true),
            "b" to ToolParameter("integer", required = true),
        ))
    )

    override suspend fun executeTool(toolName: String, parameters: Map<String, Any?>): ToolResult {
        if (toolName != "add") return ToolResult(false, error = "Could not find $toolName")
        val a = (parameters["a"] as? Number)?.toInt() ?: return ToolResult(false, error = "missing a")
        val b = (parameters["b"] as? Number)?.toInt() ?: return ToolResult(false, error = "missing b")
        if (config["fail"] == true) return ToolResult(false, error = "boom")
        return ToolResult(true, result = a + b)
    }
}

/** LLM that records the options and conversations of every call, cycling fixed responses. */
class ScriptedLLM(
    val responses: List<LLMResponse>,
    val seen: MutableList<LLMOptions> = mutableListOf(),
    val seenConversations: MutableList<List<LLMMessage>> = mutableListOf(),
    override val streams: Boolean = true,
) : LLMProtocol {
    private var cur = 0
    override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?): LLMResponse {
        seen += options ?: LLMOptions()
        seenConversations += messages.toList()
        return responses[cur++ % responses.size]
    }

    override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?) =
        kotlinx.coroutines.flow.flow<StreamFrame> { emit(StreamFrame.Text(generate(messages, options).content)) }
}

enum class Precision { LOW, HIGH }

class CalcTools {
    @XaiboTool(name = "multiply", description = "multiplies two integers")
    fun multiply(@XaiboParam("First factor") a: Int, @XaiboParam("Second factor") b: Int) = a * b

    @XaiboTool(name = "pi", description = "the value of pi at the requested precision")
    fun pi(precision: Precision) = if (precision == Precision.HIGH) 3.14159 else 3.14
}

class SlowTools {
    @XaiboTool(name = "nap", description = "Sleeps for the given milliseconds, then reports")
    fun nap(@XaiboParam("milliseconds to sleep") ms: Int): String {
        Thread.sleep(ms.toLong())
        return "woke after $ms"
    }
}

abstract class GreeterBase {
    @XaiboTool(name = "greet", description = "greets someone")
    open fun greet(@XaiboParam("Name to greet") name: String): String = "hi $name"

    @XaiboTool(description = "must not be visible") private fun secretTool() = "no"
}

class LoudGreeter : GreeterBase() {
    override fun greet(name: String) = "HI $name!!"
}

/** Streams prepared [StreamFrame] rounds; records every conversation it is shown. */
class FrameLLM(private val rounds: List<List<StreamFrame>>) : LLMProtocol {
    private var call = 0
    val seenConversations = mutableListOf<List<LLMMessage>>()
    private fun next() = rounds[minOf(call, rounds.lastIndex)].also { call++ }

    override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?): LLMResponse {
        seenConversations += messages
        return StreamAssembler().apply { next().forEach { feed(it) } }.response()
    }

    override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?) =
        kotlinx.coroutines.flow.flow<StreamFrame> {
            seenConversations += messages
            next().forEach { emit(it) }
        }
}

class RecordingResponse(
    val texts: MutableList<String> = mutableListOf(),
    val events: MutableList<Any> = mutableListOf(),
) : ResponseProtocol {
    override suspend fun getResponse() = Response()
    override suspend fun respondText(response: String) { texts += response }
    override suspend fun respondEvent(event: Any) { events += event }
    override suspend fun respondImage(bytes: ByteArray) {}
    override suspend fun respondAudio(bytes: ByteArray) {}
    override suspend fun respondFile(bytes: ByteArray) {}
    override suspend fun respond(response: Response) { response.text?.let { texts += it } }
}

open class RecordingHistory : ConversationHistoryProtocol {
    val added = mutableListOf<LLMMessage>()
    override suspend fun getHistory() = added.toList()
    override suspend fun addMessage(message: LLMMessage) { added += message }
    override suspend fun clearHistory() { added.clear() }
}

class CompactingHistory : RecordingHistory(), CompactionProtocol {
    var compactions = 0
    override suspend fun compactIfNeeded(llm: LLMProtocol) { compactions++ }
}
