package xaibo

/**
 * One frame of a streamed LLM answer — the vocabulary that travels
 * `LLMProtocol.generateStream` (and, per frame, the event bus as a YIELD
 * event, so frontends render live from events alone).
 *
 * Thinking is NOT content: it is what the model reasoned, not what it said;
 * it never mixes into [Text] and surfaces as the settled response's first-class
 * [LLMResponse.thinking] field (see [StreamAssembler]).
 */
sealed class StreamFrame {
    data class Text(val text: String) : StreamFrame()
    data class Thinking(val text: String) : StreamFrame()
    data class ToolCall(val call: LLMFunctionCall) : StreamFrame()
    data class Usage(val usage: LLMUsage) : StreamFrame()

    /** Terminal frame: carries whatever the wire knows at close (finish_reason, ids) */
    data class Done(val vendorSpecific: Map<String, Any?> = emptyMap()) : StreamFrame()
}

/**
 * Rebuilds the settled [LLMResponse] from stream frames — used by streaming
 * orchestrators (live) and by anything replaying journaled YIELD events.
 * Providers that only emit baseline [StreamFrame.Text] chunks assemble fine.
 */
class StreamAssembler {
    private val content = StringBuilder()
    private val thinking = StringBuilder()
    private val toolCalls = mutableListOf<LLMFunctionCall>()
    private var usage: LLMUsage? = null
    private val vendor = linkedMapOf<String, Any?>("streamed" to true)

    fun feed(frame: StreamFrame) = when (frame) {
        is StreamFrame.Text -> content.append(frame.text)
        is StreamFrame.Thinking -> thinking.append(frame.text)
        is StreamFrame.ToolCall -> toolCalls.add(frame.call)
        is StreamFrame.Usage -> usage = frame.usage
        is StreamFrame.Done -> vendor.putAll(frame.vendorSpecific)
    }

    fun response() = LLMResponse(
        content = content.toString(),
        toolCalls = toolCalls.ifEmpty { null },
        usage = usage,
        thinking = thinking.toString().ifEmpty { null },
        vendorSpecific = vendor,
    )
}
