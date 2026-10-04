package kaibo

/** Roles for LLM messages */
enum class LLMRole(val id: String) {
    SYSTEM("system"), USER("user"), ASSISTANT("assistant"), FUNCTION("function");

    companion object {
        fun of(id: String) = entries.first { it.id == id }
    }
}

enum class LLMMessageContentType(val id: String) { TEXT("text"), IMAGE("image") }

data class LLMFunctionCall(val id: String, val name: String, val arguments: Map<String, Any?> = emptyMap())

data class LLMFunctionResult(val id: String, val name: String, val content: String)

data class LLMMessageContent(val type: LLMMessageContentType, val text: String? = null, val image: String? = null)

/** A message in an LLM conversation */
data class LLMMessage(
    val role: LLMRole,
    val content: List<LLMMessageContent> = emptyList(),
    val name: String? = null,
    val toolCalls: List<LLMFunctionCall>? = null,
    val toolResults: List<LLMFunctionResult>? = null,
) {
    companion object {
        fun system(text: String, name: String? = null) =
            LLMMessage(LLMRole.SYSTEM, listOf(LLMMessageContent(LLMMessageContentType.TEXT, text = text)), name)

        fun user(text: String, name: String? = null) =
            LLMMessage(LLMRole.USER, listOf(LLMMessageContent(LLMMessageContentType.TEXT, text = text)), name)

        fun userImage(dataUri: String, name: String? = null) =
            LLMMessage(LLMRole.USER, listOf(LLMMessageContent(LLMMessageContentType.IMAGE, image = dataUri)), name)

        fun assistant(text: String, name: String? = null) =
            LLMMessage(LLMRole.ASSISTANT, listOf(LLMMessageContent(LLMMessageContentType.TEXT, text = text)), name)

        fun function(id: String, name: String, arguments: Map<String, Any?>) = LLMMessage(
            role = LLMRole.FUNCTION, content = emptyList(), name = name,
            toolCalls = listOf(LLMFunctionCall(id, name, arguments)),
        )

        fun functionResult(id: String, name: String, content: String) = LLMMessage(
            role = LLMRole.FUNCTION, content = emptyList(), name = name,
            toolResults = listOf(LLMFunctionResult(id, name, content)),
        )
    }
}

/**
 * How much thinking a reasoning model may do before it answers. Provider
 * modules own how a level reaches the wire; three rules they owe this enum:
 * unset sends nothing, an unexpressible level is lowered never dropped, and a
 * model without an off switch has no NONE.
 */
enum class ReasoningEffort { NONE, MINIMAL, LOW, MEDIUM, HIGH, XHIGH, MAX }

/** Common options for LLM requests */
data class LLMOptions(
    val temperature: Double? = 1.0,
    val topP: Double? = 1.0,
    val maxTokens: Int? = null,
    val stopSequences: List<String>? = null,
    val functions: List<Tool>? = null,
    val reasoningEffort: ReasoningEffort? = null,
    val vendorSpecific: Map<String, Any?> = emptyMap(),
) {
    init {
        require(temperature == null || temperature in 0.0..2.0) { "temperature must be between 0 and 2" }
        require(topP == null || topP in 0.0..1.0) { "top_p must be between 0 and 1" }
    }
}

/**
 * Token usage statistics from an LLM response. `cachedTokens` is the subset of
 * `promptTokens` served from the provider's prompt cache — never extra spend;
 * `totalTokens` stays prompt plus completion.
 */
data class LLMUsage(
    val promptTokens: Int,
    val completionTokens: Int,
    val totalTokens: Int,
    val cachedTokens: Int = 0,
)

/** Response from an LLM */
data class LLMResponse(
    val content: String,
    val toolCalls: List<LLMFunctionCall>? = null,
    val usage: LLMUsage? = null,
    /** What a reasoning model thought on the way to this answer, when the
     * provider reports it. Not content: the answer stays the answer. */
    val thinking: String? = null,
    val vendorSpecific: Map<String, Any?> = emptyMap(),
) {
    companion object {
        /** Merge multiple LLM responses into a single response */
        fun merge(responses: List<LLMResponse>) = LLMResponse(
            content = responses.joinToString("\n") { it.content },
            toolCalls = responses.flatMap { it.toolCalls ?: emptyList() }.ifEmpty { null },
            usage = responses.mapNotNull { it.usage }.reduceOrNull { a, b ->
                LLMUsage(
                    a.promptTokens + b.promptTokens,
                    a.completionTokens + b.completionTokens,
                    a.totalTokens + b.totalTokens,
                    a.cachedTokens + b.cachedTokens,
                )
            },
            thinking = responses.mapNotNull { it.thinking }.joinToString("\n").ifEmpty { null },
            vendorSpecific = responses.fold(emptyMap()) { acc, r -> acc + r.vendorSpecific },
        )

        /** Build from a map, e.g. MockLLM responses parsed from YAML config */
        fun fromMap(m: Map<String, Any?>): LLMResponse {
            fun str(k: String) = m[k] as? String
            val usage = (m["usage"] as? Map<String, Any?>)?.let { u ->
                LLMUsage(
                    (u["prompt_tokens"] as? Number)?.toInt() ?: 0,
                    (u["completion_tokens"] as? Number)?.toInt() ?: 0,
                    (u["total_tokens"] as? Number)?.toInt() ?: 0,
                    (u["cached_tokens"] as? Number)?.toInt() ?: 0,
                )
            }
            val toolCalls = (m["tool_calls"] as? List<Map<String, Any?>>)?.map {
                LLMFunctionCall(it["id"] as? String ?: "", it["name"] as? String ?: "",
                    (it["arguments"] as? Map<String, Any?>) ?: emptyMap())
            }
            return LLMResponse(str("content") ?: "", toolCalls, usage,
                thinking = str("thinking"),
                vendorSpecific = (m["vendor_specific"] as? Map<String, Any?>) ?: emptyMap(),
            )
        }
    }
}

enum class FileType { IMAGE, AUDIO, FILE }

/** Model for file attachments in responses */
data class FileAttachment(val content: ByteArray, val type: FileType) {
    constructor(content: ByteArray, typeId: String) : this(content, FileType.valueOf(typeId.uppercase()))
}

// Response events -------------------------------------------------------------

/** A tool is about to be executed */
data class ToolCallEvent(val id: String, val name: String, val arguments: Map<String, Any?>) {
    val type = "tool_call"
}

/** A tool execution finished */
data class ToolResultEvent(
    val id: String,
    val name: String,
    val success: Boolean,
    val result: Any? = null,
    val error: String? = null,
) {
    val type = "tool_result"
}

/** Token usage reported by an LLM call */
data class UsageEvent(
    val promptTokens: Int,
    val completionTokens: Int,
    val totalTokens: Int,
    val cachedTokens: Int = 0,
) {
    val type = "usage"

    companion object {
        operator fun invoke(usage: LLMUsage) =
            UsageEvent(usage.promptTokens, usage.completionTokens, usage.totalTokens, usage.cachedTokens)
    }
}

/** A reasoning model's thinking, reported separately from the answer */
data class ThinkingEvent(val text: String) {
    val type = "thinking"
}

/**
 * Model for responses that can include text, file attachments and structured
 * events. Events carry the internals of response generation (tool calls, tool
 * results, usage) so adapters can surface them, not just the final text.
 */
class Response(
    var text: String? = null,
    val attachments: MutableList<FileAttachment> = mutableListOf(),
    val events: MutableList<Any> = mutableListOf(),
)

// Tools ------------------------------------------------------------------------

/** Parameter definition for a tool */
data class ToolParameter(
    val type: String,
    val description: String? = null,
    val required: Boolean = false,
    val default: Any? = null,
    val enum: List<String>? = null,
)

/** Definition of a tool that can be executed */
data class Tool(val name: String, val description: String, val parameters: Map<String, ToolParameter> = emptyMap())

/** Result of a tool execution */
data class ToolResult(val success: Boolean, val result: Any? = null, val error: String? = null)

// Events -----------------------------------------------------------------------

enum class EventType(val value: String) { CALL("call"), RESULT("result"), EXCEPTION("exception"), YIELD("yield") }

/** A method invocation event emitted by module proxies */
data class Event(
    val agentId: String?,
    val eventName: String,
    val eventType: EventType,
    val moduleId: String?,
    val moduleClass: String,
    val methodName: String,
    val time: Double,
    val callId: String,
    val callerId: String?,
    val arguments: Map<String, Any?>? = null,
    val result: Any? = null,
    val exception: String? = null,
)
