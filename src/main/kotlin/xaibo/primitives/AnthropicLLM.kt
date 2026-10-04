package xaibo.primitives

import xaibo.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.*
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

/**
 * How Claude says "think this much". Anthropic changed the spelling between
 * generations, so the shape is a config choice, not derived from the level:
 * `effort` (adaptive thinking, Claude 4.6+; the only shape 4.7+ accepts) or
 * `budget` (4.5 and earlier: thinking.budget_tokens).
 *
 * Rules from [xaibo.ReasoningEffort]: unset sends nothing; a level the model
 * cannot express is lowered, never dropped.
 */
internal fun thinkingKwargs(effort: ReasoningEffort?, mode: String, maxTokens: Int?): Map<String, Any?> {
    if (effort == null) return emptyMap()
    val level = effort.name.lowercase()
    if (level == "none") return mapOf("thinking" to mapOf("type" to "disabled"))
    if (mode == "effort") {
        val ladder = setOf("low", "medium", "high", "xhigh", "max")
        return mapOf(
            "thinking" to mapOf("type" to "adaptive"),
            "output_config" to mapOf("effort" to if (level in ladder) level else "low"),
        )
    }
    // budget shape: depth is only a token count; rungs above high share its budget
    val budgets = mapOf("minimal" to 1024, "low" to 2048, "medium" to 8192, "high" to 16384, "xhigh" to 16384, "max" to 16384)
    val budget = budgets[level] ?: budgets["high"]!!
    val out = mutableMapOf<String, Any?>("thinking" to mapOf("type" to "enabled", "budget_tokens" to budget))
    // thinking bills against max_tokens; a level may only ever raise the ceiling
    if (maxTokens == null || maxTokens <= budget) out["max_tokens"] = budget + 1024
    return out
}

/**
 * LLM over the Anthropic Messages API.
 *
 * Config: `api_key` (or ANTHROPIC_API_KEY env), `model` (default
 * claude-3-opus-20240229), `base_url`, `timeout`, `reasoning_mode`
 * ("effort"|"budget"); other keys merge into every request body.
 */
class AnthropicLLM(val config: Map<String, Any?> = emptyMap()) : LLMProtocol {

    private val apiKey = (config["api_key"] as? String) ?: System.getenv("ANTHROPIC_API_KEY")
        ?: error("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")
    private val model = config["model"] as? String ?: "claude-3-opus-20240229"
    private val baseUrl = (config["base_url"] as? String ?: "https://api.anthropic.com").trimEnd('/')
    private val timeout = ((config["timeout"] as? Number)?.toDouble() ?: 60.0).toLong()
    private val reasoningMode = config["reasoning_mode"] as? String ?: "effort"
    private val defaultKwargs = config - setOf("api_key", "model", "base_url", "timeout", "reasoning_mode")

    private val client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(timeout)).build()
    private val json = Json { isLenient = true; ignoreUnknownKeys = true }

    override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?): LLMResponse {
        val opts = options ?: LLMOptions()
        val (prepared, system) = prepareMessages(messages)
        val body = requestKwargs(prepared, system, opts, stream = false)
        val resp = withContext(Dispatchers.IO) {
            client.send(httpRequest(json.encodeToString(JsonElement.serializer(), body.toJsonElement())),
                HttpResponse.BodyHandlers.ofString())
        }
        val root = json.parseToJsonElement(resp.body()).jsonObject

        val text = StringBuilder()
        val thinking = StringBuilder()
        var toolCall: LLMFunctionCall? = null
        for (item in root["content"].asArrayOrEmpty()) {
            val o = item.asObject() ?: continue
            when (o.str("type")) {
                "text" -> o.str("text")?.let(text::append)
                "thinking" -> o.str("thinking")?.let(thinking::append)
                "tool_use" -> toolCall = LLMFunctionCall(o.str("id") ?: "", o.str("name") ?: "",
                    (o["input"].asObject())?.toKotlin()?.let { it as? Map<String, Any?> } ?: emptyMap())
            }
        }

        val usage = root["usage"].asObject()?.let { u ->
            val inp = (u["input_tokens"] as? JsonPrimitive)?.int ?: 0
            val outp = (u["output_tokens"] as? JsonPrimitive)?.int ?: 0
            LLMUsage(inp, outp, inp + outp,
                (u["cache_read_input_tokens"] as? JsonPrimitive)?.int ?: 0)
        }

        return LLMResponse(
            content = text.toString(),
            toolCalls = listOfNotNull(toolCall),
            usage = usage,
            thinking = thinking.toString().ifEmpty { null },
            vendorSpecific = mapOf("id" to root.str("id"), "model" to root.str("model")),
        )
    }

    override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?): Flow<StreamFrame> = flow {
        val opts = options ?: LLMOptions()
        val (prepared, system) = prepareMessages(messages)
        val body = requestKwargs(prepared, system, opts, stream = true)
        val resp = client.send(httpRequest(json.encodeToString(JsonElement.serializer(), body.toJsonElement())),
            HttpResponse.BodyHandlers.ofLines())
        resp.body().use { lines ->
            for (line in lines) {
                if (!line.startsWith("data:")) continue
                val evt = runCatching { json.parseToJsonElement(line.substring(5).trim()).jsonObject }
                    .getOrNull() ?: continue
                if (evt.str("type") == "content_block_delta") {
                    // Anthropic separates answer text from reasoning on the wire;
                    // we keep them in separate frames
                    val delta = evt["delta"].asObject()
                    when (delta?.str("type")) {
                        "text_delta" -> delta.str("text")?.let { emit(StreamFrame.Text(it)) }
                        "thinking_delta" -> delta.str("thinking")?.let { emit(StreamFrame.Thinking(it)) }
                        else -> {}
                    }
                }
            }
        }
    }.flowOn(Dispatchers.IO)

    // -- wire format -------------------------------------------------------------

    private fun httpRequest(body: String) = HttpRequest.newBuilder(URI("$baseUrl/v1/messages"))
        .timeout(Duration.ofSeconds(timeout))
        .header("x-api-key", apiKey)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(body))
        .build()

    private fun requestKwargs(messages: List<Map<String, Any?>>, system: String?, opts: LLMOptions, stream: Boolean): Map<String, Any?> {
        val kwargs = linkedMapOf<String, Any?>(
            "model" to model,
            "messages" to messages,
            "temperature" to opts.temperature,
            "top_p" to opts.topP,
            "max_tokens" to (opts.maxTokens ?: 1024),
        )
        kwargs.putAll(defaultKwargs)
        kwargs.putAll(opts.vendorSpecific)
        if (stream) kwargs["stream"] = true
        if (!system.isNullOrEmpty()) kwargs["system"] = system
        opts.functions?.let { kwargs["tools"] = it.map(::toolToJson) }
        opts.stopSequences?.let { kwargs["stop_sequences"] = it }

        // the reasoning level in this model's shape; explicit values win
        for ((key, value) in thinkingKwargs(opts.reasoningEffort, reasoningMode, opts.maxTokens)) {
            if (key == "max_tokens") kwargs[key] = maxOf(kwargs[key] as Int, value as Int)
            else kwargs.putIfAbsent(key, value)
        }
        return kwargs.filterValues { it != null }
    }

    /** @return messages plus the joined system text */
    private fun prepareMessages(messages: List<LLMMessage>): Pair<List<Map<String, Any?>>, String?> {
        val out = mutableListOf<Map<String, Any?>>()
        var system: String? = null

        for (msg in messages) {
            when (msg.role) {
                LLMRole.SYSTEM -> {
                    val texts = msg.content.mapNotNull { if (it.type == LLMMessageContentType.TEXT) it.text else null }
                    if (texts.isNotEmpty()) system = texts.joinToString(" ")
                }
                LLMRole.FUNCTION -> when {
                    msg.toolCalls != null -> out += mapOf(
                        "role" to "assistant",
                        "content" to (msg.content
                            .filter { it.type == LLMMessageContentType.TEXT }
                            .map { mapOf("type" to "text", "text" to it.text) } +
                            msg.toolCalls.map {
                                mapOf("type" to "tool_use", "id" to it.id, "name" to it.name, "input" to it.arguments)
                            }),
                    )
                    msg.toolResults != null -> for (r in msg.toolResults)
                        out += mapOf("role" to "user", "content" to listOf(
                            mapOf("type" to "tool_result", "tool_use_id" to r.id, "content" to r.content)))
                    else -> System.err.println("Malformed function message - missing both tool_calls and tool_results")
                }
                else -> out += buildMap {
                    put("role", if (msg.role == LLMRole.ASSISTANT) "assistant" else "user")
                    put("content", msg.content.map {
                        if (it.type == LLMMessageContentType.TEXT) mapOf("type" to "text", "text" to it.text)
                        else imageContent(it.image ?: "")
                    })
                    msg.name?.let { put("name", it) }
                }
            }
        }
        return out to system
    }

    private fun imageContent(image: String): Map<String, Any?> =
        if (image.startsWith("data:")) {
            val (mediaType, data) = image.substringAfter(':').split(";base64,", limit = 2)
            mapOf("type" to "image", "source" to mapOf("type" to "base64", "media_type" to mediaType, "data" to data))
        } else mapOf("type" to "image", "source" to mapOf("type" to "url", "url" to image))

    private fun toolToJson(tool: Tool) = mapOf(
        "name" to tool.name,
        "description" to tool.description,
        "input_schema" to mapOf(
            "type" to "object",
            "properties" to tool.parameters.mapValues { (_, p) ->
                buildMap {
                    put("type", p.type.lowercase())
                    if (p.type.lowercase() == "array") put("items", mapOf("type" to "string"))
                    put("description", (p.description ?: "") +
                        (if (p.default != null) " Default: ${p.default}" else ""))
                    p.enum?.let { put("enum", it) }
                }
            },
            "required" to tool.parameters.filterValues { it.required }.keys.toList(),
        ),
    )
}
