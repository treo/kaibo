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
 * LLM over the OpenAI chat-completions wire format — works against
 * api.openai.com and any OpenAI-compatible gateway (`base_url` config).
 *
 * Config: `api_key` (or OPENAI_API_KEY env), `model` (default gpt-4.1-nano),
 * `base_url`, `timeout` (seconds). Any other config keys are merged into
 * every request body as defaults.
 */
class OpenAILLM(val config: Map<String, Any?> = emptyMap()) : LLMProtocol {

    private val apiKey = (config["api_key"] as? String) ?: System.getenv("OPENAI_API_KEY")
        ?: error("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
    private val model = config["model"] as? String ?: "gpt-4.1-nano"
    private val baseUrl = (config["base_url"] as? String ?: "https://api.openai.com/v1").trimEnd('/')
    private val timeout = ((config["timeout"] as? Number)?.toDouble() ?: 60.0).toLong()
    private val defaultKwargs = config - setOf("api_key", "model", "base_url", "timeout")

    private val client: HttpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(timeout)).build()
    private val json = Json { isLenient = true; ignoreUnknownKeys = true }

    override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?): LLMResponse {
        val body = buildBody(messages, options, stream = false)
        val resp = withContext(Dispatchers.IO) {
            client.send(httpRequest(body, "application/json"), HttpResponse.BodyHandlers.ofString())
        }
        val root = json.parseToJsonElement(resp.body()).asObject()
            ?: error("LLM API returned a non-object completion: ${resp.body().take(500)}")

        // Fail loudly when a gateway hides an upstream error behind HTTP 200
        val choices = root["choices"].asArray()
        if (choices.isNullOrEmpty())
            error("LLM API returned a completion without choices: ${resp.body().take(500)}")

        val choice = choices[0].asObject() ?: error("malformed choice")
        val message = choice["message"].asObject()
        val finishReason = choice.str("finish_reason")

        var truncatedToolCalls: List<Map<String, Any?>> = emptyList()
        val toolCalls = message.arr("tool_calls")?.mapNotNull { tc ->
            val fn = tc.asObject()?.obj("function")
            if (finishReason == "length") {
                // arguments are almost certainly incomplete JSON; don't execute
                truncatedToolCalls += mapOf("name" to fn?.str("name"), "raw_arguments" to fn?.str("arguments"))
                null
            } else parseToolArguments(fn?.str("arguments"))?.let {
                LLMFunctionCall(tc.asObject()?.str("id") ?: "", fn?.str("name") ?: "", it)
            }
        }?.ifEmpty { null }

        var content = message?.str("content") ?: ""
        if (truncatedToolCalls.isNotEmpty()) {
            val names = truncatedToolCalls.joinToString(", ") { it["name"].toString() }
            content = "[Error: LLM response was truncated due to token limit. Tool call(s) to $names " +
                "had incomplete arguments and could not be executed. Consider increasing max_tokens " +
                "or reducing the conversation context.]"
        }

        val usage = root.obj("usage")?.let { u ->
            LLMUsage(
                u.getInt("prompt_tokens"), u.getInt("completion_tokens"), u.getInt("total_tokens"),
                u.obj("prompt_tokens_details").getIntOrZero("cached_tokens"),
            )
        }

        return LLMResponse(
            content, toolCalls, usage,
            buildMap {
                put("id", root.str("id"))
                put("model", root.str("model"))
                put("finish_reason", finishReason)
                if (truncatedToolCalls.isNotEmpty()) put("truncated_tool_calls", truncatedToolCalls)
            },
        )
    }

    override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?): Flow<String> = flow {
        val body = buildBody(messages, options, stream = true)
        val resp = client.send(httpRequest(body, "text/event-stream"), HttpResponse.BodyHandlers.ofLines())
        resp.body().use { lines ->
            for (line in lines) {
                if (!line.startsWith("data:")) continue
                val payload = line.substring(5).trim()
                if (payload.isEmpty() || payload == "[DONE]") continue
                val delta = json.parseToJsonElement(payload).asObject()
                    ?.arr("choices")?.firstOrNull()?.asObject()?.obj("delta")
                delta?.str("content")?.let { emit(it) }
            }
        }
    }.flowOn(Dispatchers.IO)

    // -- wire format -------------------------------------------------------------

    private fun httpRequest(body: String, accept: String) = HttpRequest.newBuilder(URI("$baseUrl/chat/completions"))
        .timeout(Duration.ofSeconds(timeout))
        .header("Authorization", "Bearer $apiKey")
        .header("Content-Type", "application/json")
        .header("Accept", accept)
        .POST(HttpRequest.BodyPublishers.ofString(body))
        .build()

    private fun buildBody(messages: List<LLMMessage>, options: LLMOptions?, stream: Boolean): String {
        val o = options ?: LLMOptions()
        val kwargs = linkedMapOf<String, Any?>(
            "model" to model,
            "messages" to prepareMessages(messages),
            "temperature" to o.temperature,
            "top_p" to o.topP,
            "max_tokens" to o.maxTokens,
            "stop" to o.stopSequences,
            "tools" to o.functions?.map { toolToJson(it) },
            // the API's own name for the reasoning dial, so the level travels
            // unchanged to gateways that implement it
            "reasoning_effort" to o.reasoningEffort?.name?.lowercase(),
        )
        kwargs.putAll(defaultKwargs)
        kwargs.putAll(o.vendorSpecific)
        if (stream) kwargs["stream"] = true
        return kwargs.filterValues { it != null }.toJsonElement().toString()
    }

    private fun prepareMessages(messages: List<LLMMessage>): List<Map<String, Any?>> = messages.flatMap { msg ->
        when (msg.role) {
            LLMRole.FUNCTION -> when {
                msg.toolCalls != null -> listOf(mapOf(
                    "role" to "assistant", "content" to "",
                    "tool_calls" to msg.toolCalls.map { tc -> mapOf(
                        "id" to tc.id, "type" to "function",
                        "function" to mapOf("name" to tc.name, "arguments" to tc.arguments.toJsonString())
                    ) },
                ))
                msg.toolResults != null -> msg.toolResults.map {
                    mapOf("role" to "tool", "tool_call_id" to it.id, "content" to it.content)
                }
                else -> {
                    System.err.println("Malformed function message - missing both tool_calls and tool_results")
                    emptyList()
                }
            }
            else -> {
                val content: Any? =
                    if (msg.content.size == 1 && msg.content[0].type == LLMMessageContentType.TEXT) msg.content[0].text
                    else msg.content.map {
                        if (it.type == LLMMessageContentType.TEXT) mapOf("type" to "text", "text" to it.text)
                        else mapOf("type" to "image_url", "image_url" to mapOf("url" to it.image))
                    }
                listOf(buildMap {
                    put("role", msg.role.id)
                    put("content", content)
                    msg.name?.let { put("name", it) }
                })
            }
        }
    }

    private fun toolToJson(tool: Tool) = mapOf(
        "type" to "function",
        "function" to mapOf(
            "name" to tool.name,
            "description" to tool.description,
            "parameters" to mapOf(
                "type" to "object",
                "properties" to tool.parameters.mapValues { (name, p) ->
                    buildMap {
                        put("type", jsonSchemaType(p.type))
                        if (jsonSchemaType(p.type) == "array") put("items", mapOf("type" to "string"))
                        put("description", (p.description ?: "") +
                            (if (p.default != null) " Default: ${p.default}" else ""))
                        p.enum?.let { put("enum", it) }
                    }
                },
                "required" to tool.parameters.filterValues { it.required }.keys.toList(),
            ),
        ),
    )

    private fun jsonSchemaType(t: String) = mapOf(
        "string" to "string", "int" to "integer", "integer" to "integer", "long" to "integer",
        "float" to "number", "double" to "number", "number" to "number", "boolean" to "boolean",
        "list" to "array", "array" to "array", "map" to "object", "object" to "object",
    ).getOrDefault(t.lowercase(), t)
}

// null-tolerant JSON accessors shared by the LLM wire modules
internal fun JsonElement?.asObject(): JsonObject? = this as? JsonObject
internal fun JsonElement?.asArray(): JsonArray? = this as? JsonArray
internal fun JsonElement?.asArrayOrEmpty(): JsonArray = asArray() ?: JsonArray(emptyList())
internal fun JsonObject?.str(key: String): String? =
    this?.get(key)?.let { if (it is JsonNull) null else (it as? JsonPrimitive)?.content }
internal fun JsonObject?.obj(key: String): JsonObject? = this?.get(key).asObject()
internal fun JsonObject?.arr(key: String): JsonArray? = this?.get(key).asArray()
internal fun JsonObject?.getInt(key: String): Int = (this?.get(key) as? JsonPrimitive)?.int ?: error("missing $key")
internal fun JsonObject?.getIntOrZero(key: String): Int = runCatching { getInt(key) }.getOrDefault(0)
internal fun Map<String, Any?>.toJsonString(): String = this.toJsonElement().toString()
