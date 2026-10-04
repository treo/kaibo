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
 * LLM over Google's Gemini REST API (generativelanguage v1beta).
 *
 * Config: `api_key` (or GEMINI_API_KEY/GOOGLE_API_KEY env), `model` (default
 * gemini-2.0-flash-001), `base_url`, `timeout`, `reasoning_mode` ("level" for
 * the thinkingLevel enum on Gemini 3+, "budget" for thinking_budget on 2.5
 * and earlier); other keys merge into every generationConfig.
 */
class GoogleLLM(val config: Map<String, Any?> = emptyMap()) : LLMProtocol {

    private val apiKey = (config["api_key"] as? String)
        ?: System.getenv("GEMINI_API_KEY") ?: System.getenv("GOOGLE_API_KEY")
        ?: error("Google API key must be provided or set as GEMINI_API_KEY environment variable")
    private val model = config["model"] as? String ?: "gemini-2.0-flash-001"
    private val baseUrl = (config["base_url"] as? String ?: "https://generativelanguage.googleapis.com/v1beta").trimEnd('/')
    private val timeout = ((config["timeout"] as? Number)?.toDouble() ?: 60.0).toLong()
    private val reasoningMode = config["reasoning_mode"] as? String ?: "level"
    private val defaultKwargs = config - setOf("api_key", "model", "base_url", "timeout", "reasoning_mode")

    private val client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(timeout)).build()
    private val json = Json { isLenient = true; ignoreUnknownKeys = true }

    // Gemini's thinking ladder stops at HIGH; xhigh/max take HIGH, minimal takes
    // LOW, and `none` takes LOW too — Gemini 3 cannot disable thinking at all.
    private val levels = mapOf(
        "none" to "LOW", "minimal" to "LOW", "low" to "LOW",
        "medium" to "MEDIUM", "high" to "HIGH", "xhigh" to "HIGH", "max" to "HIGH",
    )
    private val budgets = mapOf("minimal" to 1024, "low" to 2048, "medium" to 8192, "high" to 16384, "xhigh" to 16384, "max" to 16384)

    override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?): LLMResponse {
        val body = buildBody(messages, options ?: LLMOptions(), stream = false)
        val resp = withContext(Dispatchers.IO) {
            client.send(httpRequest(":generateContent", body), HttpResponse.BodyHandlers.ofString())
        }
        return parseResponse(json.parseToJsonElement(resp.body()).asObject()
            ?: error("Gemini returned a non-object response: ${resp.body().take(500)}"))
    }

    override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?): Flow<String> = flow {
        val body = buildBody(messages, options ?: LLMOptions(), stream = true)
        val resp = client.send(httpRequest(":streamGenerateContent?alt=sse", body), HttpResponse.BodyHandlers.ofLines())
        resp.body().use { lines ->
            for (line in lines) {
                if (!line.startsWith("data:")) continue
                val root = runCatching { json.parseToJsonElement(line.substring(5).trim()).asObject() }.getOrNull()
                root?.arr("candidates")?.firstOrNull()?.asObject()?.obj("content")?.arr("parts")
                    ?.mapNotNull { it.asObject()?.str("text") }
                    ?.forEach { emit(it) }
            }
        }
    }.flowOn(Dispatchers.IO)

    // -- wire format -------------------------------------------------------------

    private fun httpRequest(verb: String, body: String) = HttpRequest.newBuilder(URI("$baseUrl/models/$model$verb"))
        .timeout(Duration.ofSeconds(timeout))
        .header("x-goog-api-key", apiKey)
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(body))
        .build()

    private fun buildBody(messages: List<LLMMessage>, opts: LLMOptions, stream: Boolean): String {
        val generationConfig = linkedMapOf<String, Any?>(
            "temperature" to opts.temperature,
            "topP" to opts.topP,
            "maxOutputTokens" to opts.maxTokens,
            "stopSequences" to opts.stopSequences,
        )
        generationConfig.putAll(defaultKwargs)
        generationConfig.putAll(opts.vendorSpecific)

        // the reasoning dial in this model's shape; unset sends nothing
        opts.reasoningEffort?.let { e ->
            val level = e.name.lowercase()
            generationConfig["thinkingConfig"] = if (reasoningMode == "level")
                mapOf("thinkingLevel" to (levels[level] ?: "LOW"))
            else mapOf("thinkingBudget" to if (level == "none") 0 else budgets[level] ?: budgets["high"])
        }

        val body = linkedMapOf<String, Any?>("contents" to contents(messages))
        systemText(messages)?.let { body["systemInstruction"] = mapOf("parts" to listOf(mapOf("text" to it))) }
        opts.functions?.takeIf { it.isNotEmpty() }?.let {
            body["tools"] = listOf(mapOf("functionDeclarations" to it.map(::functionDeclaration)))
        }
        body["generationConfig"] = generationConfig.filterValues { it != null }
        return body.filterValues { it != null }.toJsonElement().toString()
    }

    private fun systemText(messages: List<LLMMessage>) = messages
        .filter { it.role == LLMRole.SYSTEM }
        .flatMap { it.content.mapNotNull { c -> c.text } }
        .joinToString(" ").ifEmpty { null }

    private fun contents(messages: List<LLMMessage>): List<Map<String, Any?>> = messages.flatMap { msg ->
        when (msg.role) {
            LLMRole.SYSTEM -> emptyList()
            LLMRole.FUNCTION -> when {
                msg.toolCalls != null -> listOf(mapOf(
                    "role" to "model",
                    "parts" to (msg.content.mapNotNull { c -> c.text?.let { mapOf("text" to it) } } +
                        msg.toolCalls.map {
                            mapOf("functionCall" to mapOf("name" to it.name, "args" to it.arguments))
                        }),
                ))
                msg.toolResults != null -> msg.toolResults.map { r ->
                    mapOf("role" to "user", "parts" to listOf(
                        mapOf("functionResponse" to mapOf("name" to r.name, "response" to mapOf("result" to r.content)))))
                }
                else -> {
                    System.err.println("Malformed function message - missing both tool_calls and tool_results")
                    emptyList()
                }
            }
            else -> listOf(mapOf(
                "role" to if (msg.role == LLMRole.ASSISTANT) "model" else "user",
                "parts" to msg.content.mapNotNull { c ->
                    when (c.type) {
                        LLMMessageContentType.TEXT -> c.text?.let { mapOf("text" to it) }
                        LLMMessageContentType.IMAGE -> c.image?.let { imagePart(it) }
                    }
                },
            ))
        }
    }

    private fun imagePart(image: String): Map<String, Any?>? = when {
        image.startsWith("data:") -> mapOf("inlineData" to mapOf(
            "mimeType" to image.substringAfter(':').substringBefore(';'),
            "data" to image.substringAfter(','),
        ))
        else -> mapOf("fileData" to mapOf(
            "mimeType" to when {
                image.lowercase().endsWith(".png") -> "image/png"
                image.lowercase().endsWith(".gif") -> "image/gif"
                image.lowercase().endsWith(".webp") -> "image/webp"
                else -> "image/jpeg"
            },
            "fileUri" to image,
        ))
    }

    private fun functionDeclaration(tool: Tool) = mapOf(
        "name" to tool.name,
        "description" to tool.description,
        "parameters" to mapOf(
            "type" to "OBJECT",
            "properties" to tool.parameters.mapValues { (_, p) ->
                buildMap {
                    put("type", p.type.uppercase())
                    put("description", (p.description ?: "") +
                        (if (p.default != null) " Default: ${p.default}" else ""))
                    p.enum?.let { put("enum", it) }
                }
            },
            "required" to tool.parameters.filterValues { it.required }.keys.toList(),
        ),
    )

    private fun parseResponse(root: JsonObject): LLMResponse {
        val candidate = root.arr("candidates")?.firstOrNull()?.asObject()
            ?: error("Gemini returned no candidates: ${root.toString().take(500)}")
        val parts = candidate.obj("content").arr("parts").asArrayOrEmpty()
        val text = parts.mapNotNull { it.asObject()?.str("text") }.joinToString("")
        val toolCalls = parts.mapNotNull { p ->
            p.asObject()?.obj("functionCall")?.let { fc ->
                LLMFunctionCall(
                    id = fc.str("name") ?: "", // Gemini has no call ids; the name stands in
                    name = fc.str("name") ?: "",
                    arguments = (fc["args"].asObject())?.toKotlin()?.let { it as? Map<String, Any?> } ?: emptyMap(),
                )
            }
        }.ifEmpty { null }
        val usage = root.obj("usageMetadata")?.let { u ->
            val prompt = u.getIntOrZero("promptTokenCount")
            val completion = u.getIntOrZero("candidatesTokenCount")
            LLMUsage(prompt, completion, u.getIntOrZero("totalTokenCount"), u.getIntOrZero("cachedContentTokenCount"))
        }
        return LLMResponse(text, toolCalls, usage,
            mapOf("finishReason" to candidate.str("finishReason"), "model" to model))
    }
}
