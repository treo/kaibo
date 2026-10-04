package kaibo.server

import kaibo.*
import kaibo.primitives.SimpleConversation
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import java.net.InetSocketAddress

/**
 * OpenAI-compatible HTTP façade over registered agents: clients point any
 * OpenAI chat library at this server and address agents through `model`.
 *
 * GET  /v1/models            -> registered agent entry points
 * POST /v1/chat/completions  -> runs the agent against the last user message
 *
 * The conversation in the request body is injected as the agent's
 * ConversationHistoryProtocol, so stateless clients keep full context.
 */
class OpenAICompatServer(
    private val kaibo: Kaibo,
    private val port: Int = 8000,
    private val apiKey: String? = null,
    private val streamingTimeoutMs: Long = 30_000,
) {

    private lateinit var http: HttpServer

    /** The actually bound port (use port 0 to get an ephemeral one) */
    fun listeningPort(): Int = http.address.port

    fun start(): OpenAICompatServer = apply {
        http = HttpServer.create(InetSocketAddress(port), 0).apply {
            createContext("/v1/models") { ex -> if (authorized(ex)) json(ex, 200, models()) }
            createContext("/v1/chat/completions") { ex ->
                if (!authorized(ex)) return@createContext
                val data = try {
                    readJsonBody(ex)
                } catch (e: Exception) {
                    json(ex, 400, errorJson("invalid request body: ${e.message}"))
                    return@createContext
                }
                try {
                    completion(ex, data)
                } catch (e: Exception) {
                    System.err.println("Unexpected error in completion_request: ${e.stackTraceToString()}")
                    runCatching { json(ex, 500, errorJson(e.message ?: "internal error")) }
                }
            }
            start()
        }
    }

    // -- endpoints ---------------------------------------------------------------

    private fun models(): String {
        val ids = mutableListOf<String>()
        for (agent in kaibo.listAgents()) {
            kaibo.getAgentConfig(agent).exchange.filter { it.module == "__entry__" }.forEach { ex ->
                if (ex.providers.size == 1) ids.add(agent)
                else ex.providers.forEach { ids.add("$agent/$it") }
            }
        }
        return mapOf(
            "object" to "list",
            "data" to ids.map {
                mapOf("id" to it, "object" to "model", "created" to 0, "owned_by" to "organization-owner")
            },
        ).toJsonElement().toString()
    }

    private fun completion(ex: HttpExchange, data: Map<String, Any?>) {
        val messages = (data["messages"] as? List<Map<String, Any?>>).orEmpty()
        val lastUser = messages.lastOrNull { it["role"] == "user" }?.let { m ->
            when (val c = m["content"]) {
                is String -> c
                is List<*> -> c.filterIsInstance<Map<String, Any?>>().mapNotNull { it["text"] as? String }.joinToString(" ")
                else -> null
            }
        } ?: ""
        val (agentId, entryPoint) = parseModel(data.str("model") ?: "")

        // history is the context BEFORE this turn; the orchestrator appends the
        // current user message itself
        val history = if (messages.lastOrNull()?.get("role") == "user") messages.dropLast(1) else messages
        val instances = linkedMapOf<String, Any>("__conversation_history__" to SimpleConversation.fromOpenaiMessages(history))
        val bindings = mutableListOf(
            ExchangeConfig(protocol = "ConversationHistoryProtocol", providers = listOf("__conversation_history__"))
        )
        val stream = data["stream"] == true
        val chunks = if (stream) Channel<String>(Channel.UNLIMITED) else null
        val chunkFn = sseChunkFactory(data.str("model") ?: "")
        if (chunks != null) {
            instances["__response__"] = StreamingResponse(chunks, chunkFn)
            bindings.add(ExchangeConfig(protocol = "ResponseProtocol", providers = listOf("__response__")))
        }

        val agent = try {
            kaibo.getAgentWith(agentId, ConfigOverrides(instances, bindings))
        } catch (e: NoSuchElementException) {
            json(ex, 400, errorJson("model not found"))
            return
        }

        if (chunks != null) {
            streaming(ex, agent, lastUser, entryPoint, chunkFn, chunks)
        } else {
            val response = runBlocking { agent.handleText(lastUser, entryPoint) }
            json(ex, 200, mapOf(
                "id" to "chatcmpl-${randomId()}",
                "object" to "chat.completion",
                "created" to (System.currentTimeMillis() / 1000),
                "choices" to listOf(mapOf(
                    "index" to 0,
                    "message" to mapOf("role" to "assistant", "content" to response.text),
                    "finish_reason" to "stop",
                )),
                "usage" to mapOf("prompt_tokens" to 0, "completion_tokens" to 0, "total_tokens" to 0),
            ).toJsonElement().toString())
        }
    }

    private fun streaming(
        ex: HttpExchange,
        agent: Agent,
        text: String,
        entryPoint: String,
        chunkFn: (Map<String, Any?>, String?) -> String,
        chunks: Channel<String>,
    ) {
        ex.responseHeaders.add("Content-Type", "text/event-stream")
        ex.sendResponseHeaders(200, 0)
        ex.responseBody.use { out ->
            runBlocking {
                out.write(chunkFn(mapOf("content" to ""), null).toByteArray()) // flush headers
                val job = CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
                    try {
                        agent.handleText(text, entryPoint)
                    } catch (e: Exception) {
                        System.err.println("Agent task failed: ${e.stackTraceToString()}")
                    } finally {
                        chunks.close()
                    }
                }
                try {
                    loop@ while (true) {
                        val drained = chunks.tryReceive()
                        if (drained.isSuccess) {
                            out.write(drained.getOrThrow().toByteArray())
                            continue
                        }
                        if (job.isCompleted && drained.isClosed) break@loop
                        val waited = withTimeoutOrNull(streamingTimeoutMs) { chunks.receiveCatching() }
                        when {
                            waited == null -> out.write(chunkFn(mapOf("content" to ""), null).toByteArray()) // keep-alive
                            waited.isSuccess -> out.write(waited.getOrThrow().toByteArray())
                            job.isCompleted -> break@loop // closed and drained
                            else -> {} // still running; loop retries the receive
                        }
                    }
                    out.write(chunkFn(emptyMap(), "stop").toByteArray())
                    out.write("data: [DONE]\n\n".toByteArray())
                } finally {
                    // a client that walks away must not leave the agent
                    // burning LLM calls on its behalf
                    job.cancel()
                }
            }
        }
    }

    private fun sseChunkFactory(model: String): (Map<String, Any?>, String?) -> String {
        val id = "chatcmpl-${randomId()}"
        return { delta, finish ->
            "data: " + mapOf(
                "id" to id,
                "object" to "chat.completion.chunk",
                "created" to (System.currentTimeMillis() / 1000),
                "model" to model,
                "choices" to listOf(mapOf("delta" to delta, "finish_reason" to finish, "index" to 0)),
            ).toJsonElement().toString() + "\n\n"
        }
    }

    // -- plumbing ----------------------------------------------------------------

    private fun authorized(ex: HttpExchange): Boolean {
        if (apiKey == null) return true
        if (ex.requestHeaders["Authorization"]?.any { it == "Bearer $apiKey" } == true) return true
        json(ex, 401, errorJson("invalid api key"))
        return false
    }

    private fun json(ex: HttpExchange, code: Int, body: String) {
        val bytes = body.toByteArray()
        ex.responseHeaders.add("Content-Type", "application/json")
        ex.sendResponseHeaders(code, bytes.size.toLong())
        ex.responseBody.use { it.write(bytes) }
    }

    private fun errorJson(message: String) =
        mapOf("error" to mapOf("message" to message, "type" to "invalid_request_error")).toJsonElement().toString()

    private fun randomId() = (1..16).map { "abcdefghijklmnopqrstuvwxyz0123456789".random() }.joinToString("")

    private fun parseModel(model: String): Pair<String, String> =
        if ('/' in model) model.substringBefore('/') to model.substringAfter('/') else model to "__entry__"

    /** The response lane for streaming: text parts become SSE chunks */
    private class StreamingResponse(
        private val chunks: Channel<String>,
        private val chunkFn: (Map<String, Any?>, String?) -> String,
    ) : ResponseProtocol {
        override suspend fun getResponse(): Response = Response()
        override suspend fun respondText(text: String) { chunks.send(chunkFn(mapOf("content" to text), null)) }
        override suspend fun respondImage(bytes: ByteArray) {}
        override suspend fun respondAudio(bytes: ByteArray) {}
        override suspend fun respondFile(bytes: ByteArray) {}
        override suspend fun respond(response: Response) { response.text?.let { respondText(it) } }
        // the OpenAI chat completion grammar has no lane for agent internals
        override suspend fun respondEvent(event: Any) {}
    }

    companion object {
        fun readJsonBody(ex: HttpExchange): Map<String, Any?> =
            kotlinx.serialization.json.Json.parseToJsonElement(ex.requestBody.readBytes().decodeToString())
                .toKotlin() as Map<String, Any?>
    }
}

fun main(args: Array<String>) {
    var agentDir = "agents"
    var port = 8000
    var apiKey: String? = null
    var i = 0
    while (i < args.size) {
        when (args[i]) {
            "--agents" -> agentDir = args[++i]
            "--port" -> port = args[++i].toInt()
            "--api-key" -> apiKey = args[++i]
            else -> System.err.println("ignoring argument ${args[i]}")
        }
        i++
    }
    val kaibo = Kaibo()
    val configs = AgentConfig.loadDirectory(agentDir)
    configs.values.forEach { kaibo.registerAgent(it) }
    OpenAICompatServer(kaibo, port, apiKey).start()
    println("kaibo serving ${configs.values.map { it.id }} on http://localhost:$port/v1")
}
