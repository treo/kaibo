package kaibo

import kaibo.primitives.*
import kaibo.server.OpenAICompatServer
import kotlin.test.*
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import kotlinx.serialization.json.*

/** The OpenAI-compatible surface, end to end over real HTTP. */
class ServerTest {

    private fun client(port: Int) = HttpClient.newHttpClient()

    private fun chatBody(model: String, text: String = "hi", stream: Boolean = false) = buildString {
        append("""{"model":"$model","messages":[{"role":"system","content":"shh"},{"role":"user","content":"$text"}]""")
        if (stream) append(""","stream":true""")
        append("}")
    }

    private fun request(port: Int, path: String, body: String? = null, auth: String? = null): HttpResponse<String> {
        val builder = HttpRequest.newBuilder(URI("http://localhost:$port$path")).timeout(java.time.Duration.ofSeconds(15))
        if (auth != null) builder.header("Authorization", "Bearer $auth")
        if (body == null) builder.GET() else builder.POST(HttpRequest.BodyPublishers.ofString(body))
            .header("Content-Type", "application/json")
        return client(port).send(builder.build(), HttpResponse.BodyHandlers.ofString())
    }

    private fun testServer(apiKey: String? = null): Pair<OpenAICompatServer, Kaibo> {
        val kaibo = Kaibo()
        kaibo.registerAgent(agentConfig("echo") {
            module<EchoEntry>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hello")))
            entry("entry")
        })
        return OpenAICompatServer(kaibo, 0, apiKey).start() to kaibo
    }

    @Test fun `chat completion runs the agent and answers`() {
        val (server, _) = testServer()
        val resp = request(server.listeningPort(), "/v1/chat/completions", chatBody("echo"))
        assertEquals(200, resp.statusCode())
        val body = Json.parseToJsonElement(resp.body()).jsonObject
        assertEquals("assistant", body["choices"]!!.jsonArray[0].jsonObject["message"]!!.jsonObject["role"]?.jsonPrimitive?.content)
        assertEquals("echo:hello", body["choices"]!!.jsonArray[0].jsonObject["message"]!!.jsonObject["content"]?.jsonPrimitive?.content)
    }

    @Test fun `unknown model is a 400`() {
        val (server, _) = testServer()
        val resp = request(server.listeningPort(), "/v1/chat/completions", chatBody("nope"))
        assertEquals(400, resp.statusCode())
    }

    @Test fun `models lists entry points`() {
        val (server, _) = testServer()
        val body = Json.parseToJsonElement(request(server.listeningPort(), "/v1/models").body()).jsonObject
        assertEquals("echo", body["data"]!!.jsonArray[0].jsonObject["id"]?.jsonPrimitive?.content)
    }

    @Test fun `api key gates access`() {
        val (server, _) = testServer(apiKey = "secret")
        assertEquals(401, request(server.listeningPort(), "/v1/models").statusCode())
        assertEquals(200, request(server.listeningPort(), "/v1/models", auth = "secret").statusCode())
    }

    @Test fun `streaming delivers deltas and DONE`() {
        val (server, _) = testServer()
        val resp = request(server.listeningPort(), "/v1/chat/completions", chatBody("echo", stream = true))
        assertEquals(200, resp.statusCode())
        val lines = resp.body().lines().filter { it.startsWith("data:") }
        assertTrue(lines.any { it.contains("echo:hello") }, "chunk content present")
        assertEquals("data: [DONE]", lines.last().trim())
    }
}
