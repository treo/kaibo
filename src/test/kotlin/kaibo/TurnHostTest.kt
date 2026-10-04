package kaibo

import kaibo.primitives.*
import kaibo.server.OpenAICompatServer
import kaibo.server.TurnHost
import kotlinx.coroutines.runBlocking
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import kotlin.test.*

/**
 * A host can tell the server who is asking and which conversation a turn runs
 * on, without reimplementing the HTTP layer. The important half of the contract:
 * with no host metadata, behaviour is exactly what it was before hosts existed.
 */
class TurnHostTest {

    /** Binds the turn, and keeps one journal for the channel it recognises. */
    private class ChannelHost : TurnHost {
        var identified = 0
        var boundDuringTurn = false
        val channelJournal = SimpleConversation()
        val anonymousJournal = SimpleConversation()

        override fun identify(exchange: com.sun.net.httpserver.HttpExchange): Any? {
            val channel = exchange.requestHeaders.getFirst("X-Channel") ?: return null
            identified++
            return channel
        }

        override fun history(messages: List<Map<String, Any?>>, turn: Any?): ConversationHistoryProtocol =
            if (turn == null) anonymousJournal else channelJournal

        override suspend fun <T> withTurn(turn: Any?, block: suspend () -> T): T {
            boundDuringTurn = turn != null
            return block()
        }
    }

    private fun serverWith(host: TurnHost): OpenAICompatServer {
        val kaibo = Kaibo()
        kaibo.registerAgent(agentConfig("hosted") {
            module<JournalEntry>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hi")))
            entry("entry")
        })
        return OpenAICompatServer(kaibo, 0, host = host).start()
    }

    private fun post(port: Int, body: String, channel: String? = null): HttpResponse<String> {
        var request = HttpRequest.newBuilder(URI("http://localhost:$port/v1/chat/completions"))
            .timeout(java.time.Duration.ofSeconds(15))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
        if (channel != null) request = request.header("X-Channel", channel)
        return HttpClient.newHttpClient().send(request.build(), HttpResponse.BodyHandlers.ofString())
    }

    private fun historyOf(conversation: SimpleConversation): List<String> =
        runBlocking { conversation.getHistory() }.map { message ->
            message.content.firstOrNull { it.type == LLMMessageContentType.TEXT }?.text ?: ""
        }

    @Test fun `an ordinary request keeps answering from the body alone`() {
        val host = ChannelHost()
        val server = serverWith(host)
        val response = post(server.listeningPort(),
            """{"model":"hosted","messages":[{"role":"user","content":"ping"}]}""")
        assertEquals(200, response.statusCode())
        assertTrue(response.body().contains("echo:hi"), response.body())
        assertEquals(listOf("ping", "echo:hi"), historyOf(host.anonymousJournal),
            "the default history is the request's own, and this host said which")
        assertEquals(0, host.identified, "no channel header, no host turn identity")
        assertTrue(host.boundDuringTurn.not(), "withTurn is not treated as a requirement")
    }

    @Test fun `the host chooses the conversation and wraps the turn it runs`() {
        val host = ChannelHost()
        val server = serverWith(host)

        post(server.listeningPort(),
            """{"model":"hosted","messages":[{"role":"user","content":"one"}]}""", "channel-7")
        assertTrue(host.boundDuringTurn, "withTurn must wrap the agent turn, not merely be called")
        assertEquals(listOf("one", "echo:hi"),
            historyOf(host.channelJournal), "the host's journal received the whole turn")

        post(server.listeningPort(),
            """{"model":"hosted","messages":[{"role":"user","content":"two"}]}""", "channel-7")
        val journal = historyOf(host.channelJournal)
        assertEquals("one", journal.first(), "the first turn is still there")
        assertEquals(4, journal.size, "the second turn appended to the same journal: $journal")
        assertTrue("two" in journal, journal.toString())
        assertTrue(historyOf(host.anonymousJournal).isEmpty(),
            "an anonymous request must not land in the channel's journal")
    }
}
