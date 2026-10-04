package xaibo

/**
 * A wired agent: an id plus the exchange that instantiates and injects its
 * modules. Incoming messages are delegated to an entry module; the reply is
 * collected from the `__response__` module.
 */
class Agent(val id: String, val exchange: Exchange) {

    override fun toString() = "Agent '$id'"

    fun getEntryPointIds(): List<String> = exchange.entryPointIds()

    suspend fun handleText(text: String, entryPoint: String = "__entry__"): Response {
        val entry = exchange.getModule(entryPoint, "agent:$id") as? TextMessageHandlerProtocol
            ?: throw NoSuchElementException("Entry module does not implement TextMessageHandlerProtocol")
        entry.handleText(text)
        return responseModule().getResponse()
    }

    suspend fun handleImage(image: ByteArray, entryPoint: String = "__entry__"): Response {
        val entry = exchange.getModule(entryPoint, "agent:$id") as? ImageMessageHandlerProtocol
            ?: throw NoSuchElementException("Entry module does not implement ImageMessageHandlerProtocol")
        entry.handleImage(image)
        return responseModule().getResponse()
    }

    suspend fun handleAudio(audio: ByteArray, entryPoint: String = "__entry__"): Response {
        val entry = exchange.getModule(entryPoint, "agent:$id") as? AudioMessageHandlerProtocol
            ?: throw NoSuchElementException("Entry module does not implement AudioMessageHandlerProtocol")
        entry.handleAudio(audio)
        return responseModule().getResponse()
    }

    suspend fun handleVideo(video: ByteArray, entryPoint: String = "__entry__"): Response {
        val entry = exchange.getModule(entryPoint, "agent:$id") as? VideoMessageHandlerProtocol
            ?: throw NoSuchElementException("Entry module does not implement VideoMessageHandlerProtocol")
        entry.handleVideo(video)
        return responseModule().getResponse()
    }

    private fun responseModule(): ResponseProtocol =
        exchange.getModule("__response__", "agent:$id") as ResponseProtocol
}
