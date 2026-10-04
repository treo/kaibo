package xaibo.primitives

import xaibo.*

/**
 * Answers a user message by looping LLM calls and tool executions until the
 * model stops calling tools or `max_thoughts` is reached. Tool failures raise
 * the temperature — the model works under increasing stress.
 *
 * Config: `system_prompt` (string), `max_thoughts` (int, default 10).
 */
class SimpleToolOrchestrator(
    val response: ResponseProtocol,
    val llm: LLMProtocol,
    val toolProvider: ToolProviderProtocol,
    val history: ConversationHistoryProtocol,
    val config: Map<String, Any?> = emptyMap(),
) : TextMessageHandlerProtocol {

    private val systemPrompt = config["system_prompt"] as? String ?: ""
    private val maxThoughts = (config["max_thoughts"] as? Number)?.toInt() ?: 10

    override suspend fun handleText(text: String) {
        val conversation = history.getHistory().toMutableList()
        if (systemPrompt.isNotEmpty()) conversation.add(0, LLMMessage.system(systemPrompt))
        conversation.add(LLMMessage.user(text))

        val tools = toolProvider.listTools()
        var thoughts = 0
        var stressLevel = 0.0 // raise temperature when tools fail

        while (thoughts < maxThoughts) {
            thoughts++
            if (thoughts == maxThoughts)
                conversation.add(LLMMessage.system("Maximum tool usage reached. Tools Unavailable"))

            val options = LLMOptions(
                temperature = stressLevel,
                functions = if (thoughts < maxThoughts) tools else null,
            )
            val llmResponse = llm.generate(conversation, options)
            llmResponse.usage?.let { response.respondEvent(UsageEvent(it)) }

            conversation.add(
                LLMMessage(
                    role = if (llmResponse.toolCalls != null) LLMRole.FUNCTION else LLMRole.ASSISTANT,
                    content = listOf(LLMMessageContent(LLMMessageContentType.TEXT, text = llmResponse.content)),
                    toolCalls = llmResponse.toolCalls,
                )
            )

            val toolCalls = llmResponse.toolCalls.orEmpty()
            if (thoughts >= maxThoughts || toolCalls.isEmpty()) break

            val results = toolCalls.map { call ->
                response.respondEvent(ToolCallEvent(call.id, call.name, call.arguments))
                val result = try {
                    toolProvider.executeTool(call.name, call.arguments)
                } catch (e: Exception) {
                    ToolResult(success = false, error = e.message)
                }
                if (!result.success) stressLevel += 0.1
                response.respondEvent(
                    ToolResultEvent(call.id, call.name, result.success,
                        result = result.result?.jsonSafe().takeIf { result.success },
                        error = result.error.takeIf { !result.success })
                )
                mapOf("id" to call.id, "name" to call.name) +
                    if (result.success) mapOf("result" to result.result.jsonString())
                    else mapOf("error" to "Error: ${result.error}")
            }

            conversation.add(
                LLMMessage(
                    role = LLMRole.FUNCTION,
                    toolResults = results.map {
                        LLMFunctionResult(
                            id = it["id"] as String,
                            name = it["name"] as String,
                            content = (it["result"] ?: it["error"]) as String,
                        )
                    },
                )
            )
        }

        // The final assistant message is the answer
        response.respondText(conversation.last().content.firstOrNull()?.text ?: "")
    }
}
