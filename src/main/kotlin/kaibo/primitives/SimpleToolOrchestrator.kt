package kaibo.primitives

import kaibo.*

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

            val (resultMessages, stress) = executeToolBatch(toolProvider, response, toolCalls)
            conversation.addAll(resultMessages)
            stressLevel += stress
        }

        // The final assistant message is the answer
        response.respondText(conversation.last().content.firstOrNull()?.text ?: "")
    }
}

/**
 * Runs a batch of tool calls, emitting call/result events, and returns the
 * result messages plus the accumulated stress (one +0.1 per failure). Each
 * result becomes its own FUNCTION message — the canonical OpenAI shape, and
 * what lets a persisting loop journal them one by one.
 */
internal suspend fun executeToolBatch(
    toolProvider: ToolProviderProtocol,
    response: ResponseProtocol,
    toolCalls: List<LLMFunctionCall>,
): Pair<List<LLMMessage>, Double> {
    var stress = 0.0
    val messages = toolCalls.map { call ->
        response.respondEvent(ToolCallEvent(call.id, call.name, call.arguments))
        val result = try {
            toolProvider.executeTool(call.name, call.arguments)
        } catch (e: Exception) {
            ToolResult(success = false, error = e.message)
        }
        if (!result.success) stress += 0.1
        response.respondEvent(
            ToolResultEvent(call.id, call.name, result.success,
                result = result.result?.jsonSafe().takeIf { result.success },
                error = result.error.takeIf { !result.success }),
        )
        val content = if (result.success) result.result.jsonString() else "Error: ${result.error}"
        LLMMessage.functionResult(call.id, call.name, content)
    }
    return messages to stress
}
