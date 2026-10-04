package xaibo.primitives

import xaibo.*
import java.io.File
import java.util.Base64

/**
 * "One shot" tools: pure-prompt tools defined entirely in config. Each tool
 * declares a conversation template; parameter values are injected into the
 * template via `$$params.<name>$$` and the conversation is run through the
 * agent's LLM.
 *
 * Config: `tools`: list of {name, description, parameters: [{name, type,
 * description}], conversation: [{role, message: [{type: text|image_url,
 * text?, url?}]}]}
 */
class OneShotTools(val llm: LLMProtocol, val config: Map<String, Any?> = emptyMap()) : ToolProviderProtocol {

    private class Spec(val map: Map<String, Any?>) {
        val name = map["name"] as? String ?: error("one-shot tool needs a name")
        val description = map["description"] as? String ?: ""
        val parameters = (map["parameters"] as? List<Map<String, Any?>>).orEmpty()
        val conversation = (map["conversation"] as? List<Map<String, Any?>>).orEmpty()
    }

    private val tools = (config["tools"] as? List<Map<String, Any?>>).orEmpty()
        .map { it["name"] as String to Spec(it) }.toMap()

    override suspend fun listTools(): List<Tool> = tools.values.map { spec ->
        Tool(
            name = spec.name,
            description = spec.description,
            parameters = spec.parameters.associate {
                it["name"] as String to ToolParameter(
                    type = it["type"] as? String ?: "string",
                    description = it["description"] as? String,
                    required = true,
                )
            },
        )
    }

    override suspend fun executeTool(toolName: String, parameters: Map<String, Any?>): ToolResult {
        val spec = tools[toolName] ?: return ToolResult(false, error = "Could not find $toolName")
        val messages = spec.conversation.map { entry ->
            LLMMessage(
                role = LLMRole.of(entry["role"] as? String ?: "user"),
                content = (entry["message"] as? List<Map<String, Any?>>).orEmpty().map { part ->
                    when (part["type"]) {
                        "image_url" -> LLMMessageContent(
                            LLMMessageContentType.IMAGE,
                            image = inject(part["url"] as? String ?: error("Empty image URL in tool conversation"), parameters)
                                .let(::toDataUri),
                        )
                        else -> LLMMessageContent(
                            LLMMessageContentType.TEXT,
                            text = inject(part["text"] as? String ?: error("Empty text message content in tool conversation"), parameters),
                        )
                    }
                },
            )
        }
        val response = llm.generate(messages)
        return if (response.content.isEmpty()) ToolResult(false, error = "No response generated")
        else ToolResult(true, result = response.content)
    }

    private fun inject(template: String, parameters: Map<String, Any?>): String =
        parameters.entries.fold(template) { text, (k, v) -> text.replace("\$\$params.$k\$\$", v.toString()) }

    /** A local path becomes a base64 data URI; anything else travels unchanged */
    private fun toDataUri(url: String): String {
        val file = File(url)
        if (!file.isFile) return url
        val mime = mapOf(
            "png" to "image/png", "jpg" to "image/jpeg", "jpeg" to "image/jpeg",
            "gif" to "image/gif", "webp" to "image/webp",
        ).getOrDefault(file.extension.lowercase(), "application/octet-stream")
        return "data:$mime;base64," + Base64.getEncoder().encodeToString(file.readBytes())
    }
}
