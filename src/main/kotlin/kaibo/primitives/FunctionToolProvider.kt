package kaibo.primitives

import kaibo.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import kotlin.reflect.KClass
import kotlin.reflect.KFunction
import kotlin.reflect.KParameter
import kotlin.reflect.KVisibility
import kotlin.reflect.full.declaredMemberFunctions

/** Marks a function as an kaibo tool. `description` is required — a tool the
 * model cannot read about is a tool the model cannot use well; the compiler
 * refuses to build one silently. */
@Target(AnnotationTarget.FUNCTION)
annotation class KaiboTool(val description: String, val name: String = "")

/** Documents a tool parameter; lands in the JSON schema the model sees. */
@Target(AnnotationTarget.VALUE_PARAMETER)
annotation class KaiboParam(val description: String = "")

/**
 * Exposes annotated Kotlin functions as tools.
 *
 * Config: `tool_classes` — list of class names with a no-arg constructor whose
 * @KaiboTool functions become tools; `tools` — ready-made receiver objects
 * (from code) scanned the same way. Describe parameters with @KaiboParam;
 * Kotlin enum parameters become JSON-schema enums.
 *
 * Long-running tools: functions run on [Dispatchers.IO] so the agent's own
 * coroutine stays a good citizen; `call_timeout_ms` caps how LONG the agent
 * *waits*, answering with a failed [ToolResult] instead of stalling the turn.
 * A blocked function cannot be interrupted — after a timeout it runs to
 * completion unseen, so tool side effects should be idempotent.
 */
class FunctionToolProvider(val config: Map<String, Any?> = emptyMap()) : ToolProviderProtocol {

    private val callTimeoutMs = (config["call_timeout_ms"] as? Number)?.toLong()
    // tools outlive the call that launched them: a timed-out call is abandoned,
    // not joined — the agent moves on while the function finishes unseen
    private val ioScope = CoroutineScope(Dispatchers.IO)

    private class Entry(
        val tool: Tool,
        val receiver: Any,
        val fn: KFunction<*>,
        val params: List<KParameter>,
    )

    private val entries: List<Entry> = run {
        val objects: List<Any> =
            (config["tool_classes"] as? List<String>).orEmpty().map {
                Class.forName(it).getDeclaredConstructor().newInstance()
            } + (config["tools"] as? List<Any>).orEmpty()

        objects.flatMap(::scan).distinctBy { it.tool.name }
    }

    /**
     * Tools are collected by walking the class hierarchy and reading the
     * annotations on each class's own declarations — the most-derived
     * annotated signature wins. Invoking a base declaration on a subclass
     * receiver dispatches virtually, so overrides need no annotation.
     */
    private fun scan(obj: Any): List<Entry> {
        val out = LinkedHashMap<String, Entry>()
        var k: KClass<*>? = obj::class
        while (k != null && k != Any::class) {
            for (fn in k.declaredMemberFunctions) {
                val ann = fn.annotations.filterIsInstance<KaiboTool>().firstOrNull() ?: continue
                if (fn.visibility != KVisibility.PUBLIC) {
                    System.err.println("skipping non-public tool function '${fn.name}'")
                    continue
                }
                if (fn.parameters.any { it.kind == KParameter.Kind.EXTENSION_RECEIVER }) {
                    System.err.println("skipping tool function with extension receiver '${fn.name}'")
                    continue
                }
                val name = ann.name.ifEmpty { fn.name }
                if (name in out) continue // a more-derived annotation already claimed it
                val params = fn.parameters.filter { it.kind == KParameter.Kind.VALUE }
                out[name] = Entry(
                    Tool(
                        name = name,
                        description = ann.description.ifEmpty { fn.name },
                        parameters = params.associate { p -> p.name!! to schemaFor(p) },
                    ),
                    obj, fn, params,
                )
            }
            k = k.java.superclass?.kotlin
        }
        return out.values.toList()
    }

    /** Parameter -> schema entry: type, description, required, enum-from-Kotlin-enums */
    private fun schemaFor(p: KParameter): ToolParameter {
        val cls = p.type.classifier as? KClass<*>
        val required = !p.isOptional && !p.type.isMarkedNullable
        val desc = p.annotations.filterIsInstance<KaiboParam>().firstOrNull()?.description?.ifEmpty { null }
        return if (cls?.java?.isEnum == true)
            ToolParameter("string", desc, required,
                enum = cls.java.enumConstants.map { (it as Enum<*>).name })
        else ToolParameter(jsonType(cls), desc, required)
    }

    override suspend fun listTools(): List<Tool> = entries.map { it.tool }

    override suspend fun executeTool(toolName: String, parameters: Map<String, Any?>): ToolResult {
        val entry = entries.firstOrNull { it.tool.name == toolName }
            ?: return ToolResult(success = false, error = "Could not find $toolName")
        val args = try {
            entry.params.mapNotNull { p ->
                val value = parameters[p.name]
                when {
                    value != null -> p to coerce(value, p.type.classifier as? KClass<*>)
                    p.isOptional -> null // the default applies
                    p.type.isMarkedNullable -> p to null
                    else -> return ToolResult(false, error = "missing parameter ${p.name}")
                }
            }.toMap()
        } catch (e: Exception) {
            return ToolResult(success = false, error = "${e::class.simpleName}: ${e.message}")
        }
        return try {
            val deferred = ioScope.async {
                val callArgs = HashMap<KParameter, Any?>()
                entry.fn.parameters.firstOrNull { it.kind == KParameter.Kind.INSTANCE }?.let { callArgs[it] = entry.receiver }
                callArgs.putAll(args)
                entry.fn.callBy(callArgs)
            }
            val value = try {
                if (callTimeoutMs != null) withTimeout(callTimeoutMs) { deferred.await() }
                else deferred.await()
            } catch (e: TimeoutCancellationException) {
                return ToolResult(false, error = "tool '$toolName' timed out after ${callTimeoutMs}ms")
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e // the agent itself was cancelled; do not swallow
            }
            ToolResult(success = true, result = value)
        } catch (e: Exception) {
            ToolResult(success = false, error = "${e::class.simpleName}: ${e.message}")
        }
    }

    private fun jsonType(cls: KClass<*>?) = when (cls) {
        Int::class, Long::class, Short::class -> "integer"
        Double::class, Float::class -> "number"
        Boolean::class -> "boolean"
        List::class -> "array"
        Map::class -> "object"
        else -> "string"
    }

    /** JSON parses numbers generically; nudge them into what the function wants */
    private fun coerce(value: Any?, cls: KClass<*>?): Any? = when {
        cls == Int::class && value is Number -> value.toInt()
        cls == Long::class && value is Number -> value.toLong()
        cls == Double::class && value is Number -> value.toDouble()
        cls == Float::class && value is Number -> value.toFloat()
        cls?.java?.isEnum == true && value is String ->
            cls.java.enumConstants.first { (it as Enum<*>).name.equals(value, ignoreCase = true) }
        else -> value
    }
}
