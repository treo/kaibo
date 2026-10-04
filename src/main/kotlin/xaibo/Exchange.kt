package xaibo

import kotlin.coroutines.Continuation
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.intrinsics.COROUTINE_SUSPENDED
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.lang.reflect.InvocationHandler
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Method
import java.lang.reflect.Proxy
import java.util.concurrent.atomic.AtomicLong
import kotlin.reflect.KClass
import kotlin.reflect.full.primaryConstructor
import kotlin.reflect.jvm.kotlinFunction

/** A listener: receives events whose name starts with [prefix] ("" = all). */
data class EventListener(val prefix: String, val handler: (Event) -> Unit)

/**
 * Handles module instantiation and dependency injection for an agent — the
 * heart of xaibo. Modules are created from [AgentConfig] and wired according
 * to the exchange list: constructor parameters typed with a protocol
 * interface receive the provider module, wrapped in an event-emitting proxy.
 */
class Exchange(
    val config: AgentConfig? = null,
    overrideConfig: ConfigOverrides? = null,
    private val eventListeners: List<EventListener> = emptyList(),
    specificModules: List<String>? = null,
) {
    val moduleInstances: MutableMap<String, Any> = linkedMapOf("__exchange__" to this)

    /** The effective exchange list: config + the self-binding */
    private val exchange = mutableListOf<ExchangeConfig>()
    /** Runtime override bindings: they replace config bindings for the same parameter */
    private val overrideExchange = mutableListOf<ExchangeConfig>()
    private val creating = mutableListOf<String>()

    init {
        if (config != null) {
            exchange += config.exchange
            // Every module can ask for the exchange itself
            exchange.add(ExchangeConfig(protocol = "Exchange", providers = listOf("__exchange__")))
            overrideConfig?.let { o ->
                moduleInstances.putAll(o.instances)
                for (ex in o.exchange) {
                    exchange.removeAll { it.module == ex.module && it.protocol == ex.protocol && it.fieldName == ex.fieldName }
                    overrideExchange.add(ex)
                }
            }
            val ids = specificModules ?: config.modules.map { it.id }
            ids.forEach { getModule(it) }
        }
    }

    fun entryPointIds(): List<String> =
        (overrideExchange + exchange).firstOrNull { it.module == "__entry__" }?.providers ?: emptyList()

    /**
     * Get a module instance, wrapped in a proxy that emits call/result events.
     * `__entry__` resolves through the exchange list. Returns null for unknown ids.
     */
    fun getModule(moduleId: String, callerId: String? = null): Any? {
        val id = if (moduleId == "__entry__") entryModuleId() else moduleId
        val instance = moduleInstances[id]
            ?: if (config?.moduleMapping?.containsKey(id) == true) instantiate(id) else return null
        return proxyFor(instance, id, callerId)
    }

    /** Like [getModule] but fails instead of returning null. */
    fun requireModule(moduleId: String, callerId: String? = null): Any =
        getModule(moduleId, callerId) ?: throw IllegalArgumentException("Requested module $moduleId could not be found!")

    private fun entryModuleId(): String =
        (overrideExchange + exchange).firstOrNull { it.module == "__entry__" }?.providers?.firstOrNull()
            ?: throw IllegalStateException("No message handler found in exchange config")

    // -- instantiation ----------------------------------------------------------

    private fun instantiate(moduleId: String): Any {
        moduleInstances[moduleId]?.let { return it }
        val mc = config?.moduleMapping?.get(moduleId)
            ?: throw IllegalArgumentException("Requested module $moduleId could not be found!")
        check(moduleId !in creating) { "Circular dependency instantiating $moduleId (chain: $creating)" }
        val cls = AgentConfig.loadClass(mc.module)
            ?: throw IllegalArgumentException("Could not load module class ${mc.module}")
        val ctor = cls.primaryConstructor
            ?: throw IllegalArgumentException("Module ${mc.module} needs a primary constructor")

        creating.add(moduleId)
        try {
            val dependencies = dependenciesFor(mc)
            val args = mutableMapOf<kotlin.reflect.KParameter, Any?>()
            for (p in ctor.parameters) {
                if (p.name == "config") {
                    args[p] = mc.config
                    continue
                }
                val depIds = dependencies[p.name] ?: emptyList()
                val isList = p.type.classifier == List::class || p.type.classifier == Collection::class
                // interface-typed dependencies get event proxies; concrete-class
                // ones must stay raw (a JDK proxy cannot extend a class)
                val classifier =
                    if (isList) (p.type.arguments.firstOrNull()?.type?.classifier as? KClass<*>)
                    else p.type.classifier as? KClass<*>
                val wantsProxy = classifier?.java?.isInterface == true
                when {
                    isList -> args[p] = depIds.map { inject(it, moduleId, wantsProxy) }
                    depIds.size == 1 -> args[p] = inject(depIds[0], moduleId, wantsProxy)
                    depIds.isEmpty() && p.isOptional -> {}
                    else -> throw IllegalArgumentException(
                        "Expected to find exactly one dependency resolution for module `$moduleId` parameter " +
                            "`${p.name}`, but found ${depIds.size} `$depIds`"
                    )
                }
            }
            return ctor.callBy(args).also { moduleInstances[moduleId] = it }
        } finally {
            creating.removeAt(creating.size - 1)
        }
    }

    private fun inject(depId: String, callerId: String, proxy: Boolean): Any {
        val instance = moduleInstances[depId] ?: instantiate(depId)
        return if (proxy) proxyFor(instance, depId, callerId) else instance
    }

    /**
     * Which provider ids are bound to each injectable constructor parameter of
     * [mc], by matching exchange entries for this module (or global ones) on
     * field name or protocol.
     */
    private fun dependenciesFor(mc: ModuleConfig): Map<String, List<String>> {
        val cls = AgentConfig.loadClass(mc.module) ?: return emptyMap()
        val params = injectableParams(cls)
        val byProtocol = params.groupBy { it.protocol }.mapValues { (_, ps) -> ps.map { it.name } }
        val deps = params.associate { it.name to mutableListOf<String>() }

        for (ex in exchange.filter { it.module == mc.id || it.module == null }) {
            val targets = ex.fieldName?.let { listOf(it) } ?: byProtocol[ex.protocol] ?: emptyList()
            // a module is never its own provider
            for (t in targets) deps[t]?.addAll(ex.providers.filterNot { it == mc.id })
        }
        // override bindings win: they replace what was bound before them
        for (ex in overrideExchange.filter { it.module == mc.id || it.module == null }) {
            val targets = ex.fieldName?.let { listOf(it) } ?: byProtocol[ex.protocol] ?: emptyList()
            for (t in targets) deps[t]?.apply { clear(); addAll(ex.providers.filterNot { it == mc.id }) }
        }
        return deps
    }

    // -- proxying ---------------------------------------------------------------

    private fun proxyFor(instance: Any, moduleId: String, callerId: String?): Any {
        val interfaces = allInterfaces(instance.javaClass)
        if (interfaces.isEmpty()) return instance // concrete-class wiring stays raw so casts hold
        val handler = EventProxy(instance, eventListeners, config?.id, callerId, moduleId)
        return Proxy.newProxyInstance(instance.javaClass.classLoader, interfaces.toTypedArray(), handler)
    }

    private fun allInterfaces(cls: Class<*>): List<Class<*>> {
        val out = LinkedHashSet<Class<*>>()
        var c: Class<*>? = cls
        while (c != null && c != Any::class.java) {
            for (i in c.interfaces) out += allInterfaces(i) + i
            c = c.superclass
        }
        return out.filter { !it.name.startsWith("java.") }.toList()
    }
}

/**
 * Wraps a module instance so every interface call emits CALL / RESULT /
 * EXCEPTION (and YIELD for streaming Flow calls) events to registered
 * listeners. This is the Kotlin equivalent of the Python `Proxy`/`MethodProxy`
 * pair, using a JDK dynamic proxy.
 */
internal class EventProxy(
    private val target: Any,
    private val listeners: List<EventListener>,
    private val agentId: String?,
    private val callerId: String?,
    private val moduleId: String?,
) : InvocationHandler {

    private val callId = AtomicLong()

    override fun invoke(proxy: Any, method: Method, args: Array<Any?>?): Any? {
        if (method.name == "toString" && method.declaringClass == Object::class.java)
            return "Proxy(${target.javaClass.simpleName})"
        if (method.name == "hashCode" && method.parameterCount == 0)
            return System.identityHashCode(target)
        if (method.name == "equals" && method.parameterCount == 1)
            return proxy === args?.get(0)

        val a = args ?: emptyArray()
        val last = a.lastOrNull()

        return if (last is Continuation<*>) callSuspend(method, a.dropLast(1), last)
        else if (method.returnType == Flow::class.java) callFlow(method, a)
        else callPlain(method, a.toList())
    }

    // Suspend functions: hand the callee a continuation that observes the result.
    private fun callSuspend(method: Method, plainArgs: List<Any?>, outer: Continuation<*>): Any? {
        beginCall(method, plainArgs)
        var done = false
        val wrapped = object : Continuation<Any?> {
            override val context: CoroutineContext = outer.context
            override fun resumeWith(result: kotlin.Result<Any?>) {
                done = true
                endCall(method, result)
                (outer as Continuation<Any?>).resumeWith(result)
            }
        }
        return try {
            val r = method.invoke(target, *(plainArgs + wrapped).toTypedArray())
            if (r !== COROUTINE_SUSPENDED && !done) { // completed directly; caller takes the return value
                done = true
                endCall(method, kotlin.Result.success(r))
            }
            r
        } catch (e: InvocationTargetException) {
            if (!done) endCall(method, kotlin.Result.failure(e.targetException))
            throw e.targetException
        }
    }

    // Streaming methods (Flow): events fire as the consumer collects.
    @Suppress("UNCHECKED_CAST")
    private fun callFlow(method: Method, plainArgs: Array<Any?>): Any = flow {
        beginCall(method, plainArgs.toList())
        var chunks = 0L
        try {
            val source = method.invoke(target, *plainArgs) as Flow<Any?>
            source.collect {
                chunks++
                emitEvent(EventType.YIELD, result = it, method = method)
                emit(it)
            }
            emitEvent(EventType.RESULT, method = method,
                result = mapOf("stream" to true, "chunks" to chunks))
        } catch (e: Throwable) {
            emitEvent(EventType.EXCEPTION, method = method, exception = e.stackTraceToString())
            throw e
        }
    }

    private fun callPlain(method: Method, plainArgs: List<Any?>): Any? {
        beginCall(method, plainArgs)
        return try {
            method.invoke(target, *plainArgs.toTypedArray()).also { endCall(method, kotlin.Result.success(it)) }
        } catch (e: InvocationTargetException) {
            endCall(method, kotlin.Result.failure(e.targetException))
            throw e.targetException
        }
    }

    // -- event plumbing ---------------------------------------------------------

    private fun beginCall(method: Method, args: List<Any?>) {
        if (listeners.isEmpty()) return
        emitEvent(EventType.CALL, method = method, arguments = namedArgs(method, args))
    }

    private fun endCall(method: Method, result: kotlin.Result<Any?>) {
        if (listeners.isEmpty()) return
        result.fold(
            onSuccess = { emitEvent(EventType.RESULT, method = method, result = it) },
            onFailure = { emitEvent(EventType.EXCEPTION, method = method, exception = it.stackTraceToString()) },
        )
    }

    private fun emitEvent(
        type: EventType,
        result: Any? = null,
        arguments: Map<String, Any?>? = null,
        exception: String? = null,
        method: Method? = null,
    ) {
        val m = method ?: return
        val cls = target.javaClass
        val name = "${cls.`package`?.name}.${cls.simpleName}.${m.name}.${type.value}"
        val event = Event(
            agentId = agentId,
            eventName = name,
            eventType = type,
            moduleId = moduleId,
            moduleClass = cls.simpleName,
            methodName = m.name,
            time = System.currentTimeMillis() / 1000.0,
            callId = "${System.identityHashCode(target)}-${m.name}-${callId.incrementAndGet()}",
            callerId = callerId,
            arguments = arguments,
            result = result,
            exception = exception,
        )
        for (l in listeners) {
            if (l.prefix.isEmpty() || event.eventName.startsWith(l.prefix)) {
                try {
                    l.handler(event)
                } catch (e: Throwable) {
                    System.err.println("Exception during event handling: ${e.stackTraceToString()}")
                }
            }
        }
    }

    private fun namedArgs(method: Method, args: List<Any?>): Map<String, Any?> {
        val names = runCatching { method.kotlinFunction?.parameters?.drop(1)?.map { it.name ?: "arg" } }
            .getOrNull() ?: List(args.size) { "arg$it" }
        return args.mapIndexed { i, v -> (names.getOrNull(i) ?: "arg$i") to v }.toMap()
    }
}
