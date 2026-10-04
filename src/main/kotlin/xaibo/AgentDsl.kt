package xaibo

import kotlin.reflect.KClass

/**
 * Kotlin DSL over [AgentConfig]. Produces exactly the same config object that
 * YAML files produce — so DSL-built agents can be inspected, serialized and
 * overridden with [ConfigOverrides] just like file-loaded ones.
 *
 * ```
 * val cfg = agentConfig("chat") {
 *     description = "echo bot"
 *     module<SimpleConversation>("history", "max_history" to 50)
 *     module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hi")))
 *     module<ToolCollector>("tools")
 *     module<SimpleToolOrchestrator>("entry", "system_prompt" to "You are helpful")
 *     entry("entry")
 * }
 * ```
 *
 * Unambiguous protocol wiring and the `__response__`/`__entry__` plumbing is
 * inferred; use [bind] only when the agent really has several providers.
 */
fun agentConfig(id: String, block: AgentDsl.() -> Unit): AgentConfig =
    AgentDsl(id).apply(block).build()

class AgentDsl internal constructor(val id: String) {
    var description: String? = null
    internal val modules = mutableListOf<ModuleConfig>()
    internal val exchange = mutableListOf<ExchangeConfig>()

    inline fun <reified T : Any> module(
        id: String,
        vararg config: Pair<String, Any?>,
        scope: Scope = Scope.INSTANCE,
        provides: List<KClass<*>> = emptyList(),
    ) = module(id, T::class, *config, scope = scope, provides = provides)

    fun module(
        id: String,
        type: KClass<*>,
        vararg config: Pair<String, Any?>,
        scope: Scope = Scope.INSTANCE,
        provides: List<KClass<*>> = emptyList(),
    ) {
        modules.add(
            ModuleConfig(
                module = type.java.name,
                id = id,
                scope = scope,
                provides = provides.map { it.simpleName!! }.toMutableList().ifEmpty { null },
                config = config.toMap(),
            )
        )
    }

    /** Bind protocol [P] on [module] (or everywhere, if null) to [providers]. */
    inline fun <reified P : Any> bind(vararg providers: String, module: String? = null, field: String? = null) =
        bind(P::class, providers.toList(), module, field)

    fun bind(protocol: KClass<*>, providers: List<String>, module: String? = null, field: String? = null) {
        exchange.add(ExchangeConfig(module, field, protocol.simpleName!!, providers))
    }

    /** Route incoming messages of the given handler protocol to [moduleIds]. */
    fun entry(vararg moduleIds: String, protocol: KClass<*> = TextMessageHandlerProtocol::class) {
        exchange.add(ExchangeConfig("__entry__", null, protocol.simpleName!!, moduleIds.toList()))
    }

    /** @see AgentConfig — its constructor performs the implicit wiring */
    fun build(): AgentConfig = AgentConfig(id, description, modules, exchange)
}
