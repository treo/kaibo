package xaibo

/**
 * Manages agent configurations, shared modules and event listeners, and
 * instantiates agents from registered configurations.
 */
class Registry {
    val knownAgentConfigs = linkedMapOf<String, AgentConfig>()
    private val eventListeners = mutableListOf<Triple<String, String?, (Event) -> Unit>>()
    private val serverModuleInstances = linkedMapOf<String, Any>()
    private val agentModuleInstances = linkedMapOf<String, MutableMap<String, Any>>()

    fun registerServerModule(id: String, module: Any) {
        serverModuleInstances[id] = module
    }

    fun unregisterServerModule(id: String) {
        serverModuleInstances.remove(id)
    }

    fun registerAgent(agentConfig: AgentConfig) {
        knownAgentConfigs[agentConfig.id] = agentConfig
        agentModuleInstances[agentConfig.id] = linkedMapOf()
        val agentScopeIds = agentConfig.modules.filter { it.scope == Scope.AGENT }.map { it.id }
        if (agentScopeIds.isNotEmpty()) {
            val exchange = Exchange(agentConfig, specificModules = agentScopeIds)
            agentScopeIds.forEach { agentModuleInstances[agentConfig.id]!![it] = exchange.moduleInstances[it]!! }
        }
    }

    fun unregisterAgent(agentId: String) {
        knownAgentConfigs.remove(agentId)
        agentModuleInstances.remove(agentId)
    }

    fun getAgentConfig(agentId: String): AgentConfig =
        knownAgentConfigs[agentId] ?: throw NoSuchElementException("No agent configuration found for id: $agentId")

    fun listAgents(): List<String> = knownAgentConfigs.keys.toList()

    fun getAgent(id: String): Agent = getAgentWith(id, null)

    /**
     * Instantiate an agent, applying custom instances/bindings and merging in
     * agent-scoped and server-scoped modules (also bound by their class name
     * so modules can request them directly).
     */
    fun getAgentWith(
        id: String,
        overrideConfig: ConfigOverrides? = null,
        additionalEventListeners: List<EventListener> = emptyList(),
    ): Agent {
        val config = knownAgentConfigs[id]
            ?: throw NoSuchElementException("No agent configuration found for id: $id")

        val listeners = eventListeners
            .filter { (_, agentFilter, _) -> agentFilter == null || agentFilter == id }
            .map { (prefix, _, handler) -> EventListener(prefix, handler) }
            .toMutableList()
        listeners += additionalEventListeners

        val overrides = overrideConfig ?: ConfigOverrides()
        val exchange = Exchange(
            config,
            overrideConfig = ConfigOverrides(
                instances = overrides.instances + agentModuleInstances[id].orEmpty() + serverModuleInstances,
                exchange = overrides.exchange + serverModuleInstances.map { (moduleId, module) ->
                    ExchangeConfig(protocol = module.javaClass.simpleName, providers = listOf(moduleId))
                },
            ),
            eventListeners = listeners,
        )
        return Agent(id = id, exchange = exchange)
    }

    /**
     * Listen to module call/result events. `prefix` matches the start of an
     * event name `{package}.{class}.{method}.{call|result|...}`; "" for all.
     */
    fun registerEventListener(prefix: String, handler: (Event) -> Unit, agentId: String? = null) {
        eventListeners.add(Triple(prefix, agentId, handler))
    }
}

/** The primary entry point to the xaibo framework. */
class Xaibo {
    val registry = Registry()

    init {
        if (System.getenv("XAIBO_DEBUG") != null) {
            xaibo.primitives.registerDebugListener(registry)
        }
        registerServerModule("__xaibo__", this)
    }

    fun registerServerModule(id: String, module: Any) = registry.registerServerModule(id, module)
    fun unregisterServerModule(id: String) = registry.unregisterServerModule(id)
    fun registerAgent(agentConfig: AgentConfig) = registry.registerAgent(agentConfig)
    fun unregisterAgent(agentId: String) = registry.unregisterAgent(agentId)
    fun listAgents(): List<String> = registry.listAgents()
    fun getAgentConfig(agentId: String): AgentConfig = registry.getAgentConfig(agentId)
    fun registerEventListener(prefix: String, handler: (Event) -> Unit, agentId: String? = null) =
        registry.registerEventListener(prefix, handler, agentId)

    fun getAgent(agentId: String): Agent = registry.getAgent(agentId)
    fun getAgentWith(
        agentId: String,
        overrideConfig: ConfigOverrides?,
        additionalEventListeners: List<EventListener> = emptyList(),
    ): Agent = registry.getAgentWith(agentId, overrideConfig, additionalEventListeners)
}
