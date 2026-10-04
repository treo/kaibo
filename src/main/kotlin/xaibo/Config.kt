package xaibo

import com.charleskorn.kaml.Yaml
import com.charleskorn.kaml.YamlList
import com.charleskorn.kaml.YamlMap
import com.charleskorn.kaml.YamlNull
import com.charleskorn.kaml.YamlScalar
import java.io.File
import kotlin.reflect.KClass
import kotlin.reflect.KParameter
import kotlin.reflect.full.primaryConstructor

enum class Scope {
    INSTANCE, AGENT;

    companion object {
        fun of(id: String?) = if (id?.lowercase() == "agent") AGENT else INSTANCE
    }
}

/** One module in an agent config. `module` is a class name (Kotlin FQN). */
data class ModuleConfig(
    val module: String,
    val id: String,
    val scope: Scope = Scope.INSTANCE,
    var provides: MutableList<String>? = null,
    var uses: MutableList<String>? = null,
    val config: Map<String, Any?> = emptyMap(),
)

/**
 * One dependency binding. `module == null` means the binding applies to every
 * module that has a constructor parameter matching `protocol` (or `fieldName`).
 * `protocol` is the simple name of the protocol (interface) or class needed.
 */
data class ExchangeConfig(
    val module: String? = null,
    val fieldName: String? = null,
    val protocol: String,
    val providers: List<String>,
)

/** Custom instances and bindings to override the defaults at instantiation. */
data class ConfigOverrides(
    val instances: Map<String, Any> = emptyMap(),
    val exchange: List<ExchangeConfig> = emptyList(),
)

/**
 * The declarative definition of an agent: a list of modules plus a list of
 * exchange (dependency-injection) bindings. Loadable from YAML
 * ([fromYaml]/[loadDirectory]) or built with the `agentConfig { }` DSL.
 *
 * On construction, unambiguous pieces of wiring are inferred ([populateImplicits]):
 * the `__response__` module, protocol bindings matched by uniqueness, and the
 * `__entry__` message handlers.
 */
class AgentConfig(
    val id: String,
    val description: String? = null,
    val modules: MutableList<ModuleConfig> = mutableListOf(),
    val exchange: MutableList<ExchangeConfig> = mutableListOf(),
) {
    val moduleMapping: Map<String, ModuleConfig> get() = modules.associateBy { it.id }

    init {
        populateImplicits()
    }

    fun populateImplicits() {
        // The response handler is always available
        if (modules.none { it.id == "__response__" }) {
            modules.add(ModuleConfig(
                module = "xaibo.primitives.ResponseHandler",
                id = "__response__",
                provides = mutableListOf("ResponseProtocol"),
            ))
        }

        val providers = protocolProviders()
        val requirements = moduleRequirements()

        // Teach each module what its constructor actually needs
        for (module in modules) {
            val needs = requirements[module.id] ?: continue
            val uses = module.uses ?: mutableListOf<String>().also { module.uses = it }
            for (param in needs.values) if (param.protocol !in uses) uses.add(param.protocol)
        }

        // Bind every used protocol that has exactly one provider
        for (module in modules) {
            val listProtocols = requirements[module.id]?.values
                ?.filter { it.isList }?.map { it.protocol }?.toSet() ?: emptySet()
            for (protocol in module.uses ?: emptyList()) {
                if (exchange.any { it.module == module.id && it.protocol == protocol }) continue
                val ps = (providers[protocol] ?: emptyList()).filterNot { it == module.id }
                when {
                    ps.isEmpty() || ps.size == 1 || protocol in listProtocols ->
                        if (ps.isNotEmpty()) exchange.add(ExchangeConfig(module = module.id, protocol = protocol, providers = ps))
                    else -> throw IllegalArgumentException(
                        "Multiple providers found for protocol $protocol used by module ${module.id}: $ps"
                    )
                }
            }
        }

        // Entry handlers: each message protocol with a unique handler
        val handlers = mutableMapOf(
            "TextMessageHandlerProtocol" to mutableListOf<String>(),
            "ImageMessageHandlerProtocol" to mutableListOf<String>(),
            "AudioMessageHandlerProtocol" to mutableListOf<String>(),
            "VideoMessageHandlerProtocol" to mutableListOf<String>(),
        )
        for (module in modules) {
            for (protocol in module.provides ?: emptyList()) {
                handlers[protocol]?.add(module.id)
            }
        }
        for ((protocol, hs) in handlers) {
            if (exchange.any { it.module == "__entry__" && it.protocol == protocol }) continue
            if (hs.size > 1) throw IllegalArgumentException(
                "Multiple handlers found for message protocol $protocol: $hs"
            )
            if (hs.size == 1) exchange.add(ExchangeConfig(module = "__entry__", protocol = protocol, providers = hs))
        }
    }

    /** protocol name -> module ids that provide it (explicit config or detected interfaces) */
    private fun protocolProviders(): MutableMap<String, MutableList<String>> {
        val providers = mutableMapOf<String, MutableList<String>>()
        fun add(protocol: String, moduleId: String) =
            providers.getOrPut(protocol) { mutableListOf() }.let { if (moduleId !in it) it.add(moduleId) }

        for (module in modules) {
            for (protocol in module.provides ?: emptyList()) add(protocol, module.id)
            loadClass(module.module)?.let { cls ->
                for (protocol in detectedProtocols(cls)) {
                    module.provides ?: run { module.provides = mutableListOf() }
                    if (protocol !in module.provides!!) module.provides!!.add(protocol)
                    add(protocol, module.id)
                }
            }
        }
        return providers
    }

    /** protocol name(s) required by each module's injectable constructor params */
    private fun moduleRequirements(): Map<String, Map<String, InjectableParam>> =
        modules.mapNotNull { m ->
            loadClass(m.module)?.let { cls ->
                m.id to injectableParams(cls).associateBy { it.name }
            }
        }.toMap()

    /**
     * Framework protocols a class provides: xaibo interfaces anywhere in its
     * hierarchy. This is the Kotlin analogue of scanning a Python class's MRO
     * for typing.Protocol bases.
     */
    private fun detectedProtocols(cls: KClass<*>): List<String> {
        val found = LinkedHashSet<String>()
        var c: Class<*>? = cls.java
        while (c != null && c.name.startsWith("xaibo")) {
            for (i in c.interfaces) if (i.isInterface && i.name.startsWith("xaibo.")) found.add(i.simpleName)
            c = c.superclass
        }
        return found.toList()
    }

    // ------------------------------------------------------------------ loading

    companion object {
        /** Parse one agent config from a YAML string */
        fun fromYaml(yaml: String): AgentConfig = fromMap(requireNotNull(Yaml.default.parseToYamlNode(yaml).toKotlin() as? Map<*, *>) {
            "Agent config must be a mapping"
        })

        /** Load all agent configs in a directory (recursively, *.yml / *.yaml) */
        fun loadDirectory(directory: String): Map<String, AgentConfig> {
            val configs = linkedMapOf<String, AgentConfig>()
            File(directory).walkTopDown().filter { it.isFile && it.extension in listOf("yml", "yaml") }.forEach { f ->
                try {
                    configs[f.path] = fromYaml(f.readText())
                } catch (e: Exception) {
                    throw IllegalArgumentException("Invalid agent config in ${f.path}: ${e.message}", e)
                }
            }
            return configs
        }

        fun fromMap(m: Map<*, *>): AgentConfig {
            fun strList(v: Any?) = (v as? List<*>)?.map { it.toString() }?.toMutableList()
            fun providerList(v: Any?) = when (v) {
                is List<*> -> v.map { it.toString() }
                null -> emptyList()
                else -> listOf(v.toString())
            }
            val modules = (m["modules"] as? List<Map<String, Any?>> ?: emptyList()).map { mm ->
                ModuleConfig(
                    module = mm["module"] as? String ?: error("module entry needs a 'module' class"),
                    id = mm["id"] as? String ?: error("module entry needs an 'id'"),
                    scope = Scope.of(mm["scope"] as? String),
                    provides = strList(mm["provides"]),
                    uses = strList(mm["uses"]),
                    config = (mm["config"] as? Map<String, Any?>) ?: emptyMap(),
                )
            }
            val exchange = (m["exchange"] as? List<Map<String, Any?>> ?: emptyList()).map { ee ->
                ExchangeConfig(
                    module = ee["module"] as? String,
                    fieldName = ee["field_name"] as? String,
                    protocol = ee["protocol"] as? String ?: error("exchange entry needs a 'protocol'"),
                    providers = providerList(ee["provider"]),
                )
            }
            return AgentConfig(
                id = m["id"] as? String ?: error("agent config needs an 'id'"),
                description = m["description"] as? String,
                modules = modules.toMutableList(),
                exchange = exchange.toMutableList(),
            )
        }

        fun loadClass(name: String): KClass<*>? = try {
            Class.forName(name).kotlin
        } catch (e: Throwable) {
            null
        }
    }
}

/** Injectable constructor parameter: name + the protocol that fills it. */
data class InjectableParam(val name: String, val protocol: String, val isList: Boolean)

/**
 * The primary-constructor parameters the exchange must inject: every parameter
 * except `config`. A `List<P>` parameter asks for all providers of P; any
 * other parameter asks for the single provider of its type.
 */
internal fun injectableParams(cls: KClass<*>): List<InjectableParam> {
    val ctor = cls.primaryConstructor ?: return emptyList()
    return ctor.parameters.filter { it.name != "config" }.map { p ->
        val classifier = p.type.classifier as? KClass<*>
        val isList = classifier == List::class || classifier == Collection::class
        val protocol = if (isList) {
            (p.type.arguments.firstOrNull()?.type?.classifier as? KClass<*>)?.simpleName ?: "Any"
        } else {
            classifier?.simpleName ?: "Any"
        }
        InjectableParam(p.name!!, protocol, isList)
    }
}

/** kaml YAML nodes -> plain Kotlin values (String/Int/Long/Double/Boolean/null/List/Map) */
fun com.charleskorn.kaml.YamlNode.toKotlin(): Any? = when (this) {
    is YamlNull -> null
    is YamlScalar -> when (content) {
        "null", "~", "" -> null
        "true" -> true
        "false" -> false
        else -> content.toLongOrNull()?.let { if (it in Int.MIN_VALUE..Int.MAX_VALUE) it.toInt() else it }
            ?: content.toDoubleOrNull()
            ?: content
    }
    is YamlList -> items.map { it.toKotlin() }
    is YamlMap -> entries.entries.associate { it.key.content to it.value.toKotlin() }
    else -> contentToString()
}
