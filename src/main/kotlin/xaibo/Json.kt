package xaibo

import kotlinx.serialization.json.*

/**
 * Minimal JSON <-> plain-Kotlin bridging. LLM payloads are maps read from
 * YAML/JSON, so we convert explicitly instead of dragging serializers through
 * the whole model layer.
 */
fun JsonElement.toKotlin(): Any? = when (this) {
    is JsonNull -> null
    is JsonPrimitive -> run {
        when {
            isString -> content
            booleanOrNull != null -> boolean
            longOrNull != null && longOrNull in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong() -> int
            longOrNull != null -> long
            doubleOrNull != null -> double
            else -> content
        }
    }
    is JsonArray -> map { it.toKotlin() }
    is JsonObject -> map { (k, v) -> k to v.toKotlin() }.toMap()
}

@Suppress("UNCHECKED_CAST")
fun Any?.toJsonElement(): JsonElement = when (this) {
    null -> JsonNull
    is JsonElement -> this
    is String -> JsonPrimitive(this)
    is Boolean -> JsonPrimitive(this)
    is Number -> JsonPrimitive(this)
    is Map<*, *> -> JsonObject(entries.associate { (k, v) -> k.toString() to v.toJsonElement() })
    is List<*> -> JsonArray(map { it.toJsonElement() })
    is Array<*> -> JsonArray(map { it.toJsonElement() })
    else -> JsonPrimitive(toString())
}

fun Any?.jsonString(): String = Json.encodeToString(JsonElement.serializer(), toJsonElement())

/** Coerce a value into JSON-safe plain form (like Python json.loads(json.dumps(v, default=repr))) */
fun Any?.jsonSafe(): Any? = try {
    Json.parseToJsonElement(jsonString()).toKotlin()
} catch (e: Exception) {
    toString()
}

@Suppress("UNCHECKED_CAST")
fun Map<String, Any?>.str(key: String) = this[key] as? String

/** Parse tool-call argument JSON tolerantly: "" -> {}, control chars allowed */
fun parseToolArguments(raw: String?): Map<String, Any?>? {
    if (raw.isNullOrEmpty()) return emptyMap()
    return try {
        Json { isLenient = true }.parseToJsonElement(raw).toKotlin() as? Map<String, Any?>
    } catch (e: Exception) {
        null
    }
}
