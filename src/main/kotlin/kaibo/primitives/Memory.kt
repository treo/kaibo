package kaibo.primitives

import kaibo.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.*
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.UUID

// -- memory protocols ----------------------------------------------------------

/** Chunks text for embedding into a vector space */
interface ChunkingProtocol { suspend fun chunk(text: String): List<String> }

/** Embeds modalities into a vector space */
interface EmbeddingProtocol {
    suspend fun textToEmbedding(text: String): DoubleArray
    suspend fun imageToEmbedding(bytes: ByteArray): DoubleArray
    suspend fun audioToEmbedding(bytes: ByteArray): DoubleArray
}

data class VectorSearchResult(val vectorId: String, val similarityScore: Double, val attributes: Map<String, Any?>? = null)

/** Stores and searches normalized vectors */
interface VectorIndexProtocol {
    suspend fun addVectors(vectors: List<DoubleArray>, attributes: List<Map<String, Any?>>? = null)
    suspend fun search(queryVector: DoubleArray, k: Int = 10): List<VectorSearchResult>
}

data class MemorySearchResult(
    val memoryId: String,
    val content: String,
    val similarityScore: Double,
    val attributes: Map<String, Any?>? = null,
)

/** Store/retrieve/semantically search text memories */
interface MemoryProtocol {
    suspend fun storeMemory(text: String, attributes: Map<String, Any?>? = null): String
    suspend fun getMemory(memoryId: String): Map<String, Any?>?
    suspend fun searchMemory(query: String, k: Int = 10): List<MemorySearchResult>
    suspend fun listMemories(): List<Map<String, Any?>>
    suspend fun deleteMemory(memoryId: String): Boolean
    suspend fun updateMemory(memoryId: String, text: String, attributes: Map<String, Any?>? = null): Boolean
}

// -- implementations -------------------------------------------------------------

/**
 * Splits text into overlapping chunks measured in words — word counts
 * approximate tokens closely enough to bound chunk size for embedding.
 * Config: `window_size` (512 words), `window_overlap` (50 words).
 */
class WordChunker(val config: Map<String, Any?> = emptyMap()) : ChunkingProtocol {
    private val windowSize = (config["window_size"] as? Number)?.toInt() ?: 512
    private val windowOverlap = (config["window_overlap"] as? Number)?.toInt() ?: 50

    override suspend fun chunk(text: String): List<String> {
        if (text.isBlank()) return emptyList()
        val words = text.split(" ")
        if (words.size <= windowSize) return listOf(text)
        val step = maxOf(1, windowSize - windowOverlap)
        return (words.indices step step).map { i -> words.subList(i, minOf(i + windowSize, words.size)).joinToString(" ") }
    }
}

/**
 * Brute-force cosine similarity over unit-normalized vectors, persisted as one
 * JSON file per `storage_dir`. The search is a linear scan — fine for
 * thousands to tens of thousands of chunks; when you outgrow that, swap a
 * real ANN engine in behind the same [VectorIndexProtocol].
 */
class JsonVectorIndex(val config: Map<String, Any?> = emptyMap()) : VectorIndexProtocol {

    private val storageFile = File(
        config["storage_dir"] as? String ?: error("storage_dir is required in config"),
        "index.json",
    )
    private val json = Json { prettyPrint = false; ignoreUnknownKeys = true }
    private var dimension: Int? = null

    private data class Entry(val vector: DoubleArray, val attributes: Map<String, Any?>)

    private val entries: MutableList<Entry> = mutableListOf()

    init { load() }

    private fun load() {
        if (!storageFile.isFile) return
        val root = json.parseToJsonElement(storageFile.readText()).jsonObject
        dimension = root.getIntOrZero("dimension").takeIf { it > 0 }
        root.arr("entries")?.forEach { e ->
            val o = e.jsonObject
            entries.add(Entry(
                o.arr("vector")!!.map { (it as JsonPrimitive).double }.toDoubleArray(),
                o.obj("attributes")?.toKotlin()?.let { @Suppress("UNCHECKED_CAST") (it as Map<String, Any?>) } ?: emptyMap(),
            ))
        }
    }

    private fun save() {
        storageFile.parentFile?.mkdirs()
        val body = buildJsonObject {
            put("dimension", dimension?.let { JsonPrimitive(it) } ?: JsonNull)
            put("entries", JsonArray(entries.map { e ->
                buildJsonObject {
                    put("vector", JsonArray(e.vector.map { JsonPrimitive(it) }))
                    put("attributes", e.attributes.toJsonElement())
                }
            }))
        }
        storageFile.writeText(json.encodeToString(JsonElement.serializer(), body))
    }

    override suspend fun addVectors(vectors: List<DoubleArray>, attributes: List<Map<String, Any?>>?) {
        val attrs = attributes ?: List(vectors.size) { emptyMap() }
        require(vectors.size == attrs.size) { "Number of vectors and attributes must match" }
        for (v in vectors) {
            dimension?.let { require(v.size == it) { "Vector dimension mismatch. Expected $it, got ${v.size}" } }
                ?: run { dimension = v.size }
        }
        vectors.forEachIndexed { i, v -> entries.add(Entry(normalize(v), attrs[i])) }
        save()
    }

    override suspend fun search(queryVector: DoubleArray, k: Int): List<VectorSearchResult> {
        val q = normalize(queryVector)
        return entries.mapIndexed { i, e -> i to cosine(q, e.vector) }
            .sortedByDescending { it.second }
            .take(k)
            .map { (i, score) -> VectorSearchResult("$i", score, entries[i].attributes) }
    }

    /** Clears a memory's chunks from the index (used by VectorMemory updates) */
    internal fun removeByAttribute(key: String, value: Any?) {
        entries.removeAll { it.attributes[key] == value }
        save()
    }

    companion object {
        fun normalize(v: DoubleArray): DoubleArray {
            val norm = kotlin.math.sqrt(v.sumOf { it * it })
            return if (norm > 0) DoubleArray(v.size) { v[it] / norm } else v
        }
        private fun cosine(a: DoubleArray, b: DoubleArray) =
            if (a.size != b.size) 0.0 else a.zip(b.toList()).sumOf { (x, y) -> x * y }
    }
}

/**
 * OpenAI text embeddings.
 * Config: `api_key` (or OPENAI_API_KEY), `model` (text-embedding-3-small),
 * `base_url`. Image/audio embedding is not offered by this endpoint.
 */
class OpenAIEmbedder(val config: Map<String, Any?> = emptyMap()) : EmbeddingProtocol {
    private val apiKey = (config["api_key"] as? String) ?: System.getenv("OPENAI_API_KEY")
        ?: error("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
    private val model = config["model"] as? String ?: "text-embedding-3-small"
    private val baseUrl = (config["base_url"] as? String ?: "https://api.openai.com/v1").trimEnd('/')
    private val client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(60)).build()
    private val json = Json { ignoreUnknownKeys = true }

    override suspend fun textToEmbedding(text: String): DoubleArray = withContext(Dispatchers.IO) {
        val body = mapOf("model" to model, "input" to text).toJsonElement().toString()
        val resp = client.send(
            HttpRequest.newBuilder(URI("$baseUrl/embeddings"))
                .timeout(Duration.ofSeconds(60))
                .header("Authorization", "Bearer $apiKey")
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)).build(),
            HttpResponse.BodyHandlers.ofString(),
        )
        json.parseToJsonElement(resp.body()).jsonObject.arr("data")?.firstOrNull()?.asObject()
            ?.arr("embedding")?.map { (it as JsonPrimitive).double }?.toDoubleArray()
            ?: error("Embedding API returned no data: ${resp.body().take(300)}")
    }

    override suspend fun imageToEmbedding(bytes: ByteArray) =
        throw UnsupportedOperationException("the text embeddings endpoint does not accept images")

    override suspend fun audioToEmbedding(bytes: ByteArray) =
        throw UnsupportedOperationException("the text embeddings endpoint does not accept audio")
}

/**
 * Semantic memory built from chunker + embedder + vector index. Memories are
 * persisted as JSON next to the index.
 * Config: `memory_file_path` (required).
 */
class VectorMemory(
    val chunker: ChunkingProtocol,
    val embedder: EmbeddingProtocol,
    val vectorIndex: VectorIndexProtocol,
    val config: Map<String, Any?> = emptyMap(),
) : MemoryProtocol {

    private val memoryFile = File(config["memory_file_path"] as? String ?: error("memory_file_path is required in config"))
    private val json = Json { prettyPrint = true }

    // id -> {id, content, attributes}
    private var memories: MutableMap<String, MutableMap<String, Any?>> = mutableMapOf()

    init {
        if (memoryFile.isFile) {
            memories = (json.parseToJsonElement(memoryFile.readText()).toKotlin()
                as Map<String, MutableMap<String, Any?>>).toMutableMap()
        } else save()
    }

    private fun save() {
        memoryFile.parentFile?.mkdirs()
        memoryFile.writeText(json.encodeToString(JsonElement.serializer(), memories.toJsonElement()))
    }

    override suspend fun storeMemory(text: String, attributes: Map<String, Any?>?): String =
        storeWithId(UUID.randomUUID().toString(), text, attributes)

    private suspend fun storeWithId(memoryId: String, text: String, attributes: Map<String, Any?>?): String {
        memories[memoryId] = mutableMapOf("id" to memoryId, "content" to text, "attributes" to (attributes ?: emptyMap()))
        save()

        val chunks = chunker.chunk(text)
        val vectors = chunks.map { embedder.textToEmbedding(it) }
        val chunkAttributes = chunks.mapIndexed { i, chunk ->
            mapOf<String, Any?>("memory_id" to memoryId, "chunk_index" to i, "chunk_text" to chunk) +
                (attributes ?: emptyMap<String, Any?>())
        }
        vectorIndex.addVectors(vectors, chunkAttributes)
        return memoryId
    }

    override suspend fun getMemory(memoryId: String): Map<String, Any?>? = memories[memoryId]

    override suspend fun searchMemory(query: String, k: Int): List<MemorySearchResult> {
        val queryVector = embedder.textToEmbedding(query)
        val seen = mutableSetOf<String>()
        return vectorIndex.search(queryVector, k).mapNotNull { result ->
            val memoryId = result.attributes?.get("memory_id") as? String
            val memory = memoryId?.let { memories[it] }
            if (memoryId == null || memoryId in seen || memory == null) null
            else {
                seen += memoryId
                MemorySearchResult(
                    memoryId, memory["content"] as String? ?: "",
                    result.similarityScore, memory["attributes"] as? Map<String, Any?>,
                )
            }
        }
    }

    override suspend fun listMemories(): List<Map<String, Any?>> = memories.values.toList()

    override suspend fun deleteMemory(memoryId: String): Boolean {
        if (memoryId !in memories) return false
        memories.remove(memoryId)
        save()
        (vectorIndex as? JsonVectorIndex)?.removeByAttribute("memory_id", memoryId)
        return true
    }

    /** Delete-then-insert with the same id, merging attributes */
    override suspend fun updateMemory(memoryId: String, text: String, attributes: Map<String, Any?>?): Boolean {
        val old = memories[memoryId] ?: return false
        val merged = (old["attributes"] as? Map<String, Any?>).orEmpty() + (attributes ?: emptyMap())
        deleteMemory(memoryId)
        storeWithId(memoryId, text, merged)
        return true
    }
}

/** Exposes a MemoryProtocol to the agent as the six memory tools. */
class MemoryToolProvider(
    val memory: MemoryProtocol,
    @Suppress("unused_constructor_parameter") val config: Map<String, Any?> = emptyMap(),
) : ToolProviderProtocol {

    private fun p(name: String, type: String, required: Boolean = false) = name to ToolParameter(type, required = required)

    private val tools = listOf(
        Tool("store_memory", "Store a new memory", mapOf(p("text", "string", true), p("attributes", "object"))),
        Tool("get_memory", "Retrieve a memory by id", mapOf(p("memory_id", "string", true))),
        Tool("search_memory", "Search memories semantically", mapOf(p("query", "string", true), p("k", "integer"))),
        Tool("list_memories", "List all memories"),
        Tool("delete_memory", "Delete a memory by id", mapOf(p("memory_id", "string", true))),
        Tool("update_memory", "Update a memory", mapOf(p("memory_id", "string", true), p("text", "string", true))),
    )

    override suspend fun listTools() = tools

    @Suppress("UNCHECKED_CAST")
    override suspend fun executeTool(toolName: String, parameters: Map<String, Any?>): ToolResult = try {
        when (toolName) {
            "store_memory" -> ToolResult(true, result = mapOf("memory_id" to
                memory.storeMemory(parameters["text"] as String, parameters["attributes"] as? Map<String, Any?>)))
            "get_memory" -> memory.getMemory(parameters["memory_id"] as String)
                ?.let { ToolResult(true, result = it) } ?: ToolResult(false, error = "Memory not found")
            "search_memory" -> ToolResult(true, result = memory.searchMemory(
                parameters["query"] as String, (parameters["k"] as? Number)?.toInt() ?: 10
            ).map { mapOf("memory_id" to it.memoryId, "content" to it.content, "similarity_score" to it.similarityScore) })
            "list_memories" -> ToolResult(true, result = memory.listMemories())
            "delete_memory" -> ToolResult(true, result = mapOf("deleted" to memory.deleteMemory(parameters["memory_id"] as String)))
            "update_memory" -> ToolResult(true, result = mapOf("updated" to memory.updateMemory(
                parameters["memory_id"] as String, parameters["text"] as String,
                parameters["attributes"] as? Map<String, Any?>)))
            else -> ToolResult(false, error = "Unknown tool: $toolName")
        }
    } catch (e: Exception) {
        ToolResult(false, error = "Error executing tool $toolName: ${e.message}")
    }
}
