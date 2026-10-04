package xaibo

import xaibo.primitives.*
import kotlin.test.*
import kotlinx.coroutines.runBlocking
import java.io.File

class FakeEmbedder : EmbeddingProtocol {
    // deterministic "bag of letters" vectors: shared letters -> shared direction
    override suspend fun textToEmbedding(text: String): DoubleArray {
        val v = DoubleArray(26)
        for (c in text.lowercase()) if (c in 'a'..'z') v[c - 'a']++
        return JsonVectorIndex.normalize(v)
    }

    override suspend fun imageToEmbedding(bytes: ByteArray) = error("not supported")
    override suspend fun audioToEmbedding(bytes: ByteArray) = error("not supported")
}

/** Assumptions of the vector-memory stack: chunking, indexing, persistence. */
class MemoryTest {

    private fun dir() = kotlin.io.path.createTempDirectory("xaibo-mem").toFile().apply { deleteOnExit() }

    @Test fun `store and semantically search memories`() {
        val d = dir()
        val memory = VectorMemory(
            WordChunker(), FakeEmbedder(), JsonVectorIndex(mapOf("storage_dir" to d.path)),
            mapOf("memory_file_path" to File(d, "memories.json").path),
        )
        runBlocking {
            val id1 = memory.storeMemory("I have a cat named Whiskers")
            memory.storeMemory("Dogs are loyal companions", mapOf("topic" to "dogs"))

            val hits = memory.searchMemory("cat whiskers", k = 3)
            assertEquals(id1, hits.first().memoryId)
            assertEquals("I have a cat named Whiskers", hits.first().content)

            // persistence: a fresh instance reads the same files back
            val reopened = VectorMemory(
                WordChunker(), FakeEmbedder(), JsonVectorIndex(mapOf("storage_dir" to d.path)),
                mapOf("memory_file_path" to File(d, "memories.json").path),
            )
            assertEquals(2, reopened.listMemories().size)

            assertTrue(reopened.updateMemory(id1, "I had a cat named Whiskers"))
            assertTrue(reopened.searchMemory("cat whiskers").first().content.startsWith("I had"))
            assertTrue(reopened.deleteMemory(id1))
            assertNull(reopened.getMemory(id1))
            assertEquals(1, reopened.listMemories().size)
        }
    }

    @Test fun `token chunker overlaps windows`() {
        val chunker = WordChunker(mapOf("window_size" to 4, "window_overlap" to 2))
        val chunks = runBlocking { chunker.chunk("a b c d e f g h") }
        // window 4, step = 4 - 2 = 2
        assertEquals(listOf("a b c d", "c d e f", "e f g h", "g h"), chunks)
        assertEquals(emptyList(), runBlocking { chunker.chunk("   ") })
    }

    @Test fun `vector index persists and searches by cosine`() {
        val d = dir()
        val index = JsonVectorIndex(mapOf("storage_dir" to d.path))
        runBlocking {
            index.addVectors(
                listOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0)),
                listOf(mapOf("tag" to "x"), mapOf("tag" to "y")),
            )
            val hits = index.search(doubleArrayOf(0.9, 0.1))
            assertEquals("x", hits.first().attributes?.get("tag"))
            assertFailsWith<IllegalArgumentException> { index.addVectors(listOf(doubleArrayOf(1.0, 2.0, 3.0))) }
        }
        // reloaded from disk
        val reloaded = JsonVectorIndex(mapOf("storage_dir" to d.path))
        assertEquals(2, runBlocking { reloaded.search(doubleArrayOf(0.0, 1.0)).size })
    }

    @Test fun `memory tool provider exposes the memory protocol as tools`() {
        val d = dir()
        val memory = VectorMemory(
            WordChunker(), FakeEmbedder(), JsonVectorIndex(mapOf("storage_dir" to d.path)),
            mapOf("memory_file_path" to File(d, "m.json").path),
        )
        val provider = MemoryToolProvider(memory)
        runBlocking {
            assertEquals(6, provider.listTools().size)
            val stored = provider.executeTool("store_memory", mapOf("text" to "the sky is blue"))
            assertTrue(stored.success)
            val found = provider.executeTool("search_memory", mapOf("query" to "blue sky", "k" to 1))
            assertTrue(found.success)
            assertEquals("the sky is blue", (found.result as List<*>).let {
                (it.first() as Map<*, *>)["content"]
            })
            assertFalse(provider.executeTool("bogus", emptyMap()).success)
        }
    }
}
