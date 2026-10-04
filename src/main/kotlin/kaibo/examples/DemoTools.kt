package kaibo.examples

import kaibo.primitives.KaiboParam
import kaibo.primitives.KaiboTool
import java.time.LocalDateTime

/** Example annotated tools, referenced by the demo agent configs. */
class DemoTools {
    @KaiboTool(name = "multiply", description = "Multiplies two integers")
    fun multiply(@KaiboParam("First factor") a: Int, @KaiboParam("Second factor") b: Int) = a * b

    @KaiboTool(name = "reverse", description = "Reverses the given text")
    fun reverse(@KaiboParam("The text to reverse") text: String) = text.reversed()

    @KaiboTool(name = "current_time", description = "Current date and time")
    fun currentTime() = LocalDateTime.now().toString()
}
