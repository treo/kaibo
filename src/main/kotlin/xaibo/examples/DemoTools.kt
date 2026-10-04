package xaibo.examples

import xaibo.primitives.XaiboParam
import xaibo.primitives.XaiboTool
import java.time.LocalDateTime

/** Example annotated tools, referenced by the demo agent configs. */
class DemoTools {
    @XaiboTool(name = "multiply", description = "Multiplies two integers")
    fun multiply(@XaiboParam("First factor") a: Int, @XaiboParam("Second factor") b: Int) = a * b

    @XaiboTool(name = "reverse", description = "Reverses the given text")
    fun reverse(@XaiboParam("The text to reverse") text: String) = text.reversed()

    @XaiboTool(name = "current_time", description = "Current date and time")
    fun currentTime() = LocalDateTime.now().toString()
}
