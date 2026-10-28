#pragma once

namespace mcvr {
// Prints the faulting address and a symbolicated stack to stderr on fatal exceptions (Windows only), since the JVM
// does not always produce an hs_err file for crashes on native threads.
void installCrashHandler();
}
