#include "core/util/crash_handler.hpp"

#if defined(_WIN32)
#    include <windows.h>
#    include <dbghelp.h>

#    include <cstdio>
#    include <cstdlib>

#    pragma comment(lib, "dbghelp.lib")

namespace {
LONG WINAPI mcvrCrashFilter(EXCEPTION_POINTERS *info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    // Only fatal exceptions: skip C++ exceptions, debugger notifications and the like.
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_STACK_OVERFLOW && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
        code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != EXCEPTION_IN_PAGE_ERROR) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // The JVM implements null checks with access violations in JIT-compiled code, which lies in no module.
    if (info->ExceptionRecord->ExceptionAddress != nullptr) {
        HMODULE faultModule = nullptr;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCSTR>(info->ExceptionRecord->ExceptionAddress), &faultModule)) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
    }

    static volatile LONG entered = 0;
    if (InterlockedExchange(&entered, 1) != 0) { return EXCEPTION_CONTINUE_SEARCH; }

    std::fprintf(stderr, "[MCVR] fatal exception 0x%08lx at %p on thread %lu", code,
                 info->ExceptionRecord->ExceptionAddress, GetCurrentThreadId());
    if (code == EXCEPTION_ACCESS_VIOLATION && info->ExceptionRecord->NumberParameters >= 2) {
        std::fprintf(stderr, " (%s address 0x%llx)", info->ExceptionRecord->ExceptionInformation[0] == 0 ? "reading" :
                     info->ExceptionRecord->ExceptionInformation[0] == 1 ? "writing" : "executing",
                     static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[1]));
    }
    std::fprintf(stderr, "\n");

    HANDLE process = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(process, nullptr, TRUE);

    CONTEXT context = *info->ContextRecord;
    if (context.Rip == 0 && !IsBadReadPtr(reinterpret_cast<void *>(context.Rsp), sizeof(DWORD64))) {
        // A call through a null function pointer: unwind from the return address pushed by the call.
        context.Rip = *reinterpret_cast<DWORD64 *>(context.Rsp);
        context.Rsp += sizeof(DWORD64);
    }
    STACKFRAME64 frame{};
    frame.AddrPC.Offset = context.Rip;
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Offset = context.Rbp;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = context.Rsp;
    frame.AddrStack.Mode = AddrModeFlat;

    for (int i = 0; i < 48; i++) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, GetCurrentThread(), &frame, &context, nullptr,
                         SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
            break;
        }
        if (frame.AddrPC.Offset == 0) { break; }
        const DWORD64 address = frame.AddrPC.Offset;

        char buffer[sizeof(SYMBOL_INFO) + 256]{};
        auto *symbol = reinterpret_cast<SYMBOL_INFO *>(buffer);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 255;
        DWORD64 displacement = 0;
        IMAGEHLP_MODULE64 module{};
        module.SizeOfStruct = sizeof(module);
        const bool hasModule = SymGetModuleInfo64(process, address, &module) != FALSE;
        char moduleName[MAX_PATH] = "?";
        HMODULE moduleHandle = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(address), &moduleHandle)) {
            GetModuleFileNameA(moduleHandle, moduleName, MAX_PATH);
        }

        std::fprintf(stderr, "[MCVR]   #%02d 0x%016llx %s+0x%llx", i, static_cast<unsigned long long>(address),
                     moduleName,
                     moduleHandle != nullptr ?
                         static_cast<unsigned long long>(address - reinterpret_cast<DWORD64>(moduleHandle)) :
                         0ull);
        (void)hasModule;
        if (SymFromAddr(process, address, &displacement, symbol)) {
            std::fprintf(stderr, " %s+0x%llx", symbol->Name, static_cast<unsigned long long>(displacement));
        }
        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(line);
        DWORD lineDisplacement = 0;
        if (SymGetLineFromAddr64(process, address, &lineDisplacement, &line)) {
            std::fprintf(stderr, " (%s:%lu)", line.FileName, line.LineNumber);
        }
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
}

void mcvr::installCrashHandler() {
    static bool installed = false;
    if (installed) { return; }
    installed = true;
    AddVectoredExceptionHandler(1, mcvrCrashFilter);
}
#else
void mcvr::installCrashHandler() {}
#endif
