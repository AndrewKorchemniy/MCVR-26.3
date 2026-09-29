#pragma once

#if defined(_WIN32) && defined(__cplusplus)
#    undef JNIEXPORT
#    define JNIEXPORT extern "C" __declspec(dllexport)
#endif
