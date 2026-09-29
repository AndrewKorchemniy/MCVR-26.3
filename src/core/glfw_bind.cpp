#include "core/all_extern.hpp"
#include <cstdlib>
#include <iostream>

#if defined(CORE_LIB)
PFN_glfwInit p_glfwInit = nullptr;
PFN_glfwTerminate p_glfwTerminate = nullptr;
PFN_glfwGetWindowSize p_glfwGetWindowSize = nullptr;
PFN_glfwCreateWindowSurface p_glfwCreateWindowSurface = nullptr;
PFN_glfwGetRequiredInstanceExtensions p_glfwGetRequiredInstanceExtensions = nullptr;
PFN_glfwSetWindowTitle p_glfwSetWindowTitle = nullptr;
PFN_glfwSetFramebufferSizeCallback p_glfwSetFramebufferSizeCallback = nullptr;
PFN_glfwGetFramebufferSize p_glfwGetFramebufferSize = nullptr;
PFN_glfwWaitEvents p_glfwWaitEvents = nullptr;
PFN_glfwGetPlatform p_glfwGetPlatform = nullptr;
#    if defined(_WIN32)
PFN_glfwGetWin32Window p_glfwGetWin32Window = nullptr;
#    elif defined(__linux__) || defined(__unix__)
PFN_glfwGetX11Display p_glfwGetX11Display = nullptr;
PFN_glfwGetX11Window p_glfwGetX11Window = nullptr;
PFN_glfwGetWaylandDisplay p_glfwGetWaylandDisplay = nullptr;
PFN_glfwGetWaylandWindow p_glfwGetWaylandWindow = nullptr;
#    endif
#endif
