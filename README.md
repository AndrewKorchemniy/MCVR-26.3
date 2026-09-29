# Minecraft Vulkan Renderer

MCVR is the C++ renderer used by [Radiance](https://github.com/Minecraft-Radiance/Radiance). It provides Vulkan world rendering, hardware ray tracing, reconstruction, and frame generation. Use the **0.1.6** renderer with the matching Radiance Java source branch.

The default UI backend is **OpenGL**. Minecraft draws its menus and HUD through OpenGL, and MCVR composites them with the Vulkan world renderer. Both APIs must use the same GPU. Player installation instructions, GPU selection guidance, and the FAQ are in the [Radiance README](https://github.com/Minecraft-Radiance/Radiance).

## Build

Prepare a C++23 compiler, CMake, a JDK matching the Java checkout, and Vulkan SDK shader tools with `glslangValidator` on `PATH`. Linux also needs the development packages required by GLFW and X11. The Windows release is built with Visual Studio 2022 / MSVC on x64.

Initialize the pinned dependencies:

```sh
git submodule update --init --recursive
```

Use the submodule commits recorded by this release, including the NRD and FidelityFX forks. They contain fixes used by Radiance. `cmake/PrepareSharc.cmake` prepares the SHARC headers required by the built-in shader packs.

Before configuring MCVR, run the following in the selected Radiance Java checkout to generate JNI headers:

```sh
./gradlew compileJava -PradianceUiBackend=opengl
```

On Windows, use `gradlew.bat`. Generate headers separately for each Minecraft version and loader.

### Linux

Replace `/path/to/Radiance` with the absolute path to that Java checkout:

```sh
cmake -S . -B build-linux-opengl \
  -DCMAKE_BUILD_TYPE=Release \
  -DJAVA_PROJECT_ROOT_DIR=/path/to/Radiance \
  -DMCVR_UI_BACKEND=OPENGL \
  -DMCVR_ENABLE_NRD=ON \
  -DUSE_AMD=ON
cmake --build build-linux-opengl --parallel
cmake --install build-linux-opengl
```

### Windows

Run in a Visual Studio developer shell, replacing `C:/path/to/Radiance` with the Java checkout path:

```powershell
cmake -S . -B build-windows-opengl -G "Visual Studio 17 2022" -A x64 -DJAVA_PROJECT_ROOT_DIR="C:/path/to/Radiance" -DMCVR_UI_BACKEND=OPENGL -DMCVR_ENABLE_NRD=ON -DUSE_AMD=ON
cmake --build build-windows-opengl --config Release --parallel
cmake --install build-windows-opengl --config Release
```

Installation writes the native library, shader resources, and built-in shader pack archives into the Java project's resources. Return to that checkout and run:

```sh
./gradlew build -PradianceUiBackend=opengl
```

The resulting mod JAR is under `build/libs/` in the Java checkout. Build each OS and Minecraft/loader combination with its corresponding native library and JNI headers.

### UI backend selection

| UI backend | MCVR / CMake | Radiance / Gradle |
| --- | --- | --- |
| OpenGL, default | `-DMCVR_UI_BACKEND=OPENGL` | `-PradianceUiBackend=opengl` |
| Vulkan | `-DMCVR_UI_BACKEND=VULKAN` | `-PradianceUiBackend=vulkan` |

This is a build option. Switching the in-game ray-tracing pipeline does not change the UI backend. Use separate native build directories and rebuild both components when changing it. The Java packaging tasks check that the native and Java UI variants match.

## Shader packs

MCVR installs the `vanilla-pt` and `advanced` built-in packs. The Advanced sources are under `src/shader/world/ray_tracing/internal/advanced/`; the standalone [Shaderpack-Advanced](https://github.com/Minecraft-Radiance/Shaderpack-Advanced) repository also includes the shared shader files needed to load the pack independently.

A complete pack includes its own sources, `src/shader/util`, `src/common`, and the prepared SHARC headers when used. Copying only the internal Advanced directory leaves required includes out of the pack.

## Runtime libraries

The mod packages do not include NVIDIA DLSS runtime binaries. Obtain those separately from NVIDIA if you want to use DLSS features; see the [installation instructions](https://github.com/Minecraft-Radiance/Radiance#optional-dlss-runtime-libraries). Available reconstruction and frame generation modes depend on the GPU, driver, platform, and installed runtimes.

## License

See [LICENSE.md](LICENSE.md). Third-party dependencies retain their own licenses.
