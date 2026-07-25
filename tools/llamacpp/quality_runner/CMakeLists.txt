cmake_minimum_required(VERSION 3.16)

project(nanoquant_llamacpp_quality LANGUAGES CXX)

if(NOT DEFINED LLAMA_CPP_ROOT)
    message(FATAL_ERROR "LLAMA_CPP_ROOT must point at the NanoQuant llama.cpp checkout")
endif()

get_filename_component(LLAMA_CPP_ROOT "${LLAMA_CPP_ROOT}" ABSOLUTE)
set(LLAMA_INCLUDE_DIR "${LLAMA_CPP_ROOT}/include")

if(WIN32)
    set(_llama_library_hints
        "${LLAMA_CPP_ROOT}/build/src/Release"
        "${LLAMA_CPP_ROOT}/build/src"
    )
else()
    set(_llama_library_hints
        "${LLAMA_CPP_ROOT}/build/src"
        "${LLAMA_CPP_ROOT}/build/bin"
    )
endif()

find_library(
    LLAMA_LIBRARY
    NAMES llama
    HINTS ${_llama_library_hints}
    NO_DEFAULT_PATH
    REQUIRED
)

add_executable(nanoquant-llamacpp-quality main.cpp)
target_compile_features(nanoquant-llamacpp-quality PRIVATE cxx_std_17)
target_include_directories(
    nanoquant-llamacpp-quality
    PRIVATE
        "${LLAMA_INCLUDE_DIR}"
        "${LLAMA_CPP_ROOT}/ggml/include"
)
target_link_libraries(nanoquant-llamacpp-quality PRIVATE "${LLAMA_LIBRARY}")

if(MSVC)
    target_compile_options(nanoquant-llamacpp-quality PRIVATE /W4 /EHsc)
else()
    target_compile_options(nanoquant-llamacpp-quality PRIVATE -Wall -Wextra -Wpedantic)
endif()
