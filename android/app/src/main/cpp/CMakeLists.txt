cmake_minimum_required(VERSION 3.22)
project(mobile_rvc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Contents of the onnxruntime-android AAR (headers/, jni/<abi>/libonnxruntime.so),
# extracted by the Gradle task extractOrt<Flavor>.
set(ORT_ROOT "" CACHE PATH "Extracted onnxruntime AAR")
if(NOT EXISTS "${ORT_ROOT}/headers/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ORT_ROOT does not contain onnxruntime headers: ${ORT_ROOT}")
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/headers")

set(RVC_OPT -O3 -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter)

add_library(rvc_engine STATIC
  engine/conv_f32.cpp
  engine/conv_f16.cpp
  engine/dsp.cpp
  engine/ivf_index.cpp
  engine/model_import.cpp
  engine/native_decoder.cpp
  engine/noise_gate.cpp
  engine/ort_model.cpp
  engine/pth_reader.cpp
  engine/rvc_engine.cpp)
target_compile_options(rvc_engine PRIVATE ${RVC_OPT})
if(ANDROID_ABI STREQUAL "arm64-v8a")
  # fp16 kernels are only executed after a runtime HWCAP check
  set_source_files_properties(engine/conv_f16.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
endif()
target_link_libraries(rvc_engine PUBLIC onnxruntime log)

add_library(rvc SHARED android/jni_bridge.cpp android/rt_driver.cpp)
target_compile_options(rvc PRIVATE ${RVC_OPT})
target_link_libraries(rvc PRIVATE rvc_engine aaudio android log)
target_link_options(rvc PRIVATE -Wl,--gc-sections)

# Command line tool for `adb shell` benchmarks (not packaged in the APK).
add_executable(rvc_cli tools/rvc_cli.cpp android/rt_driver.cpp)
target_compile_options(rvc_cli PRIVATE ${RVC_OPT})
target_link_libraries(rvc_cli PRIVATE rvc_engine aaudio android log)
