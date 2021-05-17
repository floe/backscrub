cmake_minimum_required(VERSION 3.16)

project (backscrub)

# generate GPU delegate schema with flatc
# - for which we provide a custom command and a custom target
set(GEN_HEADER_PATH ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/lite/delegates/gpu/cl)
add_custom_command(OUTPUT ${GEN_HEADER_PATH}/serialization_generated.h
    COMMAND flatc -c --scoped-enums -o ${GEN_HEADER_PATH}/ ${GEN_HEADER_PATH}/serialization.fbs
    COMMENT "Compiling schema ${GEN_HEADER_PATH}/serialization.fbs"
    DEPENDS ${GEN_HEADER_PATH}/serialization.fbs)
add_custom_target(gen-headers ALL
    DEPENDS ${GEN_HEADER_PATH}/serialization_generated.h)

# Force use of XNNPACK on CPU, without this the weak/strong symbol
# reference trickery fails with gcc and static libraries.. *sigh*
add_definitions(-D TFLITE_BUILD_WITH_XNNPACK_DELEGATE)

# Enable GPU delegate build, supporting OpenCL/OpenGl/Vulkan APIs
option(TFLITE_ENABLE_GPU "Enable GPU" ON)
# Assume v2.0 OpenCL, avoid MESA X11 headers (they collide with Abseil)
add_definitions(-D CL_TARGET_OPENCL_VERSION=200 -D MESA_EGL_NO_X11_HEADERS)

# Go build Tensorflow-lite
add_subdirectory(tensorflow/tensorflow/lite)

# add dependency from tensorflow-lite library to generated headers custom target
add_dependencies(tensorflow-lite gen-headers)

find_package(OpenCV REQUIRED)

add_executable(backscrub deepseg.cc loopback.cc transpose_conv_bias.cc)
target_link_libraries(backscrub tensorflow-lite ${OpenCV_LIBS} dl)
