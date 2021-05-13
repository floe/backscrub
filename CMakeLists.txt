cmake_minimum_required(VERSION 3.16)

project (backscrub)

add_definitions(-D TFLITE_BUILD_WITH_XNNPACK_DELEGATE)

include(tensorflow/tensorflow/lite/CMakeLists.txt)

find_package(OpenCV REQUIRED)

add_executable(backscrub deepseg.cc loopback.cc transpose_conv_bias.cc)
target_link_libraries(backscrub tensorflow-lite ${OpenCV_LIBS} dl)
