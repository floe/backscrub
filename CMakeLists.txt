cmake_minimum_required(VERSION 3.16)
project(deepbacksub CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio highgui)

# This assumes that tensorflow got checked out as submodule.
# True, if `git clone` had the additional option `--recursive`.
add_subdirectory(tensorflow/tensorflow/lite 
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(deepseg
  deepseg.cc
  loopback.cc 
  transpose_conv_bias.cc
)

target_link_libraries(deepseg 
      tensorflow-lite ${CMAKE_DL_LIBS}
      opencv_core 
      opencv_video
      opencv_videoio
      opencv_imgproc
      opencv_imgcodecs
      opencv_highgui
)


install(TARGETS deepseg)
install(DIRECTORY models 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/deepbacksub 
    FILES_MATCHING PATTERN "*.tflite" PATTERN "*.md")
