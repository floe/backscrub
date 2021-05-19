cmake_minimum_required(VERSION 3.16)
project(backscrub CXX)

find_package(Git)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --all --long --always --dirty
       OUTPUT_STRIP_TRAILING_WHITESPACE
       OUTPUT_VARIABLE DEEPSEG_VERSION)
else()
  set(DEEPSEG_VERSION v0.2.0-no-git)
endif()
message("Version: ${DEEPSEG_VERSION}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio highgui)

# This assumes that tensorflow got checked out as submodule.
# True, if `git clone` had the additional option `--recursive`.
add_subdirectory(tensorflow/tensorflow/lite 
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_compile_definitions(DEEPSEG_VERSION=${DEEPSEG_VERSION})

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
