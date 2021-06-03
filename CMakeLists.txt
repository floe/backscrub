cmake_minimum_required(VERSION 3.16)
project(backscrub CXX)

if(NOT DEFINED TENSORFLOW)
  set(TENSORFLOW tensorflow)
endif()

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

# always build PIC everywhere
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio highgui)

# force compilation of XNNPACK delegate (without this the too clever-by-half use
# of weak/strong symbol linking fails in a static library)
add_compile_definitions(TFLITE_BUILD_WITH_XNNPACK_DELEGATE)

# This assumes that tensorflow got checked out as submodule.
# True, if `git clone` had the additional option `--recursive`.
add_subdirectory(${TENSORFLOW}/tensorflow/lite
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_compile_definitions(DEEPSEG_VERSION=${DEEPSEG_VERSION})
include_directories(BEFORE .)

add_library(backscrub
  lib/libbackscrub.cc
  lib/transpose_conv_bias.cc)

target_link_libraries(backscrub
  tensorflow-lite ${CMAKE_DL_LIBS}
  opencv_core
  opencv_imgproc
)

add_library(videoio
  videoio/loopback.cc)

target_link_libraries(videoio)

add_executable(deepseg
  app/deepseg.cc
)

target_link_libraries(deepseg
  backscrub
  videoio
  opencv_core
  opencv_video
  opencv_videoio
  opencv_imgproc
  opencv_imgcodecs
  opencv_highgui
)

# Export our library, and all transitive dependencies - sadly Tensorflow Lite's
# CMakeLists.txt does not export these for us
function(get_link_libraries target outlist)
  message(STATUS "get_link_libraries(${target} ${outlist})")
  # recursive dependency munger, feed with a name and a list to extend
  get_target_property(target_type ${target} TYPE)
  if(${target_type} STREQUAL "INTERFACE_LIBRARY")
    get_target_property(libs ${target} INTERFACE_LINK_LIBRARIES)
  else()
    get_target_property(libs ${target} LINK_LIBRARIES)
  endif()
  foreach(lib IN LISTS libs)
    if(NOT TARGET ${lib})
      continue()
    endif()
    get_target_property(unalias ${lib} ALIASED_TARGET)
	if("${unalias}" STREQUAL "unalias-NOTFOUND")
      set(unalias ${lib})
    endif()
	get_target_property(imp ${unalias} IMPORTED)
	if(${imp})
	  continue()
	endif()
    list(FIND ${outlist} ${unalias} exists)
    if(NOT exists EQUAL -1)
      continue()
    endif()
    list(APPEND ${outlist} ${unalias})
    get_link_libraries(${unalias} ${outlist})
    set(${outlist} ${${outlist}} PARENT_SCOPE)
  endforeach()
endfunction()

set(BACKSCRUB_DEPS "")
get_link_libraries(backscrub BACKSCRUB_DEPS)

export(TARGETS
  backscrub
  ${BACKSCRUB_DEPS}
  FILE BackscrubTargets.cmake)

install(TARGETS deepseg)
install(TARGETS backscrub)
install(FILES libbackscrub.h DESTINATION include/backscrub)
install(DIRECTORY models
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/backscrub
    FILES_MATCHING PATTERN "*.tflite" PATTERN "*.md")
