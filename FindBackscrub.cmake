# Helper for super-projects to use libbackscrub

# search for the built target export file
file(GLOB_RECURSE _bs_targets
	"${CMAKE_CURRENT_LIST_DIR}/*/BackscrubTargets.cmake")
list(LENGTH _bs_targets _bs_len)
if(NOT ${_bs_len})
	message(FATAL_ERROR "Unable to find BackscrubTargets.cmake in ${CMAKE_CURRENT_LIST_DIR}")
endif()
list(GET _bs_targets 0 _bs_target)
message(STATUS "Using Backscrub build: ${_bs_target}")
set(BACKSCRUB_FOUND "true")
include("${_bs_target}")
set(BACKSCRUB_INCLUDE "${CMAKE_CURRENT_LIST_DIR}")
set(BACKSCRUB_LIBS backscrub)
