# iSAM master cmake file
# Modified by Humphrey Hu, 2015
# Original by Michael Kaess, 2010

project(isam)

cmake_minimum_required (VERSION 2.6)

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
set(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

# Set install paths
set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib" CACHE PATH
	"Target directory for all libraries.")
set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin" CACHE PATH
	"Target directory for all executables.")

set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include/isam")
set(LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(BINARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")
	
# do not edit - use ccmake to change
option (PROFILE "Enable profiling" OFF)
option (USE_LCM "Compile with LCM interface (lcm library needed)" OFF)
if(NOT DEFINED USE_GUI)
# SDL is optional
find_package(SDL)
if (NOT SDL_FOUND)
	message(WARNING "\nSDL not found, disabling GUI.")
	set(USE_GUI_TMP OFF)
else(NOT SDL_FOUND)
	set(USE_GUI_TMP ON)
	list(APPEND isam_LIBS ${SDL_LIBRARY})
endif()
option (USE_GUI "Compile with GUI (SDL library needed)" ${USE_GUI_TMP})
endif()

add_definitions(-Wall)
# note: -g automatic for Debug mode, -O3 -DNDEBUG automatic for Release

set(isam_VERSION_MAJOR 1)
set(isam_VERSION_MINOR 7)
set(isam_VERSION "${isam_VERSION_MAJOR}.${isam_VERSION_MINOR}")

if(PROFILE)
message(STATUS "Profiling active")
add_definitions(-pg)
endif(PROFILE)

if(USE_LCM)
add_definitions(-DUSE_LCM)
endif(USE_LCM)
if(USE_GUI)
add_definitions(-DUSE_GUI)
endif(USE_GUI)

# Eigen3 is needed
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
list(APPEND isam_INCLUDES "${EIGEN3_INCLUDE_DIR}")

include_directories("${PROJECT_SOURCE_DIR}/include")

add_subdirectory(isamlib)
add_subdirectory(isam)
add_subdirectory(examples)
add_subdirectory(misc)

export(TARGETS isam
	FILE "${PROJECT_BINARY_DIR}/isamTargets.cmake")
	
configure_file(isamConfig.cmake.in
	"${PROJECT_BINARY_DIR}/isamConfig.cmake" @ONLY)
	
configure_file(isamConfigVersion.cmake.in
	"${PROJECT_BINARY_DIR}/isamConfigVersion.cmake" @ONLY)
	
install(FILES
	"${PROJECT_BINARY_DIR}/isamConfig.cmake"
	"${PROJECT_BINARY_DIR}/isamConfigVersion.cmake"
	DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT dev)
	
install(EXPORT isamTargets
	DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT dev)
