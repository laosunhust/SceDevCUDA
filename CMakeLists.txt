cmake_minimum_required (VERSION 2.8.7)
set(PROJECT_NAME_STR beakDevGPU)
project(${PROJECT_NAME_STR} C CXX)

if(CMAKE_COMPILER_IS_GNUCXX)
        add_definitions(-Wall -ansi -Wno-deprecated -pthread -fpermissive)
endif()

SET(EXECUTABLE_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_PATH})

#cuda files and corresponding .h files are placed together in this folder
set(COMMON_CUDA_INCLUDES ${PROJECT_SOURCE_DIR}/srcGPU)
#cpp files and corresponding .h files are placed together in this folder
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/srcCPU)
# source code dir for GPU code
set(GPU_SOURCE_DIR ${PROJECT_SOURCE_DIR}/srcGPU)
# source code dir for CPU code
set(CPU_SOURCE_DIR ${PROJECT_SOURCE_DIR}/srcCPU)
#folder for external dependencies
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)
#set(PROJECT_EXE_DIR ${PROJECT_SOURCE_DIR}/exeFiles)
#location of executables
SET(CMAKE_BUILD_DIR ${CMAKE_SOURCE_DIR}/bin)
#location of unit test files
SET(CMAKE_UNITTEST_DIR ${CMAKE_SOURCE_DIR}/unitTest)
#location of model test files
SET(CMAKE_MODELTEST_DIR ${CMAKE_SOURCE_DIR}/modelTest)


# make VERBOSE =1

#find_path(STL_INCLUDE_PATH exception)
find_path(STDIO_INCLUDE_PATH stdio.h)
SET(STL_INCLUDE_PATH "${STDIO_INCLUDE_PATH}/c++/4.6")
SET(CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT TRUE)
#include_directories(${STL_INCLUDE_PATH} ${STDIO_INCLUDE_PATH})

SET(CMAKE_BUILD_TYPE DEBUG)
#MESSAGE("${CMAKE_INCLUDE_PATH}")
#MESSAGE("${CMAKE_BUILD_PATH}")
#SET(CMAKE_SRC_PATH "${CMAKE_SOURCE_DIR}/src/")

#SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModule")
#MESSAGE(CMAKE_MODULE_PATH)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
endif (CUDA_FOUND)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -DTHRUST_DEBUG -gencode arch=compute_20,code=sm_20)
CUDA_INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}
	${CUDA_INSTALL_PREFIX}/include
	)
SET(CUDA_LIBRARIES "${CUDA_LIBRARIES};${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse.so")

#-- Add an Option to toggle the generation of the API documentation
SET(BUILD_DOCUMENTATION ON)
if(BUILD_DOCUMENTATION)
  message("Building documentation!")
  FIND_PACKAGE(Doxygen)
  if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR 
      "Doxygen is needed to build the documentation. Please install it correctly")
  endif()
  #-- Configure the Template Doxyfile for our specific project
  configure_file(Doxyfile.in 
                 ${PROJECT_BINARY_DIR}/Doxyfile  @ONLY IMMEDIATE)
  #-- Add a custom target to run Doxygen when ever the project is built
  add_custom_target (Docs ALL 
  										COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
  										SOURCES ${PROJECT_BINARY_DIR}/Doxyfile)
  # IF you do NOT want the documentation to be generated EVERY time you build the project
  # then leave out the 'ALL' keyword from the above command.
endif()

add_subdirectory(${EXT_PROJECTS_DIR}/gtest)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/srcCPU)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/srcGPU)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/unitTest)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/modelTest)

#LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/src  ${CMAKE_SOURCE_DIR}/srcCUDA)

ENABLE_TESTING()


