cmake_minimum_required (VERSION 2.8.7)
set(PROJECT_NAME_STR beakDevGPU)
project(${PROJECT_NAME_STR} C CXX)

if(CMAKE_COMPILER_IS_GNUCXX)
        add_definitions(-Wall -ansi -Wno-deprecated -pthread)
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
include_directories(${STL_INCLUDE_PATH} ${STDIO_INCLUDE_PATH})

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

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_20,code=sm_20)

add_subdirectory(${EXT_PROJECTS_DIR}/gtest)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/srcCPU)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/srcGPU)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/unitTest)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/modelTest)

#LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/src  ${CMAKE_SOURCE_DIR}/srcCUDA)


ENABLE_TESTING()

#set(PROJECT_TEST_NAME ./build/${PROJECT_NAME_STR}_test)
#include_directories(${GTEST_INCLUDE_DIRS} ${COMMON_INCLUDES})

#file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/unitTest/*.cpp)
#add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
#add_dependencies(${PROJECT_TEST_NAME} googletest)
#add_dependencies(${PROJECT_TEST_NAME} SceCPU)

#if(NOT WIN32)
#        target_link_libraries(${PROJECT_TEST_NAME} pthread)
#        target_link_libraries(${PROJECT_TEST_NAME} ${CPU_SOURCE_DIR}/libSceCPU.a)
#        target_link_libraries(${PROJECT_TEST_NAME}
#                ${GTEST_LIBS_DIR}/libgtest.a
#                ${GTEST_LIBS_DIR}/libgtest_main.a
#        )
#endif()

#add_test(test1 ${PROJECT_TEST_NAME})
