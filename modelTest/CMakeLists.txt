ENABLE_TESTING()

set(PROJECT_TEST_NAME ${PROJECT_NAME_STR}_model_test)
MESSAGE(${PROJECT_TEST_NAME})
include_directories(${GTEST_INCLUDE_DIRS} ${COMMON_INCLUDES} ${COMMON_CUDA_INCLUDES})

file(GLOB TEST_SRC_FILES ${CMAKE_MODELTEST_DIR}/*.cpp)
MESSAGE(${TEST_SRC_FILES})
cuda_add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
add_dependencies(${PROJECT_TEST_NAME} googletest)
add_dependencies(${PROJECT_TEST_NAME} SceGPU)

if(NOT WIN32)
        target_link_libraries(${PROJECT_TEST_NAME} pthread)
        target_link_libraries(${PROJECT_TEST_NAME} ${GPU_SOURCE_DIR}/libSceGPU.a)
        target_link_libraries(${PROJECT_TEST_NAME}
                ${GTEST_LIBS_DIR}/libgtest.a
                ${GTEST_LIBS_DIR}/libgtest_main.a
        )
endif()

add_test(modelTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_TEST_NAME})

