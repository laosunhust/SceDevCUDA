#MESSAGE(${GPU_SOURCE_DIR})
include_directories(${COMMON_INCLUDES})
CUDA_ADD_LIBRARY(SceGPU
        ${CPU_SOURCE_DIR}/ConfigParser.h
	${CPU_SOURCE_DIR}/ConfigParser.cpp
	SceNodes.h
	SceNodes.cu
)
