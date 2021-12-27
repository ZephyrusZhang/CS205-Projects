cmake_minimum_required(VERSION 3.16)
project(Project_3 C)

message("Project name: " ${PROJECT_NAME})

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_BUILD_TYPE RELEASE)

include_directories(${PROJECT_NAME}/include)
include_directories(${CUDA_INCLUDE_DIRS})

aux_source_directory(./src DIR_SRCS)

add_compile_options(-mfma)

link_directories(${PROJECT_SOURCE_DIR}/xianyi-OpenBLAS-efe4248/lib)
link_directories(${CUDA_LIBRARIES})

add_executable(Project_3 main.c ${DIR_SRCS})

find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_link_libraries(Project_3 libopenblas.a)
target_link_libraries(Project_3 -lm)
target_link_libraries(Project_3 -lpthread)