cmake_minimum_required(VERSION 3.16)
project(Project_4 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE DEBUG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

option(ENABLE_OMP "use OpenMP" OFF)
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    message("-- x86_64 architecture CPU detected")
    option(ENABLE_AVX2 "use AVX2 IS in x86 CPU" ON)
endif ()
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    message("-- arm architecture CPU detected")
    option(ENABLE_NEON "use NEON IS in arm CPU" OFF)
endif ()
option(ENABLE_O3 "use compiler O3 optimization" ON)

if (ENABLE_O3)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
            OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message("-- Turn on compiler O3 optimization.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif ()
endif ()

if (ENABLE_OMP)
    message("Use OpenMP")
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        message("-- OpenMP Found.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-D_ENABLE_OMP)
    else ()
        message("-- OpenMP Not Found.")
    endif ()
endif ()

if (ENABLE_AVX2)
    message("-- Use AVX2.")
    add_definitions(-D_ENABLE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
endif ()

if (ENABLE_NEON)
    message("-- Use NEON.")
    add_definitions(-D_ENABLE_NEON)
endif ()

find_package(Threads)

include_directories(3rdPartyLib/CImg)
include_directories(include)
aux_source_directory(src DIR_SRCS)

add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME} -lX11)

message("-- CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message("-- O3 Optimization = ${ENABLE_O3}")
message("-- OpenMP = ${ENABLE_OMP}")
message("-- NEON = ${ENABLE_NEON}")
message("-- AVX2 = ${ENABLE_AVX2}")