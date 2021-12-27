cmake_minimum_required(VERSION 3.16)
project(CNN_Facedetection)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE RELEASE)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

if (CMAKE_BUILD_TYPE STREQUAL DEBUG)
    message("-- CMAKE_BUILD_TYPE = DEBUG")
    add_definitions(-DDEBUG)
else ()
    message("-- CMAKE_BUILD_TYPE = RELEASE")
    add_definitions(-DNDEBUG)
endif ()

option(ENABLE_O3 "use compiler O3 optimization" ON)

if (CMAKE_BUILD_TYPE STREQUAL RELEASE)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
            OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message("-- Turn on compiler O3 optimization.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    elseif ()
        message("-- Turn off compiler O3 optimization")
    endif ()
endif ()

include_directories(include)
aux_source_directory(src DIR_SRCS)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

message("-- CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")