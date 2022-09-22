cmake_minimum_required(VERSION 3.23)
project(onnxruntime_handler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings -pthread -Wl,-Bsymbolic")

if(MSVC)
    set(ARCHITECTURE "win-x64")
    set(LIB_FORMAT "lib")
    set(DLL_FORMAT "dll")

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/${ARCHITECTURE})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/${ARCHITECTURE})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/${ARCHITECTURE})

    set(CMAKE_CXX_FLAGS "/wd4013 /wd4018 /wd4028 /wd4530 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
else()
    set(ARCHITECTURE "linux-x64")
    set(LIB_PREFIX "lib")
    set(LIB_FORMAT "so")
    set(DLL_FORMAT "so")

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/${ARCHITECTURE}/${CMAKE_BUILD_TYPE})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/${ARCHITECTURE}/${CMAKE_BUILD_TYPE})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/${ARCHITECTURE}/${CMAKE_BUILD_TYPE})
endif()

set(TARGET "orthandler")
set(ONNXRUNTIME_VERSION "1.12.1")
set(ONNXRUNTIME_DIR "${CMAKE_HOME_DIRECTORY}/onnxruntime/onnxruntime-${ARCHITECTURE}-${ONNXRUNTIME_VERSION}")

add_library(${TARGET} SHARED onnxruntime_handler/OnnxRuntimeHandler.cpp)

# Link Onnxruntime
target_include_directories(${TARGET} PRIVATE ${ONNXRUNTIME_DIR}/include/)
target_link_libraries(${TARGET} PRIVATE ${ONNXRUNTIME_DIR}/lib/${LIB_PREFIX}onnxruntime.${LIB_FORMAT})

# Export Library
target_compile_definitions(${TARGET} PRIVATE EXPORT_ORT_HANDLE)


# Copy Onnxruntime binary to output directory
add_custom_command(TARGET ${TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} -E
        copy ${ONNXRUNTIME_DIR}/lib/${LIB_PREFIX}onnxruntime.${DLL_FORMAT}
        ${CMAKE_HOME_DIRECTORY}/${ARCHITECTURE}/${CMAKE_BUILD_TYPE}/${LIB_PREFIX}onnxruntime.${DLL_FORMAT})


#############################
#      For Google Test      #
#############################
add_subdirectory("googletest")

set(TARGET_TEST "test_orthandler")

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES FALSE "tests/*.cpp")

add_executable(${TARGET_TEST} ${TEST_SOURCES})

target_include_directories(${TARGET_TEST} PUBLIC onnxruntime_handler)
target_link_libraries(${TARGET_TEST} PUBLIC "${CMAKE_HOME_DIRECTORY}/${ARCHITECTURE}/${CMAKE_BUILD_TYPE}/${LIB_PREFIX}orthandler.${LIB_FORMAT}")
target_link_libraries(${TARGET_TEST} PUBLIC gtest)

add_test(${TARGET_TEST} COMMAND ${TARGET_TEST})



