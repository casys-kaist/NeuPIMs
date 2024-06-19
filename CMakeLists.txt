cmake_minimum_required(VERSION 3.15.0)
set(project_name "AiFrameworkSim")
project(${project_name})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("${CMAKE_SOURCE_DIR}/build/conanbuildinfo.cmake")
conan_basic_setup()

# find_package(Boost 1.70 REQUIRED COMPONENTS program_options)
# message("LIB ${Boost_LIBRARY_DIRS}")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/extern)

# C++ setttings
set(CMAKE_CXX_STANDARD 17)
set(ONNX_ML 1)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/lib")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message("BINARY DIR ${CMAKE_BINARY_DIR}")

# For debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-g -O0 -Wall -Wextra --debug-cpp --trace -rdynamic")

# Build source
add_subdirectory("${PROJECT_SOURCE_DIR}/src")

# Add library NewtonSim
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/NewtonSim")

# Add libaray booksim
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/booksim")

target_include_directories(Simulator PUBLIC)
target_link_libraries(Simulator dramsim3 booksim2)
target_link_libraries(Simulator ${CONAN_LIBS} stdc++fs)

target_include_directories(Simulator_lib PUBLIC)
target_link_libraries(Simulator_lib dramsim3 booksim2)
target_link_libraries(Simulator_lib ${CONAN_LIBS} stdc++fs)

enable_testing()
# add_subdirectory("${PROJECT_SOURCE_DIR}/tests")

