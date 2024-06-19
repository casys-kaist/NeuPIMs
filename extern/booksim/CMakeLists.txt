cmake_minimum_required(VERSION 3.16)
project(booksim2_project)

file(GLOB_RECURSE BOOKSIM_SRCS CONFIGURE_DEPENDS src/*.cpp)
add_library(booksim2 STATIC ${BOOKSIM_SRCS})
target_include_directories(booksim2 
  PUBLIC include
  PRIVATE include/booksim2
  PRIVATE src
  PRIVATE src/allocators
  PRIVATE src/arbiters
  PRIVATE src/power
  PRIVATE src/routers
  PRIVATE src/networks
)
target_compile_options(booksim2 PRIVATE -Wall -O3)
