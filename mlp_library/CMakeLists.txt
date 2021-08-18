cmake_minimum_required(VERSION 3.19)
project(mlp_library)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD 20)

include_directories("${PROJECT_SOURCE_DIR}\\Eigen")

add_executable(mlp_library mlp.cpp mlp.h)

#target_link_libraries(mlp_library ${PROJECT_SOURCE_DIR}\\Eigen)