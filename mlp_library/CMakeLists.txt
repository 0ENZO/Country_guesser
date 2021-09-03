cmake_minimum_required(VERSION 3.19)
project(mlp_library)

set(CMAKE_CXX_STANDARD 14)

add_library(mlp_library SHARED library.cpp library.h)