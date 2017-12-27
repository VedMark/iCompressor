cmake_minimum_required(VERSION 3.5)
project(iCompressor C)

set(CMAKE_C_STANDARD 11)

link_directories(/src/libsoil/)

set (CMAKE_C_FLAGS "-Wall")
add_executable(iCompressor src/main.c src/compressor.c src/include/compressor.h)
target_link_libraries(iCompressor -lSOIL -lglut -lGL -lGLU -lm )