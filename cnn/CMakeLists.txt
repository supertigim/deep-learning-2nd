cmake_minimum_required( VERSION 3.0 )
project( CONVOLUTIONAL_NEURAL_NETWORK )

#ADD_SUBDIRECTORY( "../core" "${CMAKE_CURRENT_BINARY_DIR}/lib_build" )
INCLUDE_DIRECTORIES( ../tiny-dnn )
INCLUDE_DIRECTORIES( ../share )

add_definitions(-DUSE_AVX)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

MESSAGE( STATUS "Compiling Convolitional Neural Network" )
set ( SOURCE_FILES ConvNN-main.cpp )
add_executable( cnn ${SOURCE_FILES} )
target_link_libraries( cnn )

