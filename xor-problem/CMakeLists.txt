cmake_minimum_required( VERSION 3.0 )
project( SOLVING_XOR_PROBLEM_WITH_CNN )

#ADD_SUBDIRECTORY( "../core" "${CMAKE_CURRENT_BINARY_DIR}/lib_build" )
INCLUDE_DIRECTORIES( ../tiny-dnn )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

MESSAGE( STATUS "Compiling Fully Connected Neural Network to Solve XOR Problem" )
set ( SOURCE_FILES XOR-Problem-Main.cpp )
add_executable( xor_problem ${SOURCE_FILES} )
target_link_libraries( xor_problem )
