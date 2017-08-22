cmake_minimum_required( VERSION 3.0 )
project( MULTI_SELF_DRIVING_CAR )

FIND_PACKAGE(OpenGL REQUIRED)
#FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(glfw3 3.2 REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)

ADD_SUBDIRECTORY( "../share" "${CMAKE_CURRENT_BINARY_DIR}/lib_build" )
INCLUDE_DIRECTORIES( /usr/local/Cellar/glm/0.9.8.3/include/ ../share ../tiny-dnn )

set (EXTRA_LIBS ${GLUT_LIBRARIES} ${OpenGL_LIBRARIES} ${GLEW_LIBRARY} ${OPENGL_glu_LIBRARY} glfw pthread )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

MESSAGE( STATUS "Compiling Multi Self Driving Car" )

set ( SOURCE_FILES shader.cpp Object.cpp SquareObj.cpp AICar.cpp Scene.cpp SelfDrivingWorld.cpp SelfDrivingCarMain.cpp AutoCarNet.cpp)

add_executable( multi_self_driving_car ${SOURCE_FILES} )
target_link_libraries( multi_self_driving_car ${EXTRA_BOOST_LIBS} ${EXTRA_LIBS} DLShare)	
