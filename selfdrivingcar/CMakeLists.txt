cmake_minimum_required( VERSION 3.0 )
project( SELF_DRIVING_CAR )

FIND_PACKAGE( Boost COMPONENTS system REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(glfw3 REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)

INCLUDE_DIRECTORIES( /usr/local/Cellar/glm/0.9.8.3/include/ )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} ../core)
INCLUDE_DIRECTORIES( ../tiny-dnn )

set (EXTRA_BOOST_LIBS ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})
set (EXTRA_LIBS ${GLUT_LIBRARY} ${OpenGL_LIBRARIES} ${GLEW_LIBRARY} ${OPENGL_glu_LIBRARY} glfw pthread )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

MESSAGE( STATUS "Compiling Self Driving Car" )

set ( SOURCE_FILES SelfDrivingCarMain.cpp shader.cpp Object.cpp SquareObj.cpp Car.cpp TestDrivingScene.cpp Scene.cpp Agent.cpp ReinforcementLearning.cpp)

add_executable( self_driving_car ${SOURCE_FILES} )
target_link_libraries( self_driving_car ${EXTRA_BOOST_LIBS} ${EXTRA_LIBS})	
