
#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Scene
{
public:
	GLFWwindow* window = nullptr;
	GLuint VertexArrayID;
	GLuint programID;

	int init();
	void swapBuffers();
	void pollEvents();
	bool getKeyPressed(const int& key);
	bool getWindowShouldClose();
	void clear();
};

// end of file
