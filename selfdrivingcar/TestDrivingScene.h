#pragma once 

#include "SquareObj.h"
#include "LineObj.h"
#include "Car.h"
#include "Scene.h"

class Agent;

class TestDrivingScene : public Scene
{
public:
	Car car_;
	std::vector<std::unique_ptr<Object>> obj_list;
	int prev_action_;

	bool compat_state_ = true; // false: image state for conv net
	vec_t state_buffer_;

	glm::mat4 View_;
	GLuint MatrixID_;
	glm::mat4 Projection_;

public:
	TestDrivingScene(){}

	void init();
	void processInput(const int& action) ;

	// flag = 0 : continue, 1 : terminal
	void update(const bool& update_render_data, float& reward, int& flag);
	int getNumStateVariables();
	int getNumActions();
	// update state buffer and return it
	const vec_t& getStateBuffer();

	void run();
	void render();
};

// end of file
