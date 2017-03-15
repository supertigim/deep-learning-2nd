#include "TestDrivingScene.h"
#include <glm/gtc/matrix_transform.hpp>
#include "Agent.h"

//#include <iostream>

void TestDrivingScene::init() {
	
	Scene::init();

	car_.init();

	{
		Object *temp = new Object;
		temp->initCircle(glm::vec3(0.5f, 0.5f, 0.0f), 1.0f, 30);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp)));
	}

	{
		Object *temp = new Object;
		temp->initCircle(glm::vec3(0.5f, 0.5f, 0.0f), 0.65f, 30);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp)));
	}

	// hurdles for training 
	{
		Object *temp = new Object;
		//temp->initCircle(glm::vec3(0.98f, 0.0f, 0.0f), 0.05f, 6);
		temp->initCircle(glm::vec3(1.01f, 0.0f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}
	
	{
		Object *temp = new Object;
		temp->initCircle(glm::vec3(1.45f, 0.5f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}

	{
		Object *temp = new Object;
		//temp->initCircle(glm::vec3(0.98f, 1.0f, 0.0f), 0.05f, 6);
		temp->initCircle(glm::vec3(1.0f, 1.0f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}
	
	{
		Object *temp = new Object;
		//temp->initCircle(glm::vec3(0.5f, 1.45f, 0.0f), 0.05f, 6);
		temp->initCircle(glm::vec3(0.5f, 1.25f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}

	// Camera matrix
	View_ = glm::lookAt(
		glm::vec3(0.5, 0.5, 3), // Camera is at (4,3,3), in World Space
		glm::vec3(0.5, 0.5, 0), // and looks at the origin
		glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
	);

	MatrixID_ = glGetUniformLocation(programID, "MVP");
	Projection_ = glm::perspective(glm::radians(45.0f), 1024.0f / 768.0f, 0.1f, 100.0f);
	//glm::mat4 Projection = glm::ortho(-1.0f,1.0f,-1.0f,1.0f,0.0f,100.0f); // In world coordinates


	//std::cout  << getNumStateVariables() << std::end;
	
	state_buffer_.resize(getNumStateVariables(), 0);
}

void TestDrivingScene::processInput(const int& action) {
	prev_action_ = action;

	switch (action)
	{
	case 0:
		car_.turnLeft(); // right
		break;
	case 1:
		car_.turnRight(); // left
		break;
	case 2:
		// stay
		break;
	default:
		std::cout << "Wrong action " << endl;
		exit(1);
		break;
	}
	// always accel in this example
	car_.accel();
}

void TestDrivingScene::update(const bool& update_render_data, float& reward, int& flag) {
	flag = 0;
	reward = 0.0f;
	
	car_.update();
	car_.updateSensor(obj_list, update_render_data);

	int rayNums = car_.distances_from_sensors_.size();
	float aheadReward = 0.0f; 
	float normalized = 5.0f;
	float collideWarningThreshold = 0.1f;
	float discount = 0.0f;

	for (int i = 0 ; i < rayNums; ++i){
		if(car_.distances_from_sensors_[i] < collideWarningThreshold)
			discount += 0.005f;
			//std::cout << "WARNING" << endl;
	}
	reward /= (rayNums * normalized);

	if( rayNums % 2){
		assert(rayNums >=3);
		aheadReward = (car_.distances_from_sensors_[rayNums/2-2] +
						car_.distances_from_sensors_[rayNums/2-1] +
						car_.distances_from_sensors_[rayNums/2] +
						car_.distances_from_sensors_[rayNums/2+1]+
						car_.distances_from_sensors_[rayNums/2+2])/(5.0f*normalized);
	} else
	{
		aheadReward = (car_.distances_from_sensors_[rayNums/2-2] +
						car_.distances_from_sensors_[rayNums/2-1] +
						car_.distances_from_sensors_[rayNums/2]+
						car_.distances_from_sensors_[rayNums/2+2])/(4.0f*normalized);
	}

	//std::cout << "aheadReward" << aheadReward << endl;
	// velocity reward
	const float speed = glm::dot(car_.vel_, car_.dir_);
	const float max_speed = 0.01f;
	//reward = speed / max_speed;
	//reward = 0.1f;
	reward = 0.05f + aheadReward - discount; // constant reward for this example

	//std::cout << "reward" << reward << endl;

	// collision check
	glm::vec3 col_line_center;
	if (car_.body_.checkCollisionLoop(obj_list, col_line_center) == true) {
		//static int count = 0;
		//std::cout << "Collision " << count++ << endl;
		
		// reset car status
		car_.init();
		car_.body_.model_matrix_ = glm::mat4();

		//reward = 0.0f;	// no reward
		reward = -0.5f;	// no reward
		flag = 1;		// terminal //TODO: use enum 
	}
}

int TestDrivingScene::getNumStateVariables() {
	
	return car_.distances_from_sensors_.size(); // sensor inputs
}

int TestDrivingScene::getNumActions() {
	// left, right, stay
	return 3; 
}

const vec_t& TestDrivingScene::getStateBuffer() {
		

	for (int i = 0; i < car_.distances_from_sensors_.size(); i++) {
		//Note: most of distances are larger than 1. Don't clamp.
		state_buffer_[i] = car_.distances_from_sensors_[i]; 

		//std::cout << state_buffer_[i];
	}	

	return state_buffer_;
}

void TestDrivingScene::run(){
	while(true)
	{
		//glfwPollEvents();

		// Check if the ESC key was pressed or the window was closed
		if (getKeyPressed(GLFW_KEY_ESCAPE) || getWindowShouldClose())
			break;

		// animate update
		if (getKeyPressed(GLFW_KEY_LEFT) == true) processInput(0);
		if (getKeyPressed(GLFW_KEY_RIGHT) == true) processInput(1);

		processInput(2); // always accelerate in this example

		// dummy reward
		float reward;
		int flag;

		update(true, reward, flag);

		render();
	}

	clear();
		
}

void TestDrivingScene::render(){

	// Clear the screen
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT);

	// Use our shader
	glUseProgram(programID);
	glEnableVertexAttribArray(0);

	//glm::mat4 Model = glm::mat4(1.0f);
	//glm::mat4 MVP = Projection_ * View_ * Model;

	//glUniformMatrix4fv(MatrixID_, 1, GL_FALSE, &MVP[0][0]);

	// draw
	car_.body_.drawLineLoop(MatrixID_, Projection_ * View_);
	car_.sensing_lines.drawLineLoop(MatrixID_, Projection_ * View_);

	//for (auto itr : obj_list) // this doesn't work with unique ptr
	for (int i = 0; i < obj_list.size(); i++)
	{
		obj_list[i]->drawLineLoop(MatrixID_, Projection_ * View_);
	}

	glDisableVertexAttribArray(0);

	swapBuffers();
}


// end of file
