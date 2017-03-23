#include "TestDrivingScene.h"
#include <glm/gtc/matrix_transform.hpp>
#include "Agent.h"

//#include <iostream>

void TestDrivingScene::init() {
	
	Scene::init();

	car_.init();

	const float	world_center_x = 0.5f,
				world_center_y = 0.5,
				world_radius = 1.2f,
				obstacle_radius = 0.05f,
				margin = 0.00f;


	{
		Object *temp = new Object;
		temp->initCircle(glm::vec3(world_center_x, world_center_y, 0.0f), world_radius, 30);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp)));
	}

	/*
	for(int i = 0 ; i < 40; ++i){
		float x,y;
		//set_random_seed(i);

		while (true){
			//std::cout << "minx: " << world_center_x - (world_radius - obstacle_radius)
			//		<< "maxx: " << world_center_x + (world_radius - obstacle_radius) << endl;
			x = uniform_rand(world_center_x - (world_radius - obstacle_radius - margin)
								, world_center_x + (world_radius - obstacle_radius - margin));
			y = uniform_rand(world_center_y - (world_radius - obstacle_radius - margin)
								, world_center_y + (world_radius - obstacle_radius - margin));

			//std::cout << "x: " << x << " y: " << y << endl;
			if( world_radius > std::sqrt((x-world_center_x)*(x-world_center_x)
										 + (y-world_center_y)*(y-world_center_y)))
				break;
		}

		Object *temp = new Object;
		temp->initCircle(glm::vec3(x, y, 0.0f), obstacle_radius, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 

	}
	//*/
//*
	{
		Object *temp = new Object;
		temp->initCircle(glm::vec3(0.5f, 0.5f, 0.0f), 0.75f, 20);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp)));
	}

	// hurdles for training 
	{
		Object *temp = new Object;
		//temp->initCircle(glm::vec3(0.98f, 0.0f, 0.0f), 0.05f, 6);
		temp->initCircle(glm::vec3(-0.43, 0.0f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}
	
	{
		Object *temp = new Object;
		temp->initCircle(glm::vec3(1.65f, 0.5f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}

	{
		Object *temp = new Object;
		//temp->initCircle(glm::vec3(0.98f, 1.0f, 0.0f), 0.05f, 6);
		temp->initCircle(glm::vec3(1.0f, 1.1f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}
	
	{
		Object *temp = new Object;
		//temp->initCircle(glm::vec3(0.5f, 1.45f, 0.0f), 0.05f, 6);
		temp->initCircle(glm::vec3(0.2f, 1.3f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}

	{
		Object *temp = new Object;
		//temp->initCircle(glm::vec3(0.5f, 1.45f, 0.0f), 0.05f, 6);
		temp->initCircle(glm::vec3(0.50f, -0.35f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}
//*/
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
	case 1:
		car_.turnLeft(); // left
		break;
	case 2:
		car_.turnRight(); // right
		break;
	case 0:
		// stay straight
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
/*
	for (int i = 0 ; i < rayNums; ++i){
		if(car_.distances_from_sensors_[i] < collideWarningThreshold)
			discount += 0.02f;
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
*/
	for( int i = 0 ; i < rayNums ; ++i) {
		aheadReward += car_.distances_from_sensors_[i];
	}
	aheadReward /= (rayNums * 10.0f);

	//std::cout << "aheadReward: " << aheadReward << endl; 

	// velocity reward
	const float speed = glm::dot(car_.vel_, car_.dir_);
	const float max_speed = 0.01f;
	//reward = speed / max_speed;
	//reward = 1.0f;
	//reward = 0.5f;
	//reward = reward/2 + aheadReward/2 ;// - discount; // constant reward for this example
	reward = aheadReward;

	//std::cout << "reward" << reward << endl;

	// normalized into 0.1 or -0.1 
	//reward == 0 ?
    //          0 :
    //          reward /= std::abs(reward)*10.0f;


	// collision check
	glm::vec3 col_line_center;
	if (car_.body_.checkCollisionLoop(obj_list, col_line_center) == true) {
		//static int count = 0;
		//std::cout << "Collision " << count++ << endl;
		
		// reset car status
		car_.init();
		//car_.body_.model_matrix_ = glm::mat4();

		reward = -1.0f;	// no reward
		//reward = -0.20f;		// no reward
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
		if (getKeyPressed(GLFW_KEY_LEFT) == true) processInput(1);
		if (getKeyPressed(GLFW_KEY_RIGHT) == true) processInput(2);

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
