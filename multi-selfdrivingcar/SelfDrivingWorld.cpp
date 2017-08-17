#include "SelfDrivingWorld.h"
#include <glm/gtc/matrix_transform.hpp>
#include "models/selfdrivingcarnet.h"

const int CAR_PLAYED_NUM = 7;	// number of cars driving on the world

SelfDrivingWorld::SelfDrivingWorld()
	:is_training_(0)
{}

void SelfDrivingWorld::initialize() {
	
	Scene::init();

	// Create number of AI cars 
	createAICars(CAR_PLAYED_NUM);

	// Make neural network
	std::shared_ptr<network<sequential>> nn = std::make_shared<network<sequential>>();
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
	
	TDNN_Models::self_driving_car_alt2_net(*nn,
								backend_type,
								AICar::NETWORK_INPUT_NUM * AICar::INPUT_FRAME_CNT,
								AICar::ACT_MAX);

	dqn_ = std::make_unique<DDQN>();
	dqn_->initialize(nn);

	// Create World
	//createScene_RandomObstacles();
	//createScene_Basic();
	createScene_Road();

	// Camera matrix
	View_ = glm::lookAt(
		glm::vec3(0.5, 0.5, 3), // Camera is at (4,3,3), in World Space
		glm::vec3(0.5, 0.5, 0), // and looks at the origin
		glm::vec3(0, 1, 0)  	// Head is up (set to 0,-1,0 to look upside-down)
	);

	MatrixID_ = glGetUniformLocation(programID, "MVP");
	Projection_ = glm::perspective(glm::radians(45.0f), 1024.0f / 768.0f, 0.1f, 100.0f);
	//glm::mat4 Projection = glm::ortho(-1.0f,1.0f,-1.0f,1.0f,0.0f,100.0f); // In world coordinates
}

void SelfDrivingWorld::createAICars(int nums){

	for(int i = 0; i < nums; ++i){
		AICar *car = new AICar(1 << i);		
		car->initialize();
		car_list_.push_back(std::move(std::unique_ptr<AICar>(car)));
	}
}

void SelfDrivingWorld::createScene_Road(){
	const float	world_center_x = 0.5f,
			world_center_y = 0.5,
			world_radius = 1.2f;
	
	// outer barrier
	{
		SquareObj* box = new SquareObj();
		box->update(glm::vec3(world_center_x, world_center_y, 0.0f), world_radius, world_radius);
		obj_list.push_back(std::move(std::unique_ptr<Object>(box)));
	}

	{
		SquareObj* box = new SquareObj();
		box->update(glm::vec3(world_center_x - 0.47f, world_center_y + 0.50f, 0.0f), 0.3f, 0.3f);
		obj_list.push_back(std::move(std::unique_ptr<Object>(box)));
	}

	{
		SquareObj* box = new SquareObj();
		box->update(glm::vec3(world_center_x - 0.47f, world_center_y - 0.50f, 0.0f), 0.3f, 0.3f);
		obj_list.push_back(std::move(std::unique_ptr<Object>(box)));
	}

	{
		SquareObj* box = new SquareObj();
		box->update(glm::vec3(world_center_x + 0.47f, world_center_y + 0.50f, 0.0f), 0.3f, 0.3f);
		obj_list.push_back(std::move(std::unique_ptr<Object>(box)));
	}

	{
		SquareObj* box = new SquareObj();
		box->update(glm::vec3(world_center_x + 0.47f, world_center_y - 0.50f, 0.0f), 0.3f, 0.3f);
		obj_list.push_back(std::move(std::unique_ptr<Object>(box)));
	}
}

void SelfDrivingWorld::createScene_Basic(){
	const float	world_center_x = 0.5f,
			world_center_y = 0.5,
			world_radius = 1.2f;

	// outer barrier		
	{
		Object *temp = new Object;
		temp->initCircle(glm::vec3(world_center_x, world_center_y, 0.0f), world_radius, 30);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp)));
	}

	// inner barrier
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
		temp->initCircle(glm::vec3(0.2f, 1.4f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}

	{
		Object *temp = new Object;
		//temp->initCircle(glm::vec3(0.5f, 1.45f, 0.0f), 0.05f, 6);
		temp->initCircle(glm::vec3(0.990f, -0.35f, 0.0f), 0.05f, 6);
		obj_list.push_back(std::move(std::unique_ptr<Object>(temp))); 
	}
}

void SelfDrivingWorld::createScene_RandomObstacles(){
	const float	world_center_x = 0.5f,
			world_center_y = 0.5,
			world_radius = 1.2f,
			obstacle_radius = 0.05f,
			margin = 0.05f;
	
	// outer barrier
	{
		//Object *temp = new Object;
		//temp->initCircle(glm::vec3(world_center_x, world_center_y, 0.0f), world_radius, 30);
		//obj_list.push_back(std::move(std::unique_ptr<Object>(temp)));
		SquareObj* box = new SquareObj();
		box->update(glm::vec3(world_center_x, world_center_y, 0.0f), world_radius, world_radius);
		obj_list.push_back(std::move(std::unique_ptr<Object>(box)));

	}

	for(int i = 0 ; i < 20; ++i){
		float x,y;
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
}

void SelfDrivingWorld::render(){

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Use our shader
	glUseProgram(programID);
	glEnableVertexAttribArray(0);

	for(int i = 0 ; i < car_list_.size() ; ++i){
		car_list_[i]->render(MatrixID_, Projection_ * View_);
	}

	//for (auto itr : obj_list) // this doesn't work with unique ptr
	for (int i = 0; i < obj_list.size(); i++){
		obj_list[i]->drawLineLoop(MatrixID_, Projection_ * View_);
	}

	glDisableVertexAttribArray(0);
	swapBuffers();
}

bool SelfDrivingWorld::handleKeyInput(){
	glfwPollEvents();

	//if (getKeyPressed(GLFW_KEY_LEFT) == true) car_list_[0]->processInput(AICar::ACT_LEFT);
	//else if (getKeyPressed(GLFW_KEY_RIGHT) == true) car_list_[0]->processInput(AICar::ACT_RIGHT);

	//if (getKeyPressed(GLFW_KEY_UP) == true) car_list_[0]->processInput(AICar::ACT_ACCEL);
	//else if (getKeyPressed(GLFW_KEY_DOWN) == true) car_list_[0]->processInput(AICar::ACT_BRAKE);

	//if (getKeyPressed(GLFW_KEY_A) == true) car_list_[1]->processInput(AICar::ACT_ACCEL);
	//else if (getKeyPressed(GLFW_KEY_Z) == true) car_list_[1]->processInput(AICar::ACT_BRAKE);

	// Check if the ESC key was pressed or the window was closed
	if (getKeyPressed(GLFW_KEY_ESCAPE) || getWindowShouldClose())
		return false;

	if (getKeyPressed(GLFW_KEY_Q) == true) {
		//nn_.save("SELF-DRIVING-CAR-MODEL");
		std::cout << "writing complete" << endl;
	}

	// training mode change key input
	static bool key_reset_flag = true;
	if (getKeyPressed(GLFW_KEY_SPACE) == true || 
		getKeyPressed(GLFW_KEY_A) == true) {

		if(key_reset_flag == true) {
			if(!is_training_){
				if(getKeyPressed(GLFW_KEY_A) == true)	is_training_ = INT_MAX;
				else									is_training_ = 1 << uniform_rand(0, (int)car_list_.size()-1);
				std::cout << "Training mode " << endl;
			}
			else {
				is_training_ = 0;
				std::cout << "Normal mode" << endl;
			} 

			key_reset_flag = false;

			if (is_training_) {
				
			}
			else {
				
			}
		}
	}
	else {
		key_reset_flag = true;
	}

	return true;
}


void SelfDrivingWorld::run() {

	while(true)
	{
		if(!handleKeyInput()) break;

		for(int i = 0 ; i < car_list_.size() ; ++i){
			car_list_[i]->drive();
		}
		//if(!is_training_)
			render();
	}
	clear();
}

// end of file
