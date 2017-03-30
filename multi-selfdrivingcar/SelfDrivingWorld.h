#pragma once 

#include "SquareObj.h"
#include "LineObj.h"
#include "AICar.h"
#include "Scene.h"
#include "DQN.h"

class Agent;

class SelfDrivingWorld : public Scene
{
public:
	std::vector<std::unique_ptr<AICar>> car_list_;
	std::vector<std::unique_ptr<Object>> obj_list;
	
protected:
	glm::mat4 View_;
	GLuint MatrixID_;
	glm::mat4 Projection_;

	int input_frame_cnt_;
	int is_training_;

	std::unique_ptr<DQN> dqn_;
public:
	DQN& dqn(){return *dqn_;}
	int is_training() {return is_training_;}

	void initialize();
	void createAICars(int nums);

	// flag = 0 : continue, 1 : terminal
	void update(const bool& update_render_data, float& reward, int& flag);

	const std::vector<std::unique_ptr<Object>>& getObjects() { return obj_list;}
	const std::vector<std::unique_ptr<AICar>>& getCars() { return car_list_; }

	void render();
	void run();

	static SelfDrivingWorld& get(){
		static SelfDrivingWorld main_scene;
		return main_scene;
	}

	bool handleKeyInput();

protected:
	SelfDrivingWorld();
	void createScene_RandomObstacles();
	void createScene_Basic();
	void createScene_Road();
};

// end of file
