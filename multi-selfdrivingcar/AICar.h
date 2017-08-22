#pragma once 

#include "SquareObj.h"
#include "LineObj.h"
#include "DQN.h"

class AICar : public SquareObj{
public:
	enum {
		ACT_ACCEL = 0,
		ACT_BRAKE = 1,
		ACT_LEFT = 2,
		ACT_RIGHT = 3,
		//ACT_STAY = 4,
		ACT_MAX = 4
	};

	static float NETWORK_INPUT_NUM;
	static float INPUT_FRAME_CNT;

protected:
	const int ID_;
	LineObj sensing_lines;
	
	glm::vec3 dir_, vel_;
	glm::vec3 previous_pos_;

	float turn_coeff_;
	float accel_coeff_;
	float brake_coeff_;
	float fric;
	float sensing_radius;

	int sensor_min, sensor_max, sensor_di;
	std::vector<float> distances_from_sensors_;

	float car_length_;
	float direction_degree_;

	// for DQN 
	int batch_size_;
	std::deque<vec_t> past_states_;
	PEReplay replay_;

public:
	AICar(int id);
	void initialize();
	int ID() {return ID_;}

	void setDirection(float dir);
	float getDirection() { return direction_degree_;}
	float getSpeed();

	void drive();
	void render(const GLint& MatrixID, const glm::mat4 vp);

	void processInput(const int& action);

protected:
	void turnLeft();
	void turnRight();
	void accel();
	void decel();
	
	void updateAll(label_t& action, float& reward, vec_t& t);
	void updateSensor();
	void updateStateVector();
	
	bool isTerminated();
	void calculateReward(float& reward);
	bool getState(vec_t& t);
};

// end of file
