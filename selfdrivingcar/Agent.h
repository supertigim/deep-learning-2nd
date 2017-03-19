#pragma once 

#include "ReinforcementLearning.h"

class TestDrivingScene;

class Agent {
protected:
	TestDrivingScene* simul_;
	bool is_training_;
	ReinforcementLearning rl_;

	float reward_sum_;
	float reward_max_;
public:
	Agent(TestDrivingScene* simul);

	void init();

	const int getSelectedDir(const vec_t& q_values);
	void driveCar();
protected:
	bool handleKey();
	void initMemory();
	bool is_good_enough(float& distance_to_travel);
};

// end of file
