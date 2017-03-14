#pragma once

#include "RLMemory.h"

class ReinforcementLearning {
public:
	int num_input_histories_;		// input to nn
	int num_state_variables_;   	// dimension of the state variables of the game
	int num_game_actions_;			// num outputs of the game

	float gamma_;
	
	network<sequential> nn_;
	RLMemory memory_;

	vec_t old_input_vector_;
public:
	void initialize();
	void initializeConv2D(int height, int width);

	void trainReward();
	void trainRewardMemory();
	void trainRandomReplay(int minibatch);
	void trainReward(const int ix_from_end);
	
	vec_t forward();
	vec_t forward(const vec_t& state_vector);

	void recordHistory(const vec_t& state_vector, const float& reward, const label_t& choice, const vec_t& q_values , const int& terminated);
	void makeInputVectorFromHistory(const int& ix_from_end, vec_t& input);

	label_t getOutputLabelwithEpsilonGreedy(const vec_t& q_values, const float& epsilon);
	const label_t getMaxQLabel(const vec_t& q_values);
	const float_t getMaxQValue(const vec_t& q_values);
};

// end of file
