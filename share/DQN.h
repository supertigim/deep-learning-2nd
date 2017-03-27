#pragma once 

#include "Replay.h"

const float DEFAULT_EPSILON = 0.20f;
const float DEFAULT_LEARNING_RATE = 0.95f;

/**
 *   Deep Q-Network
 */
class DQN: public learnig_algorithm {
public:
	explicit DQN(network<sequential>& nn, 
				const int& input_size, 
				const int& input_frame_count, 
				const int& output_nums);
	~DQN(){}
	void setLearnigRate(float gamma){gamma_ = gamma;}
	void update(int batch_size);
	label_t selectAction(const vec_t& state, float epsilon = DEFAULT_EPSILON);
	vec_t forward(const vec_t& state_vector);
	void addTransition(const Transition& transition){
		replay_.push_back(transition);
	}

	void printQValues(const vec_t& state_vector);
	int replay_memory_size(){ return replay_.size();}

protected:
	network<sequential>& nn_;
	Replay replay_;
	float gamma_;
	int output_nums_;
};

// end of file
