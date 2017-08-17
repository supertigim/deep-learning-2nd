/*
    Copied and Pasted by Jay (JongYoon) Kim, jyoon95@gmail.com 
*/
#pragma once 

#include "Replay.h"

const float DEFAULT_EPSILON = 0.20f;
const float MAX_EPSILON = 0.40f;
const float MIN_EPSILON = 0.10f;
const float EPSILON_DECAY_RATE = 0.9999f;
const float DEFAULT_LEARNING_RATE = 0.95f;

/**
 *   Deep Q-Network
 */
class DQN {
public:
	DQN();
	virtual ~DQN(){}
	virtual void initialize(std::shared_ptr<network<sequential>>& nn_ptr);
	void setLearnigRate(float gamma){gamma_ = gamma;}
	virtual void update(Replay& replay, int batch_size);
	label_t selectAction(const vec_t& state, bool is_greedy = false);
	vec_t forward(const vec_t& state_vector);
	void printQValues(const vec_t& state_vector);

protected:
	//std::shared_ptr<network<sequential>> nn_ptr_;
	std::shared_ptr<tiny_dnn::network<tiny_dnn::sequential>> nn_ptr_;
	float gamma_;
	float epsilon_;
};


/**
 *   Double Deep Q-Network
 */
class DDQN: public DQN {
public:
	DDQN();
	~DDQN(){}
	void update(Replay& replay, int batch_size);
	void initialize(std::shared_ptr<network<sequential>>& nn_ptr);
protected:
	std::shared_ptr<network<sequential>> target_nn_ptr_;
};

// end of file
