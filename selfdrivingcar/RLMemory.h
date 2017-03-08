#pragma once

#include <vector>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

const int MAX_RESERVE = 1000;

class RLMemory {
public:
	int num_elements_;
	int num_reserve_;

	std::vector<vec_t> state_vector_array_;
	std::vector<label_t> selected_array_;
	std::vector<float_t> reward_array_;
	std::vector<vec_t> q_values_array_;
	std::vector<int> terminated_array_;

public:
	RLMemory()
		: num_reserve_(MAX_RESERVE)
		, num_elements_(0)
	{}

	int count() { return num_elements_;}
	
	void reserve(const int& num_reserve)
	{
		state_vector_array_.reserve(num_reserve);
		selected_array_.reserve(num_reserve);
		reward_array_.reserve(num_reserve);
		q_values_array_.reserve(num_reserve);

		num_reserve_ = num_reserve;
	}

	void reset()
	{
		num_elements_ = 0;

		state_vector_array_.clear();
		selected_array_.clear();
		reward_array_.clear();
		q_values_array_.clear();

		reserve(num_reserve_);
	}

	void append(const vec_t& state_vector, const label_t& choice, const float& reward, const vec_t& q_values, const int& terminated)
	{
		assert(num_elements_ < num_reserve_);

		state_vector_array_.push_back(state_vector);
		selected_array_.push_back(choice);
		reward_array_.push_back(reward);
		q_values_array_.push_back(q_values);
		terminated_array_.push_back(terminated);

		++num_elements_;
	}

	// ix_from_last = 0 returns last element, use -1, -2 ,...
	const vec_t& getStateVectorFromLast(const int& ix_from_last) {
		return state_vector_array_[num_elements_ - 1 + ix_from_last];
	}

	const label_t& getSelectedIxFromLast(const int& ix_from_last) {
		return selected_array_[num_elements_ - 1 + ix_from_last];
	}

	const float& getRewardFromLast(const int& ix_from_last) {
		return reward_array_[num_elements_ - 1 + ix_from_last];
	}

	// ix_from_last = 0 returns last element, use -1, -2 ,...
	const vec_t& getQValuesFromLast(const int& ix_from_last) {
		return q_values_array_[num_elements_ - 1 + ix_from_last];
	}

	// ix_from_last = 0 returns last element, use -1, -2 ,...
	const int& getTermiatedFromLast(const int& ix_from_last) {
		return terminated_array_[num_elements_ - 1 + ix_from_last];
	}
};

// end of file
