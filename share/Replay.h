#pragma once 

#include "Common.h"

const int DEF_MAX_REPLAY_CNT	= 5000;
const int DEF_INPUT_FRAME_CNT	= 1;

using Transition = std::tuple<vec_t,	// input data
							label_t,	// action
							float,		// reward
							int>;		// is terminate ? (1 if terminated)

/**
 *   Replay class 
 * 
 *   to store past transitions 
 */
class Replay {
public:
	Replay()
	:max_memory_size_(DEF_MAX_REPLAY_CNT)
	,input_frame_count_(DEF_INPUT_FRAME_CNT)
	,input_size_(0)
	//,high_td_(INT_MIN)
	//,low_td_(INT_MAX)
	{}
	~Replay() {}

	//static Transition make_Transition(int inputSize){
	//	auto ptr = std::make_unique<Transition>(inputSize,
	//											0,
	//											0.0f,
	//											0);
	//	return std::move(*ptr.get());
	//}

	void init(const int& input_size, const int& input_frame_count){
		input_size_ = input_size;
		input_frame_count_ = input_frame_count;
	}

	void clear() {
		memory_.clear();
		//while(memory_.size())
		//	memory_.pop_front();
	}

	int size() { return memory_.size(); }

	void push_back(const Transition& t){
		assert(input_size_ != 0);
		
		//float td = std::get<2>(t);
		//if( low_td_ > td) low_td_ = td;
		//else if( high_td_ < td) high_td_ = td;

		//memory_.push_back(std::move(t));
		memory_.push_back(t);

		if(max_memory_size_ < memory_.size()){
			memory_.pop_front();
		}
	}

	Transition& operator [] (const int index){
		assert(index >= 0 && index < memory_.size());
		return memory_[index];
	}

	bool getSampleIdxVector(std::vector<int>& v, const int size) {

		if(size < 0 || memory_.size() == 0 || size > memory_.size()){
			return false;
		}

		v.reserve(size);
/*
		int count = 0;
		int found = 0;
		float mid_value = (high_td_+low_td_)/2;
		while (count < memory_.size() &&  found < size ) {
			const int idx = uniform_rand((int)(0 + (input_frame_count_ - 1)) , (int)(memory_.size() - 1));
			
			float td = std::get<2>(memory_[idx]);
			if( td < mid_value ){
				v.push_back(idx);
				++found;
			}
			++count;
		}

		for (int i = found; i < size; ++i) {
			const int idx = uniform_rand((int)(0 + (input_frame_count_ - 1)) , (int)(memory_.size() - 1));
			v.push_back(idx);
		}
*/
		for (auto i = 0; i < size; ++i) {
			const int idx = uniform_rand((int)(0 + (input_frame_count_ - 1)) , (int)(memory_.size() - 1));
			v.push_back(idx);
		}
				
		return true;
	}

	bool getState(int idx, vec_t& s1, vec_t& s2){
		assert(idx >= 0);
		if( memory_.size() == 0)				return false;
		if(idx - (input_frame_count_ - 1) < 0) 	return false;	// for checking s1
		if(idx + 1 > memory_.size() - 1) 		return false;	// for checking s2 or s`

		const int input_to_net_size = input_size_ * input_frame_count_;

		s1.resize(input_to_net_size);
		for(int i = 0, count = 0; i < input_frame_count_; ++i, count += input_size_){
			const vec_t& forS1 = std::get<0>(memory_[idx - i]);
			std::copy(forS1.begin(), forS1.end(), s1.begin() + count);
		}

		// not terminated
		if(!std::get<3>(memory_[idx])){
			s2.resize(input_to_net_size);

			for(int i = 0, count = 0; i < input_frame_count_; ++i, count += input_size_){
				const vec_t& forS2 = std::get<0>(memory_[idx + 1 - i]);
				std::copy(forS2.begin(), forS2.end(), s2.begin() + count);
			}
		}
		return true;
	}

	bool getAction(const int idx, label_t& action){
		assert(idx >= 0);
		if(memory_.size() == 0)		return false;
		if(idx > memory_.size() - 1)return false;
		action = std::get<1>(memory_[idx]);
		return true;
	}
	
	bool getReward(const int idx, float& reward){
		assert(idx >= 0);
		if(memory_.size() == 0)		return false;
		if(idx > memory_.size() - 1)return false;	
		reward = std::get<2>(memory_[idx]);
		return true;
	}

protected:
	std::deque<Transition> memory_;
	int max_memory_size_;
	int input_size_;
	int input_frame_count_;
	//float high_td_, low_td_;
};

// end of file
