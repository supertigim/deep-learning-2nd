/*
    Copied and Pasted by Jay (JongYoon) Kim, plenum@naver.com 
*/
#pragma once 

#include "common.h"

const int DEF_MAX_REPLAY_CNT	= 1000;
const int DEF_INPUT_FRAME_CNT	= 1;

/**
 *   Prioritized Experience Replay class 
 * 
 *   to store past transitions 
 */
class PEReplay {
public:
	PEReplay()
	:max_memory_size_(DEF_MAX_REPLAY_CNT)
	,input_frame_count_(DEF_INPUT_FRAME_CNT)
	,input_size_(0)
	//,accum_(0.0f)
	{}
	~PEReplay() {}

	void init(const int& input_size, const int& input_frame_count){
		input_size_ = input_size;
		input_frame_count_ = input_frame_count;
	}

	void clear() {
		memory_.clear();
		//accum_ = 0.0f;
	}

	int size() { return memory_.size(); }

	void addTransition(std::unique_ptr<Transition> t_ptr){
		assert(input_size_ != 0);
		if(!memory_.size()){
			memory_.push_back(std::move(t_ptr));
			return;
		}

		float& td = std::get<4>(*t_ptr);
		//accum_ += td/100.0f ;	// 평균값을 max_memory_size_로 노말라이즈
		//const float mean = accum_ / memory_.size();

		// reward가 크면, higher priotiry! 
		if(std::abs(td) > 0.1f) memory_.push_back(std::move(t_ptr));
		//if(td > 0.1f) memory_.push_back(std::move(t_ptr));
		else 					memory_.push_front(std::move(t_ptr));

		if(max_memory_size_ < memory_.size()){
			memory_.pop_front();
		}	

	}

	bool getSampleIdxVector(std::vector<int>& v, int size) {

		if(size < 0 || memory_.size() == 0 || size > memory_.size()){
			return false;
		}
		v.reserve(size);

		for (auto i = 0; i < size; ++i) {
			const int idx = uniform_rand((int)(memory_.size()/2) , (int)(memory_.size()-1));
			v.push_back(idx);
		}
		
		return true;
	}

	Transition& operator[] (int idx){
		return *memory_[idx];
	}

protected:
	std::deque<std::unique_ptr<Transition>> memory_;
	int max_memory_size_;
	int input_size_;
	int input_frame_count_;
	//float accum_;
};

// end of file
