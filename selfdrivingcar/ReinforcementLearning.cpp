#include "ReinforcementLearning.h"

// initialize neural network
void ReinforcementLearning::initialize() {
	
	std::cout << "[ReinforcementLearning::initialize]input num:" << num_state_variables_ * num_input_histories_ << endl;
	
	const int fc1_n_c = num_state_variables_ * num_input_histories_;
	const int fc2_n_c = num_state_variables_ * num_input_histories_;

	nn_	<< fully_connected_layer<relu>(num_state_variables_ * num_input_histories_,fc1_n_c) 
		<< fully_connected_layer<relu>(fc1_n_c,fc2_n_c) 
		<< fully_connected_layer<relu>(fc2_n_c,num_game_actions_);

	// initialize replay memory
	memory_.reserve(1e5);

	gamma_ = 0.95f;

	old_input_vector_.resize(num_state_variables_ * num_input_histories_, 0.0f);
}

void ReinforcementLearning::initializeConv2D(int height, int width) {
	/*
	const int num_channels = 1;

	//assert(num_exp_replay_ >= num_input_histories_);

	nn_.initialize(num_state_variables_ * num_input_histories_
				, num_game_actions_
				, 2);

	nn_.layers_[0].act_type_ = LayerBase::LReLU;
	nn_.layers_[1].act_type_ = LayerBase::LReLU;
	nn_.layers_[2].act_type_ = LayerBase::LReLU;

	nn_.eta_ = 1e-4;
	nn_.alpha_ = 0.9;

	nn_.layers_[0].act_type_ = LayerBase::LReLU;
	//conn 0 : filter
	nn_.layers_[1].act_type_ = LayerBase::LReLU;
	//conn 1 : averaging
	nn_.layers_[2].act_type_ = LayerBase::LReLU;
	//conn 2 : full
	nn_.layers_[3].act_type_ = LayerBase::LReLU;

	nn_.layers_[1].initialize(num_input_histories_ * num_state_variables_ * num_channels + 1, LayerBase::LReLU);
	nn_.layers_[2].initialize(num_input_histories_ * num_channels + 1, LayerBase::LReLU);
	nn_.setFullConnection(1, 0.1f, 0.01f);
	nn_.setFullConnection(2, 0.1f, 0.01f);

	const int first_filter_size = 5; // 5 by 5 
	{
		ConvFilter2D filter;
		filter.initialize(first_filter_size, first_filter_size
						, 1, 1
						, 2, 2
						, 0.1, 0.01);
		ConvImage2D im;
		im.width_ = width;
		im.height_ = height;;
		ConvImage2D om;
		om.width_ = width;
		om.height_ = height;

		ConvConnection2D *new_conv = nn_.setConvConnection2D(0);

		int in_count = 0;
		int out_count = 0;
		for (int h = 0; h < num_input_histories_; ++h) {
			for (int ch = 0; ch < num_channels; ++ch) {
				new_conv->channel_list_.push_back(new ConvChannel2D(filter, (F)0.1, (F)0.01, im, in_count, om, out_count));
				out_count += num_state_variables_;
			}
			in_count += num_state_variables_;
		}
	}

	{
		ConvFilter2D filter;
		filter.initialize(width, height, 1, 1, 0, 0, 0.1, 0.01);
		ConvImage2D im;
		im.width_ = width;
		im.height_ = height;
		ConvImage2D om;
		om.width_ = 1;
		om.height_ = 1;

		int in_count = 0;
		int out_count = 0;

		ConvConnection2D *new_conv = nn_.setConvConnection2D(1);

		for (int h = 0; h < num_input_histories_; ++h) {
			for (int ch = 0; ch < num_channels; ++ch) {
				new_conv->channel_list_.push_back(new ConvChannel2D(filter, (F)0.1, (F)0.01, im, in_count, om, out_count));

				in_count += num_state_variables_;
				++out_count;
			}
		}
	}

	memory_.reserve(1e5);

	gamma_ = 0.95f;

	old_input_vector_.initialize(nn_.num_input_, true);
	next_input_vector_.initialize(nn_.num_input_, true);
	*/
}

// train with last memory
void ReinforcementLearning::trainReward() {
	trainReward(0);	
}

void ReinforcementLearning::trainRewardMemory() {
	//std::cout << "Memory count: " << memory_.count() << endl;
	for (int ix_from_end = 0; ix_from_end > -(memory_.count() - num_input_histories_); --ix_from_end)
		trainReward(ix_from_end);
}

void ReinforcementLearning::trainReward(const int ix_from_end) {

	//std::cout << "Training!!!!" << ix_from_end << endl;

	// last history is in one step future 
	const float reward_ix = memory_.getRewardFromLast(ix_from_end);
	const float maxQ = getMaxQValue(memory_.getQValuesFromLast(ix_from_end));
	const label_t selected_dir = memory_.getSelectedIxFromLast(ix_from_end);
	vec_t q_values = memory_.getQValuesFromLast(ix_from_end);
	if(memory_.getTermiatedFromLast(ix_from_end)){
		q_values[selected_dir] = reward_ix;
	}
	else {
		q_values[selected_dir] = reward_ix + gamma_ * maxQ;
	}

	makeInputVectorFromHistory(ix_from_end-1, old_input_vector_);

	size_t batch_size = 1;
	size_t epochs = 1;
	gradient_descent optimizer;

	// prepare training set 
	std::vector<vec_t> train_input, desired_output;
	train_input.push_back(old_input_vector_);
	desired_output.push_back(q_values);

	nn_.fit<mse>(optimizer, train_input, desired_output, batch_size, epochs);
}

vec_t ReinforcementLearning::forward() {
	makeInputVectorFromHistory(0, old_input_vector_);
	return nn_.predict(old_input_vector_);
}

vec_t ReinforcementLearning::forward(const vec_t& state_vector){
	return nn_.predict(state_vector);
}

// push back this to history
void ReinforcementLearning::recordHistory(const vec_t& state_vector, const float& reward, const label_t& choice, const vec_t& q_values, const int& terminated) {
	memory_.append(state_vector, choice, reward, q_values, terminated);
}

void ReinforcementLearning::makeInputVectorFromHistory(const int& ix_from_end, vec_t& input) {
	
	assert(input.size() == num_state_variables_ * num_input_histories_);
	//input.clear();

	for (int r = 0, count = 0; r < num_input_histories_; ++r, count += num_state_variables_) {
		const vec_t &state_vector 
				= memory_.getStateVectorFromLast(ix_from_end - r);

		//std::copy(state_vector.begin(), state_vector.end(), std::back_inserter(input));
		std::copy(state_vector.begin(), state_vector.end(), input.begin() + count);
	}
}

// epsilon-greedy
label_t ReinforcementLearning::getOutputLabelwithEpsilonGreedy(const vec_t& q_values, const float& epsilon){
	if (epsilon > 0.0) {
    	if ((float)rand() / RAND_MAX < epsilon) {
            return rand() % num_game_actions_;
        }
    }
    return getMaxQValue(q_values);
}

const float_t ReinforcementLearning::getMaxQValue(const vec_t& q_values) {
	float_t ret = 0.0f;

	for(int i = 0 ; i < q_values.size() ; ++i){
		ret = std::max(ret, q_values[i]);
	}
	return ret;
}

// end of file
