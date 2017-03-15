#include "ReinforcementLearning.h"

// initialize neural network
void ReinforcementLearning::initialize() {
	
	std::cout << "[ReinforcementLearning::initialize]input num:" << num_state_variables_ * num_input_histories_ << endl;
	
	const int fc1_n_c = num_state_variables_ * num_input_histories_;
	const int fc2_n_c = num_state_variables_ * num_input_histories_;

	nn_	<< fully_connected_layer<leaky_relu>(num_state_variables_ * num_input_histories_,fc1_n_c) 
		<< fully_connected_layer<leaky_relu>(fc1_n_c,fc2_n_c) 
		<< fully_connected_layer<leaky_relu>(fc2_n_c,fc2_n_c) 
		<< fully_connected_layer<leaky_relu>(fc2_n_c,fc2_n_c) 
		<< fully_connected_layer<softmax>(fc2_n_c,num_game_actions_);

	gamma_ = 0.90f;

	/*
	nn_	<< fully_connected_layer<tan_h>(num_state_variables_ * num_input_histories_,fc1_n_c) 
		<< fully_connected_layer<tan_h>(fc1_n_c,fc2_n_c) 
		<< fully_connected_layer<tan_h>(fc1_n_c,fc2_n_c) 
		<< fully_connected_layer<tan_h>(fc2_n_c,num_game_actions_);

	gamma_ = 0.95f;
	//*/

	old_input_vector_.resize(num_state_variables_ * num_input_histories_, 0.0f);
}

void ReinforcementLearning::initializeConv2D(int height, int width) {

	typedef convolutional_layer<activation::identity> conv;
	typedef max_pooling_layer<relu> pool;

	// by default will use backend_t::tiny_dnn unless you compiled
	// with -DUSE_AVX=ON and your device supports AVX intrinsics
	core::backend_t backend_type = core::default_engine();

	const int kernel_size = 5;		// 5*5 kernel

	const serial_size_t n_fmaps = height;  ///< number of feature maps for upper layer
	const serial_size_t n_fmaps2 =height*2;  ///< number of feature maps for lower layer
	const serial_size_t n_fc =height*2;  ///< number of hidden units in fully-connected layer

	nn_ << conv(height, width, kernel_size, num_input_histories_, n_fmaps, padding::same, true, 1, 1, backend_type)
		<< pool(height, width, n_fmaps, 2, backend_type)
		<< conv(height/2, width/2, kernel_size, n_fmaps, n_fmaps2, padding::same, true, 1, 1, backend_type)
		<< pool(height/2, width/2, n_fmaps2, 2, backend_type)
		<< fully_connected_layer<relu>(height/4 * width/4 * n_fmaps2, n_fc, true, backend_type)
		<< fully_connected_layer<relu>(n_fc, num_game_actions_, true, backend_type);

	//const int num_channels = 1;
	//const int fc_n_c = width * height;

	//// by default will use backend_t::tiny_dnn unless you compiled
	//// with -DUSE_AVX=ON and your device supports AVX intrinsics
  	//core::backend_t backend_type = core::default_engine();

	//const int c1_kernel_size = 5;		// 5*5 kernel
	//// input channel <= here use 4 scree captures
	//const int c1_input_channel= num_input_histories_;
	//// feature map (output channel)
	//const int c1_fmaps = num_channels * num_input_histories_;	

	//const int c2_kernel_size = 5;		// 5*5 kernel 
	//// feature map (output channel)
	//const int c2_fmaps = num_channels * num_input_histories_;	

	//nn_	<< conv<tan_h>(width, height, c1_kernel_size, c1_input_channel
	//				,c1_fmaps, padding::same, true /*add bias*/, 1/*w-stride*/, 1/*h-stride*/, backend_type)
	//	<< conv<leaky_relu>(width, height, c2_kernel_size,c1_fmaps
	//				,c2_fmaps, padding::same, true /*add bias*/, 1/*w-stride*/, 1/*h-stride*/, backend_type)
	//	<< fully_connected_layer<leaky_relu>(width*height*c2_fmaps, fc_n_c)
	//	<< fully_connected_layer<leaky_relu>(fc_n_c, num_game_actions_);

	gamma_ = 0.95f;

	old_input_vector_.resize(num_state_variables_ * num_input_histories_, 0.0f);

}

// train with last memory
void ReinforcementLearning::trainReward() {
	trainReward(0);	
}

// DQN
void ReinforcementLearning::trainRandomReplay(int minibatch) {

	if(memory_.num_elements_ < num_input_histories_ + minibatch){
		std::cout << "not enough" << endl ;
		return;
	}

	std::vector<vec_t> train_input, desired_output;
	size_t batch_size = minibatch;
	size_t epochs = minibatch; // to make it faster... but why... 
	//gradient_descent optimizer;
	adam optimizer;
	
	while(minibatch > 0){
		//std::vector<vec_t> train_input, desired_output;

		const int m = uniform_rand(num_input_histories_
						,memory_.num_elements_ - 1 - num_input_histories_);
		const int ix_from_end = m - (memory_.num_elements_ - 1);

		const float reward_ix = memory_.getRewardFromLast(ix_from_end);
		vec_t q_values = memory_.getQValuesFromLast(ix_from_end);
		const label_t action = memory_.getSelectedIxFromLast(ix_from_end);

		makeInputVectorFromHistory(ix_from_end-1, old_input_vector_);
		const float maxQ = getMaxQValue(forward(old_input_vector_));	

		if(memory_.getTermiatedFromLast(ix_from_end)){
			q_values[action] = reward_ix;
		}
		else {
			q_values[action] = reward_ix + gamma_ * maxQ;
		}

		train_input.push_back(old_input_vector_);
		desired_output.push_back(q_values);

		//nn_.train<mse>(optimizer, train_input, desired_output);	

		--minibatch;
	}

  	optimizer.alpha *=
    	static_cast<tiny_dnn::float_t>(sqrt(batch_size) * gamma_);
	
	nn_.fit<mse>(optimizer, train_input, desired_output, batch_size, epochs);
}

void ReinforcementLearning::trainRewardMemory(){
	for (int ix_from_end = 0; ix_from_end > -(memory_.count() - num_input_histories_); --ix_from_end)
		trainReward(ix_from_end);
}

void ReinforcementLearning::trainReward(const int ix_from_end) {

	const float reward_ix = memory_.getRewardFromLast(ix_from_end);
	const label_t selected_dir = memory_.getSelectedIxFromLast(ix_from_end);
	vec_t q_values = memory_.getQValuesFromLast(ix_from_end);

	makeInputVectorFromHistory(ix_from_end-1, old_input_vector_);
	const float maxQ = getMaxQValue(forward(old_input_vector_));	

	if(memory_.getTermiatedFromLast(ix_from_end)){
		q_values[selected_dir] = reward_ix;
	}
	else {
		q_values[selected_dir] = reward_ix + gamma_ * maxQ;
	}

	//gradient_descent optimizer;
	adam optimizer;

	// prepare training set 
	std::vector<vec_t> train_input, desired_output;
	train_input.push_back(old_input_vector_);
	desired_output.push_back(q_values);

	nn_.train<mse>(optimizer, train_input, desired_output);
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

	for (int r = 0, count = 0; r < num_input_histories_; ++r, count += num_state_variables_) {
		const vec_t &state_vector 
				= memory_.getStateVectorFromLast(ix_from_end - r);

		std::copy(state_vector.begin(), state_vector.end(), input.begin() + count);
	}
}

// epsilon-greedy
label_t ReinforcementLearning::getOutputLabelwithEpsilonGreedy(const vec_t& q_values, const float& epsilon){
	if (epsilon > 0.0) {
		if(bernoulli(epsilon)){
			//std::cout << "Random Action" << endl;
			return 	uniform_rand(0,num_game_actions_ -1);
		}
		//std::cout << "Action from Q-Values" << endl;

		//const int m = uniform_rand(num_input_histories_
		//				,memory_.num_elements_ - 1 - num_input_histories_);

    	//if ((float)rand() / RAND_MAX < epsilon) {
        //    return rand() % num_game_actions_;
        //}
    }
    return getMaxQLabel(q_values);
}

const label_t ReinforcementLearning::getMaxQLabel(const vec_t& q_values) {

	label_t ix = 0;
	float max = 0.0f;

    for (int d = 0; d < q_values.size(); ++d) {
        if (max < q_values[d]) {
            max = q_values[d];
            ix = d;
        }
    }

    return ix;
}

const float_t ReinforcementLearning::getMaxQValue(const vec_t& q_values) {
	float_t ret = 0.0f;

	for(int i = 0 ; i < q_values.size() ; ++i){
		ret = std::max(ret, q_values[i]);
	}
	return ret;
}

// end of file
