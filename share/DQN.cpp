#include "DQN.h"

DQN::DQN(network<sequential>& nn, const int& input_size, const int& input_frame_count, const int& output_nums)
	:nn_(nn), gamma_(DEFAULT_LEARNING_RATE), output_nums_(output_nums)
{
	replay_.init(input_size, input_frame_count);
	output_nums_ = output_nums;
}

void DQN::update(int batch_size){

	std::vector<vec_t> train_input_vector, desired_output_vector;
	
	//gradient_descent optimizer;
	//RMSprop optimizer;
	adam optimizer;

	std::vector<int> idx_vector;
	replay_.getSampleIdxVector(idx_vector, batch_size);

	int i ;
	for( i = 0 ; i < idx_vector.size(); ++i){
		vec_t s1, s2;
		if(!replay_.getState(idx_vector[i],s1,s2)){
			--batch_size;
			continue;
		}
		label_t action;
		replay_.getAction(idx_vector[i],action);
		float reward;
		replay_.getReward(idx_vector[i],reward);

		// get Q-values from s1 using CURRENT Network 
		vec_t desired_output = nn_.predict(s1);
		//vec_t desired_output = forward(s2);
		//vec_t desired_output(num_game_actions_,0.0f);

		// learning algorithm 	
		if(!s2.size()){
			desired_output[action] = reward;
		}
		else {
			// get maxQ from s2
			const float maxQ = nn_.predict_max_value(s2);
			desired_output[action] = reward + gamma_ * maxQ;
		}
		train_input_vector.push_back(s1);
		desired_output_vector.push_back(desired_output);
	}

	size_t training_batch = 1;//batch_size;	//batch_size;
	size_t epochs = 1;//batch_size;			//batch_size; //to make it faster... but why... 

	if(training_batch > 1){
	  	optimizer.alpha *=
	    	static_cast<tiny_dnn::float_t>(sqrt(training_batch) * gamma_);
	}

	nn_.fit<mse>(optimizer, train_input_vector, desired_output_vector, training_batch, epochs);
	//nn_.fit<mse>(optimizer, train_input_vector, desired_output_vector);
}

label_t DQN::selectAction(const vec_t& state, float epsilon){
	if (epsilon > 0.0) {
		if(bernoulli(epsilon)){
			return (label_t)uniform_rand(0, output_nums_-1);
		}
	}
	
	return nn_.predict_label(state);
}

void DQN::printQValues(const vec_t& state_vector){
	vec_t qvalues = nn_.predict(state_vector);

	for(int i = 0; i < qvalues.size(); ++i){
		std::cout << qvalues[i] << " | ";
	}
	std::cout << endl;
}

// end of file
