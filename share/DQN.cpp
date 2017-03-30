#include "DQN.h"

////////////////////////////////////////////////////////////////////////////////
//		DQN Implementation 
////////////////////////////////////////////////////////////////////////////////
DQN::DQN()
	:gamma_(DEFAULT_LEARNING_RATE)
	,epsilon_(EPSILON_DECAY_RATE)
	,nn_ptr_(nullptr)
{}


void DQN::initialize(std::shared_ptr<network<sequential>>& nn_ptr){
	nn_ptr_ = std::move(nn_ptr);
}

void DQN::update(Replay& replay, int batch_size){
	std::vector<vec_t> train_input_vector, desired_output_vector;
	//gradient_descent optimizer;
	//RMSprop optimizer;
	adam optimizer;
	std::vector<int> idx_vector;
	bool can_train = replay.getSampleIdxVector(idx_vector, batch_size);
	if(!can_train) return;

	int i;
	for( i = 0 ; i < idx_vector.size(); ++i){
		vec_t s1, s2;
		if(!replay.getState(idx_vector[i],s1,s2)){
			--batch_size;
			continue;
		}
		label_t action;
		replay.getAction(idx_vector[i],action);
		float reward;
		replay.getReward(idx_vector[i],reward);

		// get Q-values from s1 using CURRENT Network 
		vec_t desired_output = nn_ptr_->predict(s1);

		// learning algorithm 	
		if(!s2.size()){
			desired_output[action] = reward;
		}
		else {
			// get maxQ from s2
			const float maxQ = nn_ptr_->predict_max_value(s2);
			desired_output[action] = reward + gamma_ * maxQ;
		}
		train_input_vector.push_back(s1);
		desired_output_vector.push_back(desired_output);
	}

	size_t training_batch = 1;	//batch_size;
	size_t epochs = 1;			//batch_size; 

	if(training_batch > 1){
	  	optimizer.alpha *=	static_cast<tiny_dnn::float_t>(sqrt(training_batch) * gamma_);
	}

	nn_ptr_->fit<mse>(optimizer, train_input_vector, desired_output_vector, training_batch, epochs);
	std::cout << "Loss: " << nn_ptr_->get_loss<mse>(train_input_vector, desired_output_vector) << endl;
}

label_t DQN::selectAction(const vec_t& state, bool is_greedy){
	
	if(!is_greedy) {
		if(bernoulli(epsilon_)){
			epsilon_ = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * epsilon_;
			return (label_t)uniform_rand(0, (int)nn_ptr_->out_data_size()-1);
		}
	}
	else {
		// for natural movement when not training 
		if(bernoulli(0.05)){
			return (label_t)uniform_rand(0, (int)nn_ptr_->out_data_size()-1);
		}	
	}
	
	//std::cout << state.size() << "<- state size in select action" << endl;
	return nn_ptr_->predict_label(state);
}

void DQN::printQValues(const vec_t& state_vector){
	vec_t qvalues = nn_ptr_->predict(state_vector);

	for(int i = 0; i < qvalues.size(); ++i){
		std::cout << qvalues[i] << " | ";
	}
	std::cout << endl;
}

////////////////////////////////////////////////////////////////////////////////
//		Double-DQN Implementation 
//
//		http://ishuca.tistory.com/396 내용 참고 
//	
////////////////////////////////////////////////////////////////////////////////

DDQN::DDQN()
	:DQN::DQN()
{}

void DDQN::initialize(std::shared_ptr<network<sequential>>& nn_ptr){
	DQN::initialize(nn_ptr);
	//nn_ptr = std::move(nn_ptr);
	target_nn_ptr_ = std::make_shared<network<sequential>>();
	*target_nn_ptr_ = *nn_ptr_;
}


void DDQN::update(Replay& replay, int batch_size){
	
	std::vector<vec_t> train_input_vector, desired_output_vector;
	
	//gradient_descent optimizer;
	//RMSprop optimizer;
	adam optimizer;

	std::vector<int> idx_vector;
	bool can_train = replay.getSampleIdxVector(idx_vector, batch_size);
	//bool can_train = replay.getPriotizedSampleIdxVector(idx_vector, batch_size);
	if(!can_train) return;

	nn_ptr_.swap(target_nn_ptr_);	// double Q-network 

	int cnt = 0;
	for(int i = 0 ; i < idx_vector.size(); ++i){
		vec_t s1, s2;
		if(!replay.getState(idx_vector[i],s1,s2))	continue;

		label_t action;
		replay.getAction(idx_vector[i],action);
		float reward;
		replay.getReward(idx_vector[i],reward);

		// get Q-values from s1 using Main Network 
		vec_t double_q = nn_ptr_->predict(s1);

		// learning algorithm 	
		if(!s2.size()){
			double_q[action] = reward;
		}
		else {
			// 변경 전 네트워크를 통해서 타켓 Q들을 미리 얻어 둔다. 
			vec_t target_q = target_nn_ptr_->predict(s2);

			// DQN에서는 MAX-Q 값을 가져왔지만, 여기서는 해당 인덱스만..
			label_t max_index  = nn_ptr_->predict_label(s2);
			// 미리 구해둔 Q Value들(target_q)에서 위 인덱스의 값으로 MAX Q값을 얻는다. 
			double_q[action] = reward + gamma_ * target_q[max_index];
		}
		train_input_vector.push_back(s1);
		desired_output_vector.push_back(double_q);
		++cnt;
	}

	size_t training_batch = 1;		//cnt;
	size_t epochs = 1;				//batch_size; //to make it faster... but why... 

	if(training_batch > 1){
	  	optimizer.alpha *=
	    	static_cast<tiny_dnn::float_t>(sqrt(training_batch) * gamma_);
	}

	nn_ptr_->fit<mse>(optimizer, train_input_vector, desired_output_vector, training_batch, epochs);
	std::cout << "Loss: " << nn_ptr_->get_loss<mse>(train_input_vector, desired_output_vector) << endl;
}	

// end of file
