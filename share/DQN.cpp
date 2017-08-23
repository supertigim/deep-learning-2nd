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

label_t DQN::selectAction(const vec_t& state, bool is_greedy){
	
	if(!is_greedy) {
		if(bernoulli(MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * epsilon_)){
			epsilon_ *= epsilon_;
			return (label_t)uniform_rand(0, (int)nn_ptr_->out_data_size() -1);
		}
	}
	else {
		//std::cout << "MAX OUPUT: " << ((int)nn_ptr_->out_data_size() -1) << endl;
		// for natural movement when not training 
		if(bernoulli(0.05)){
			return (label_t)uniform_rand(0, (int)nn_ptr_->out_data_size() -1);
		}	
	}
	
	//std::cout << state.size() << "<- state size in select action" << endl;
	return nn_ptr_->predict_label(state);
	//vec_t q = nn_ptr_->predict(state);
	//return getMaxQIdx(q);
}

void DQN::printQValues(const vec_t& state_vector){
	vec_t qvalues = nn_ptr_->predict(state_vector);

	for(int i = 0; i < (int)nn_ptr_->out_data_size(); ++i){
		std::cout << qvalues[i] << " | ";
	}
	std::cout << endl;
}

float DQN::getTD(label_t& action, float& reward, vec_t& s1, vec_t& s2){
	vec_t q = nn_ptr_->predict(s1);
	vec_t target_q = nn_ptr_->predict(s2);
	return std::abs(q[action] - (reward + gamma_ * target_q[action]));
}

float DQN::getMaxQ(vec_t& q){
	assert(q.size() > 0);
	float ret = q[0];
	for(int i = 1 ; i < (int)nn_ptr_->out_data_size(); ++i){
		ret = std::max(ret, q[i]);
	}
	return ret;
}

label_t DQN::getMaxQIdx(vec_t& q){
	assert(q.size() > 0);
	label_t ret = 0;
	float maxq = q[0];
	for(int i = 1 ; i < (int)nn_ptr_->out_data_size(); ++i){
		if(maxq < q[i]){
			maxq = q[i];
			ret = i;
		}
	}
	return ret;
}

void DQN::update(PEReplay& replay, size_t batch_size, size_t epochs){
	std::vector<vec_t> train_input_vector, desired_output_vector;
	std::vector<label_t> target_label;
	//gradient_descent optimizer;
	//RMSprop optimizer;
	adam optimizer;
	std::vector<int> t_idx_vector;
	bool can_train = replay.getSampleIdxVector(t_idx_vector, batch_size);
	if(!can_train) return;

	for(int i = 0 ; i < t_idx_vector.size(); ++i){
		vec_t& s1 		= std::get<0>(replay[t_idx_vector[i]]);
		label_t& action = std::get<1>(replay[t_idx_vector[i]]);
		float& reward 	= std::get<2>(replay[t_idx_vector[i]]);
		vec_t& s2 		= std::get<3>(replay[t_idx_vector[i]]);

		// get Q-values from s1 using CURRENT Network 
		vec_t desired_output = nn_ptr_->predict(s1);

		// learning algorithm 	
		if(!s2.size()){
			desired_output[action] = reward;
		}
		else {
			// get maxQ from s2
			vec_t q_ = nn_ptr_->predict(s2);
			const float maxQ = getMaxQ(q_);//nn_ptr_->predict_max_value(s2);
			desired_output[action] = reward + gamma_ * maxQ;
		}
		train_input_vector.push_back(s1);
		desired_output_vector.push_back(desired_output);
		target_label.push_back(action);
	}

	//size_t training_batch = 1;	//batch_size;
	//size_t epochs = 1;			//batch_size; 

	if(batch_size > 1){
	  	optimizer.alpha *=	static_cast<tiny_dnn::float_t>(sqrt(batch_size) * gamma_);
	}

	nn_ptr_->fit<mse>(optimizer, train_input_vector, desired_output_vector, batch_size, epochs);
	std::cout << "Loss: " << nn_ptr_->get_loss<mse>(train_input_vector, desired_output_vector) << endl;
	result ret = nn_ptr_->test(train_input_vector, target_label);
	std::cout << "accuracy: " << ret.accuracy() << endl;
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
	target_nn_ptr_ = std::make_shared<network<sequential>>();
	*target_nn_ptr_ = *nn_ptr_;
}


void DDQN::update(PEReplay& replay, size_t batch_size, size_t epochs){
	
	std::vector<vec_t> train_input_vector, desired_output_vector;
	//std::vector<label_t> target_label;
	
	//gradient_descent optimizer;
	//RMSprop optimizer;
	adam optimizer;

	std::vector<int> t_idx_vector;
	bool can_train = replay.getSampleIdxVector(t_idx_vector, batch_size);
	if(!can_train) return;

	nn_ptr_.swap(target_nn_ptr_);	// double Q-network 

	for(int i = 0 ; i < t_idx_vector.size(); ++i){
		//std::vector<vec_t> train_input_vector, desired_output_vector;

		vec_t& s1 		= std::get<0>(replay[t_idx_vector[i]]);
		label_t& action = std::get<1>(replay[t_idx_vector[i]]);
		float& reward 	= std::get<2>(replay[t_idx_vector[i]]);
		vec_t& s2 		= std::get<3>(replay[t_idx_vector[i]]);

		// get Q-values from s1 using Main Network 
		vec_t double_q = nn_ptr_->predict(s1);

		std::cout << "before:" << double_q[action];
		// learning algorithm 	
		if(!s2.size()){
			double_q[action] = reward;
		}
		else {
			// 변경 전 네트워크를 통해서 타켓 Q들을 미리 얻어 둔다. 
			vec_t target_q = target_nn_ptr_->predict(s2);

			// DQN에서는 MAX-Q 값을 가져왔지만, 여기서는 해당 인덱스만..
			label_t max_index  = getMaxQIdx(target_q);
			// 미리 구해둔 Q Value들(target_q)에서 위 인덱스의 값으로 MAX Q값을 얻는다. 
			double_q[action] = reward + gamma_ * target_q[max_index];
		}
		std::cout << "  After:" << double_q[action] << endl;
		train_input_vector.push_back(s1);
		desired_output_vector.push_back(double_q);

		//nn_ptr_->train<mse>(optimizer, train_input_vector, desired_output_vector, 1, 1);
		//std::cout << "Loss: " << nn_ptr_->get_loss<mse>(train_input_vector, desired_output_vector) << endl;
	}

	//size_t training_batch = batch_size;	
	//size_t epochs = 1; //batch_size*DEFAULT_EPOCH_RATE;		

	if(batch_size > 1){
	  	optimizer.alpha *=
	    	static_cast<tiny_dnn::float_t>(sqrt(batch_size) * gamma_);
	}

	nn_ptr_->train<mse>(optimizer, train_input_vector, desired_output_vector, batch_size, epochs);
	//std::cout << "Loss: " << nn_ptr_->get_loss<mse>(train_input_vector, desired_output_vector) << endl;
}	

// end of file
