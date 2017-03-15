#include "Agent.h"
#include "TestDrivingScene.h"

Agent::Agent(TestDrivingScene* simul)
	: simul_(simul)
	, reward_sum_(0.0f)
	, reward_max_(0.0f)
	, is_training_(false)
{}

void Agent::init(){

	simul_->compat_state_ = true;
	simul_->state_buffer_.resize(simul_->getNumStateVariables(), 0.0f);

	rl_.num_input_histories_ = 1;
	//std::cout << "Agent::init() " << simul_->getNumStateVariables() << endl;
	rl_.num_state_variables_ = simul_->getNumStateVariables();
	rl_.num_game_actions_ = simul_->getNumActions();//TODO: obtain from game, left, right, stay

	rl_.initialize();

	initMemory();
}

void Agent::initMemory(){

	rl_.memory_.reset();		

	vec_t q_values = {0.0f, 0.0f, 0.0f};
	int terminated = 0;
	for (int h = 0; h < rl_.num_input_histories_; h++) {
		rl_.recordHistory(simul_->getStateBuffer(), 0.0f, 2, q_values, terminated ); // choice 2 is stay
	}
}

const int Agent::getSelectedDir(const vec_t& q_values){
	int selected_dir = 2;

	// user supervised mode
	if (simul_->getKeyPressed(GLFW_KEY_LEFT) == true) selected_dir = 0;
	else if (simul_->getKeyPressed(GLFW_KEY_RIGHT) == true) selected_dir = 1;
	// AI mode
	else  {
		selected_dir = is_training_ == true ? rl_.getOutputLabelwithEpsilonGreedy(q_values, 0.2f) : rl_.getOutputLabelwithEpsilonGreedy(q_values, 0.0f);	
	}

	if (simul_->getKeyPressed(GLFW_KEY_Q) == true) {

		rl_.nn_.save("SELF-DRIVING-CAR-MODEL");
		std::cout << "writing complete" << endl;
	}

	return selected_dir;
}
bool Agent::handleKey(){
	
	simul_->pollEvents();

	// Check if the ESC key was pressed or the window was closed
	if (simul_->getKeyPressed(GLFW_KEY_ESCAPE) || simul_->getWindowShouldClose())
		return false;

	// training mode change key input
	static bool key_reset_flag = true;
	if (simul_->getKeyPressed(GLFW_KEY_SPACE) == true) {

		if(key_reset_flag == true) {
			is_training_ = !is_training_;

			key_reset_flag = false;

			if (is_training_) {
				std::cout << "Back ground training mode" << endl;
			}
			else {
				std::cout << "Interactive rendering mode" << endl;
			}
		}
	}
	else {
		key_reset_flag = true;
	}
	return true;
}

void Agent::driveCar() {

	int count = 0;
	while(true) {
		
		if(!handleKey()) break;

		vec_t q_values = rl_.forward();
		
		const int selected_dir = getSelectedDir(q_values);	
		simul_->processInput(selected_dir);					
		
		float reward;

		// 0 : continue
		// 1 : terminate
		int isTerminated; 
		simul_->update(!is_training_, reward, isTerminated);

		// record state and reward
		rl_.recordHistory(simul_->getStateBuffer(), reward, selected_dir, q_values, isTerminated);

		if(is_training_){	
			
			reward_sum_ += reward;

			// start state replay training at terminal state
			// this is terminal state
			if(isTerminated) {
				std::cout << "(max:" << reward_max_ << ") " << "Reward sum " << reward_sum_ << endl;
				
				if (reward_max_ < reward_sum_) {
					reward_max_ = reward_sum_;

					std::cout 	<< "**************************" << endl
								<< "** New record : " << reward_max_ << " **" << endl
								<< "**************************" << endl;

				}

				reward_sum_ = 0.0f;
			}
			if(count >= 10){
				rl_.trainRandomReplay(count);
				count =0;	
			}

			is_good_enough(reward_sum_);
		}
		else{
			simul_->render();
		}

		++count;
	}
	glfwTerminate();

}

bool Agent::is_good_enough(float distance_to_travel) {

	const float well_done_distance = 200.0f;

	if( distance_to_travel > well_done_distance) {
		std::cout 	<< endl 
				  	<< "**************************" << endl
					<< "**   Training is done!!!  " << endl
					<< "**************************" << endl;
		initMemory();
		is_training_ = false;
		return true;
	}
	return false;

}

// end of file
