#include "AIPlayer.h"
#include <thread>
#include <chrono>
#include "models/breakoutnet.h"

AIPlayer::AIPlayer(Breakout* game)
	: game_(game)
	, is_training_(true)
{
	if(is_training_ != game_->isTraining()) {
		game_->toggleTrainigMode();
	}
}

void AIPlayer::keyProcess(char ch){
	
	switch(ch){
    case Player::SPACEBAR_KEY:
    case Player::ENTER_KEY:
    	is_training_ = !is_training_;
    	game_->toggleTrainigMode();
    	break;
	}
}

void AIPlayer::initialize(){

	rl_.num_input_histories_ = 4;
	
	//std::cout << "Agent::init() " << simul_->getNumStateVariables() << endl;
	rl_.num_state_variables_ = game_->screenSize();
	rl_.num_game_actions_ = game_->getNumActions();

	rl_.initializeConv2D(game_->height(), game_->width());

	game_->makeScene();
	game_->flipBuffer();

	initMemory();

	TDNN_Models::cl_breakout_net(nn_, 
								game_->height(),
								game_->width(), 
								rl_.num_input_histories_, 
								game_->getNumActions());

	dqn_ = std::make_unique<DQN>(nn_,
								game_->screenSize(), 
								rl_.num_input_histories_, 
								game_->getNumActions());
}

void AIPlayer::initMemory(){

	rl_.memory_.reset();		

	vec_t q_values = {0.0f, 0.0f, 0.0f};
	int terminated = 0;
	for (int h = 0; h < rl_.num_input_histories_; h++) {
		rl_.recordHistory(game_->getStateBuffer(), 0.0f, 2, q_values, terminated ); // choice 2 is stay
	}
}

const int AIPlayer::getSelectedDir(const vec_t& q_values){
	 const int selected_dir = is_training_ == true ? 
	 	rl_.getOutputLabelwithEpsilonGreedy(q_values, 0.2f) : rl_.getOutputLabelwithEpsilonGreedy(q_values, 0.0f);

	 return selected_dir;
}
void AIPlayer::run(){

	std::cout << "Game Start!! \n";
	int count = 0;
	std::deque<vec_t> past_states;

	while(true){
      
        Transition transition(rl_.num_state_variables_, 0, 0.0f,0);
       
		vec_t& input_to_replay = std::get<0>(transition);
		game_->getStateBuffer(input_to_replay);

		past_states.push_back(input_to_replay);

		label_t& action = std::get<1>(transition);
		if (past_states.size() < rl_.num_input_histories_) {
			action = 0; // stay because it doesn't have enough past states to make input to the net
		}
		else {
			if (past_states.size() > rl_.num_input_histories_) {
				past_states.pop_front();
			}

			vec_t input_to_nn(rl_.num_state_variables_*rl_.num_input_histories_);
			for(int i = 0, count = 0; i < rl_.num_input_histories_ ; ++i, count += rl_.num_state_variables_){
				std::copy(past_states[i].begin(), past_states[i].end(), input_to_nn.begin() + count);	
			}
			
			//vec_t q_values = rl_.forward(input_to_nn);
			//action = getSelectedDir(q_values);
			action = (is_training_ == true)?
					dqn_->selectAction(input_to_nn):
					dqn_->selectAction(input_to_nn, 0.0f);
		}

        // s2 = policy(s1|a)
        switch (action) {
        case 2:
        	game_->movePaddle(Breakout::LEFT);
            break;
        case 1:
            game_->movePaddle(Breakout::RIGHT);
            break;
        case 0:
            // do nothing
            break;
        default:
            std::cout << "Wrong direction " << endl;
        }

        game_->makeScene();
        if(is_training_ == false) 
        	game_->render(); // need to render for conv	
        game_->flipBuffer();

        float& reward = std::get<2>(transition);
        reward = game_->updateSatus();

        int& isTerminated = std::get<3>(transition);
        //bool isTerminated = false;
        if( reward < 0){
        	isTerminated = 1;
        	//reward = 0.0f;
        }

		if(is_training_){
			// store transition 	
			dqn_->addTransition(transition);
			if(count >= 10){
				dqn_->update(count);
				count = 0;
			}
			++count;

			std::cout << "Training... " << "\n";
		}
		else{
			std::this_thread::sleep_for(std::chrono::milliseconds(40));
#if defined(__APPLE__)           
            system("clear");
#else 
            system("cls");
#endif
		}
        std::cout << "selected_dir - " << action << endl; 
	}
}

void AIPlayer::run_old(){

	int reward_sum = 0;
	int reward_max = 0;
	std::cout << "Game Start!! \n";
	int count = 0;

	while(true){

#if defined(__APPLE__)           
            //system("clear");
#else 
            system("cls");
#endif
        if (is_training_ == true)	std::cout << "Training... " << "\n";
        else{
      		system("clear");  	
        	std::this_thread::sleep_for(std::chrono::milliseconds(40));
        } 				

		vec_t q_values = rl_.forward();
        const int selected_dir = getSelectedDir(q_values);	// epsilon-greedy
        std::cout << "selected_dir - " << selected_dir << endl; 

        // s2 = policy(s1|a)
        switch (selected_dir) {
        case 2:
        	game_->movePaddle(Breakout::LEFT);
            break;
        case 1:
            game_->movePaddle(Breakout::RIGHT);
            break;
        case 0:
            // do nothing
            break;
        default:
            std::cout << "Wrong direction " << endl;
        }

        game_->makeScene();
        if(is_training_ == false) 
        	game_->render(); // need to render for conv	
        game_->flipBuffer();
        float reward = game_->updateSatus();

        bool isTerminated = false;
        if( reward < 0){
        	isTerminated = true;
        	//reward = 0.0f;
        }

        // record state and reward
		rl_.recordHistory(game_->getStateBuffer(), reward, selected_dir, q_values, isTerminated);
		reward_sum += reward;

		if (is_training_ == true && count >= 10 ){
			rl_.trainRandomReplay(count);
			count = 0;

			std::cout << "QValue: " 
					<< q_values[0] <<" | "
					<< q_values[1] <<" | "
					<< q_values[2] <<" | "
					<< "Action: " << selected_dir << endl;	
		}
		/*
		if (isTerminated) {
			//if (reward_max < reward_sum) {
			//	reward_max = reward_sum;
			//}
			reward_sum =0.0f;
			if (is_training_ == true) rl_.trainRewardMemory();
			initMemory();
		}
		*/
		++count;
	}
}


// end of file
