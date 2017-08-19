#include "AIPlayer.h"
#include <thread>
#include <chrono>
#include "models/breakoutnet.h"

AIPlayer::AIPlayer(Breakout* game)
	: game_(game)
	, is_training_(true)
	, input_frame_count_(1)
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

	input_frame_count_ = 4;

	game_->makeScene();
	game_->flipBuffer();

	std::shared_ptr<network<sequential>> nn = std::make_shared<network<sequential>>();
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
	TDNN_Models::cl_breakout_net(*nn, 
								backend_type,
								game_->height(),
								game_->width(), 
								input_frame_count_, 
								game_->getNumActions());

	//change layer initialization
	nn->weight_init(weight_init::he());
	nn->bias_init(weight_init::constant(1.0));

	dqn_ = std::make_unique<DDQN>();
	dqn_->initialize(nn);
	replay_.init(game_->screenSize(), input_frame_count_);
}

void AIPlayer::run(){

	std::cout << "Game Start!! \n";
	int count = 0;
	std::deque<vec_t> past_states;

	while(true){
      
        Transition transition(game_->screenSize(), 0, 0.0f,0);
       
		vec_t& input_to_replay = std::get<0>(transition);
		game_->getStateBuffer(input_to_replay);

		past_states.push_back(input_to_replay);

		label_t& action = std::get<1>(transition);
		if (past_states.size() < input_frame_count_) {
			action = 0; // stay because it doesn't have enough past states to make input to the net
		}
		else {
			if (past_states.size() > input_frame_count_) {
				past_states.pop_front();
			}

			vec_t input_to_nn(game_->screenSize()*input_frame_count_);
			for(int i = 0, count = 0; i < input_frame_count_ ; ++i, count += game_->screenSize()){
				std::copy(past_states[i].begin(), past_states[i].end(), input_to_nn.begin() + count);	
			}
	
			action = (is_training_ == true)? dqn_->selectAction(input_to_nn): dqn_->selectAction(input_to_nn, true);
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
        if(is_training_ == false) game_->render(); 
        game_->flipBuffer();

        float& reward = std::get<2>(transition);
        reward = game_->updateSatus();

        int& isTerminated = std::get<3>(transition);
        if( reward < 0){
        	isTerminated = 1;
        }

		if(is_training_){
			replay_.push_back(transition); // store transition 	
			if(count >= 4){
				dqn_->update(replay_, count);
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

// end of file
