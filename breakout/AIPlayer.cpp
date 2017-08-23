#include "AIPlayer.h"
#include <thread>
#include <chrono>
#include "BoNet.h"

AIPlayer::AIPlayer(Breakout* game)
	: game_(game)
	, is_training_(true)
	, input_frame_count_(4)
	, min_batch_(2)
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

	game_->makeScene();
	game_->flipBuffer();

	std::shared_ptr<network<sequential>> nn 
		= std::make_shared<BoNet>(	"Breakout Net"
									,game_->height()
									,game_->width()
									,input_frame_count_
									,game_->getNumActions());

	//change layer initialization
	nn->weight_init(weight_init::he(3));
	nn->bias_init(weight_init::constant(1.0));

	dqn_ = std::make_unique<DDQN>();
	dqn_->initialize(nn);
	replay_.init(game_->screenSize(), input_frame_count_);
}

void AIPlayer::updateStateVector(){

	vec_t t = game_->getStateBuffer();
	past_states_.push_back(t);
	if (past_states_.size() > input_frame_count_) {
		past_states_.pop_front();
	}
}


bool AIPlayer::getState(vec_t& t){
	if (past_states_.size() < input_frame_count_) { 
		t.clear();
		return false; 
	}

	for(int i = 0, count = 0; i < input_frame_count_ ; ++i, count += game_->screenSize())
		std::copy(past_states_[i].begin(), past_states_[i].end(), t.begin() + count);	
	return true;
}

void AIPlayer::run(){

	std::cout << "Game Start!! \n";

	while(true){

		std::unique_ptr<Transition> t_ptr = std::make_unique<Transition>(game_->screenSize() * input_frame_count_
																	,0
																	,0.0f
																	,game_->screenSize() * input_frame_count_
																	,0.0f);
		vec_t& state 		= std::get<0>(*t_ptr);
		label_t& action 	= std::get<1>(*t_ptr);
		float& reward 		= std::get<2>(*t_ptr);
		vec_t& next_state 	= std::get<3>(*t_ptr);
		float& td 			= std::get<4>(*t_ptr);
      
		if(getState(state))	action = (is_training_ == true)? dqn_->selectAction(state): dqn_->selectAction(state, true);
		else 				action = 0;

		std::cout << "selected_dir - " << action << endl; 
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

        reward = game_->updateSatus();

        updateStateVector();

        if( reward < 0)	next_state.clear();
        else 			getState(next_state);

        if(state.size() && is_training_){
        	std::cout << "Training... " << "\n";
        	td = reward;
			replay_.addTransition(std::move(t_ptr));	

			if (replay_.size() >= min_batch_) {
				dqn_->update(replay_, min_batch_);
			}
		}
		else{
			std::this_thread::sleep_for(std::chrono::milliseconds(40));
#if defined(__APPLE__)           
            system("clear");
#else 
            system("cls");
#endif
		}
        
	}
}

// end of file
