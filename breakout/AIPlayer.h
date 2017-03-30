#pragma once 

#include "Breakout.h"
#include "Player.h"
#include "DQN.h"

//const int DEF_CONV_CHANNEL_NUM = 5;
//const int DEF_RL_FEED_HISTORY_NUM = 4;

class AIPlayer : public Player {	
public:
	AIPlayer(Breakout* game);

	void keyProcess(char ch);
	void run();
	void initialize();

protected:
	Breakout* game_;
	ReinforcementLearning rl_;

	std::unique_ptr<DQN> dqn_;
	//network<sequential> nn_;
	Replay replay_;

	bool is_training_;
	float epsilon_;
	int input_frame_count_;
};

// end of file
