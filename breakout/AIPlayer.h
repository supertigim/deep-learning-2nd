#pragma once 

#include "Breakout.h"
#include "Player.h"
#include "ReinforcementLearning.h"
#include "DQN.h"

//const int DEF_CONV_CHANNEL_NUM = 5;
//const int DEF_RL_FEED_HISTORY_NUM = 4;

class AIPlayer : public Player {	
public:
	AIPlayer(Breakout* game);

	void keyProcess(char ch);
	void run();
	void run_old();
	void initialize();

protected:
	const int getSelectedDir(const vec_t& q_values);
	void initMemory();

protected:
	Breakout* game_;
	ReinforcementLearning rl_;

	std::unique_ptr<DQN> dqn_;
	network<sequential> nn_;



	bool is_training_;
	float epsilon_;
};

// end of file
