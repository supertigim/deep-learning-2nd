#include "AIPlayer.h"

#include <thread>
#include <chrono>
#include <boost/lexical_cast.hpp>

class HumanPlayer : public Player {
protected:
    Breakout* game_;

public:
    HumanPlayer(Breakout* game)
        : game_(game)
    { 
    }

    void keyProcess(char ch) {
        switch(ch){
        case Player::LEFT_KEY:
            game_->movePaddle(Breakout::LEFT);
            break;
        case Player::RIGHT_KEY:
            game_->movePaddle(Breakout::RIGHT);
            break;
        }
    }

    void run() {

        while(1){
            game_->updateSatus();

            game_->makeScene();
            game_->render();
            game_->flipBuffer();

            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
    }
};


int main(int argc, char** argv) {

    set_random_seed(3);
    Breakout game;

    if( argc != 2) {
        return 1;
    }

    unsigned short arg = boost::lexical_cast<unsigned short>(argv[1]);
    Player * player = nullptr;
    
    switch(arg) {
    case 1:
        player = new HumanPlayer(&game);
        break;
    case 2:
        player = new AIPlayer(&game);
        break;
    default:
        break;
    }

    if(player != nullptr){
        player->initialize();
        player->run();
        delete player;   
    }

	return 0;
}

// end of file 
