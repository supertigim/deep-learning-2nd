#include "Player.h"
#include <termios.h>
#include <unistd.h>

Player::Player()
	: work_(ios_)
{
	tg_.create_thread(
    	boost::bind(
    		&boost::asio::io_service::run, 
    		boost::ref(ios_))
	); 

    ios_.post(
    	boost::bind(
    		&Player::inputKeyThread,
    		this)
    );

    hideInput(true);
}

Player::~Player(){
	hideInput(false); 

	ios_.stop();
	tg_.join_all();
}

char Player::consoleKeyInput(){
    return std::cin.get();
}

void Player::hideInput(bool ishidden){

    termios t;
    tcgetattr(STDIN_FILENO, &t);
    if( ishidden == true){
        t.c_lflag &= ~ECHO;      // no display 
        t.c_lflag &= ~ICANON;    // no buffer    
    } 
    // restore normal terminal condition
    else {
        t.c_lflag |= ECHO;      
        t.c_lflag |= ICANON;    
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &t);  
}

void Player::inputKeyThread(){

	char ch;

	while(1){
		ch = consoleKeyInput();
		keyProcess(ch);
	} 
}

void Player::keyProcess(char ch){
	switch(ch){
	case Player::LEFT_KEY:
    	std::cout << "LEFT KEY is pressed!!! \n";
    	break;
	case Player::RIGHT_KEY:
    	std::cout << "RIGHT KEY is pressed!!! \n";
    	break;
    case Player::ENTER_KEY:
    	break;
	}
}

// end of file
