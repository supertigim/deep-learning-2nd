#pragma once 

#include <boost/thread/thread.hpp>
#include <boost/asio.hpp>

class Player {
public:
	enum{
		 LEFT_KEY = 44    /* ',' */
        ,RIGHT_KEY = 46   /* '.' */

        ,ENTER_KEY = 10
        ,SPACEBAR_KEY = 32
	} KeyType;

protected:
    boost::asio::io_service ios_;
    boost::thread_group tg_;
    const boost::asio::io_service::work work_;

public:
	Player();
	~Player();
	virtual void initialize() {}
    virtual void keyProcess(char ch);
    virtual void run() {};
    
    void hideInput(bool ishidden);
    char consoleKeyInput();
protected:
    void inputKeyThread();
};

// end of file
