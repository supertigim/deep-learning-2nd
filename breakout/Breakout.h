#pragma once

#include "common.h"
#include "ConsoleGL.h"

const float DEFAULT_WALL_WIDTH = 0.05f;
const float DEFAULT_PADDLE_WIDTH_RATIO = 0.3f;

const float PADDLE_MOVE_RATE = 0.33f;
const float START_POSITION = 0.5f;
const float DEFAULT_BALL_SPEED = 0.5f;
const float UPDATE_BALL_SPEED = 0.04f;

class Breakout : public ConsoleGL {
public:
	typedef enum {
		STAY = 0,
		RIGHT = 1,
		LEFT = 2
	} DirType;

public:
    Breakout();
    ~Breakout();

    void toggleTrainigMode() { trainig_mode_ = !trainig_mode_;}
    bool isTraining() {return trainig_mode_;}
    int getNumActions(){return 3;} //left, right, stay

    void restart();
    const vec_t& getStateBuffer();
    void printStateBuffer();

    void movePaddle(DirType dir, float dx = PADDLE_MOVE_RATE);
    float updateSatus(float dt = UPDATE_BALL_SPEED);
    
    void makeScene();
protected:
    void normalize(float& x, float& y);

private:
    float paddle_x_;    
    float ball_x_, ball_y_;
    float ball_vel_x_, ball_vel_y_;

    const float wall_thickness_;
    const float paddle_width_;

    bool trainig_mode_;

    vec_t state_buffer_;
};


// end of file
