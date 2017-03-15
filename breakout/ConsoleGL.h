#pragma once

#include <boost/shared_ptr.hpp>

const int DEF_SCR_HEIGHT = 20;
const int DEF_SCR_WIDTH = 20;

class ConsoleGL
{

protected:
	int height_, width_;
    boost::shared_ptr<unsigned char[]> front_buffer_;
    boost::shared_ptr<unsigned char[]> back_buffer_;

public:
    ConsoleGL(int height = DEF_SCR_HEIGHT, int width = DEF_SCR_WIDTH);
    ~ConsoleGL();

    void resetBuffers();
    void drawToBackBuffer(const int i, const int j, char *image);
    void render();
    void flipBuffer();

    int height()	   {return height_;}
    int width()		   {return width_;}
    int screenSize()   {return height_*width_;}
};

// end of file
