#include <iostream>

#include <stdio.h>
#include <assert.h>

#include "ConsoleGL.h"
#include <boost/make_shared.hpp>

ConsoleGL::ConsoleGL(int height, int width)
    : height_(height), width_(width)
{
    front_buffer_ = boost::make_shared<unsigned char[]>(screenSize());
    back_buffer_ = boost::make_shared<unsigned char[]>(screenSize());

    resetBuffers();
}

ConsoleGL::~ConsoleGL(){ 
}

void ConsoleGL::resetBuffers()
{
    unsigned char* front_buffer = front_buffer_.get();
    unsigned char* back_buffer = back_buffer_.get();

    std::fill(front_buffer, front_buffer + screenSize(), '\0');
    std::fill(back_buffer, back_buffer + screenSize(), '\0');
}

void ConsoleGL::drawToBackBuffer(const int i, const int j, char *image)
{
    int ix = 0;

    assert(i >= 0);
    //if(j < 0 || j >= height_)
    //{
    //    std::cout << "[ERROR] height - " << j << endl;
    //    assert(0);
    //}
    assert(j >= 0 && j < height_); 

    unsigned char* back_buffer = back_buffer_.get();
    
    while (1) {

        if (image[ix] == '\0') break;
        assert(i + ix < width_);
        //if(i + ix >= width_){
        //    std::cout << "[ERROR] width - " << i+ix << endl;
        //    assert(0);
        //}

        back_buffer[ j + ( i + ix ) * width_] = image[ix];
        
        ix++;
    }
    //std::cout << ix << endl;
}

void ConsoleGL::render()
{
    unsigned char* front_buffer = front_buffer_.get();
    
    for (int j = 0; j < height_; j++){
        for (int i = 0; i < width_; i++)
        {
            if (front_buffer[j+ i*width_] == '\0')
                printf("%c", ' ');
            else
                printf("%c", front_buffer[j+ i*width_]);
        }
        printf ("\n");
    }
}

void ConsoleGL::flipBuffer(){
    std::swap(front_buffer_, back_buffer_);
    unsigned char* back_buffer = back_buffer_.get();
    std::fill(back_buffer, back_buffer + screenSize(), '\0');
}

// end of file 
