/*
    Copied and Pasted by Jay (JongYoon) Kim, jyoon95@gmail.com 
*/
#pragma once 

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

using Transition = std::tuple<vec_t,	// state (* input_frame_count_)
							label_t,	// action
							float,		// reward
							vec_t,		// next state (* input_frame_count_) --> size() == 0 if terminated
							float>;		// td

//using namespace tiny_dnn::layers;
//using namespace tiny_dnn::activation;
//using namespace std;

//using fconnected_layer = tiny_dnn::layers::fc;
//using conv = tiny_dnn::layers::conv;
//using pool = tiny_dnn::layers::ave_pool;
//using tan_h = tiny_dnn::activation::tanh;
//using softmax = tiny_dnn::softmax_layer;
//using relu = tiny_dnn::relu_layer;

const char endl[] = "\n";

//typedef float	F;

//#define MIN2(a, b)							((a) > (b) ? (b) : (a))
//#define MAX2(a, b)							((a) > (b) ? (a) : (b))

// end of file
