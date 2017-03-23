#pragma once 

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;


const char endl[] = "\n";

typedef float	F;

#define MIN2(a, b)							((a) > (b) ? (b) : (a))
#define MAX2(a, b)							((a) > (b) ? (a) : (b))

/**
 *   Learning Algorithm Abstract Class 
 */
class learnig_algorithm {
public:
	learnig_algorithm() {}
	virtual ~learnig_algorithm() {}
	virtual void update(int batch_size) = 0;
};

// end of file
