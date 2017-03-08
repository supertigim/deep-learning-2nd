#include <iostream>
#include "tiny_dnn/tiny_dnn.h"
#include "../common.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

const int xor_input = 2;
const int xor_output = 1;

// It works well only when number of neurons on each hidden layer is above 5
// However, if you use tan_h as activation function, above 3 is good enough
const int hidden_layer_c = 5; 


// make fully connected neural network 
// input:2  / output:1 / hidden layer:1
void construct_net(network<sequential>& nn) {

	nn	<< fully_connected_layer<relu>(xor_input,hidden_layer_c) 
		<< fully_connected_layer<relu>(hidden_layer_c,hidden_layer_c) 
		<< fully_connected_layer<relu>(hidden_layer_c,xor_output);
}

int main(int argc, char** argv){

	network<sequential> nn;
	gradient_descent optimizer;

	construct_net(nn);

	// load xor data set 
  	std::vector<vec_t> 
  		train_input = {
		{0.0f, 0.0f}, 	{0.0f, 1.0f}, 	{1.0f, 0.0f}, 	{1.0f, 1.0f}
  	}
  		,desired_out = {
		{0.0f},			{1.0f},			{1.0f},			{0.0f}
  	};

	std::cout << "start training" << endl;

	size_t batch_size = 1;
    size_t epochs = 1000;

    nn.fit<mse>(optimizer, train_input, desired_out, batch_size, epochs);

    std::cout << "Test..." << endl;

	
	for(int i = 0 ; i < train_input.size() ; i++) {
		vec_t resutl = nn.predict(train_input[i]);

		std::cout << train_input[i][0] << " | " << train_input[i][1]
				<< " --> "	<<resutl[0] << endl;

	}

	return 0;
}

// end of file