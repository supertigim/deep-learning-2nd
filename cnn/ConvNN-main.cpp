#include <iostream>
#include "tiny_dnn/tiny_dnn.h"
#include "../common.h"
#include <ctime>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

// input image size 
const int width = 10;
const int height = 10;

const int output_dims = 2; 			// location value (x, y)

const int fc_n_c = width * height;	// number of neurons on between fully connected neural networks 

/* 
	[WARNING]
	color range in tiny-dnn : -1.0 ~ 1.0  (NOT!!! 0 ~ 255) 
 */
const float dotted = 0.1f;

typedef convolutional_layer<activation::identity> conv;

// make convolitional neural network  
// 2 cnn / 2 fcnn
// input: 10*10 size image   / output: point (x,y) 
void construct_net(network<sequential>& nn) {

	// by default will use backend_t::tiny_dnn unless you compiled
	// with -DUSE_AVX=ON and your device supports AVX intrinsics
  	core::backend_t backend_type = core::default_engine();

	const int c1_kernel_size = 5;		// 5*5 kernel 
	const int c1_input_channel= 1;		// greyscale (input channel)
	const int c1_fmaps = 2;				// feature map (output channel)

	const int c2_kernel_size = 5;		// 5*5 kernel 
	const int c2_fmaps = 2;				// feature map (output channel)

	nn	<< conv<relu>(width, height, c1_kernel_size, c1_input_channel
					,c1_fmaps, padding::same, true /*add bias*/, 1/*w-stride*/, 1/*h-stride*/, backend_type)
		<< conv<relu>(width, height, c2_kernel_size,c1_fmaps
					,c2_fmaps, padding::same, true /*add bias*/, 1/*w-stride*/, 1/*h-stride*/, backend_type)
		<< fully_connected_layer<relu>(width*height*c2_fmaps, fc_n_c)
		<< fully_connected_layer<relu>(fc_n_c, output_dims);
}

float getLinfNormError(const vec_t& pred, const vec_t& desired){

	float temp = 0.0f;

	for (int d = 0; d < pred.size(); ++d) {
		temp = std::max(temp, std::abs(desired[d] - pred[d]));
	}
	return temp;
}

int main(int agrc, char** argv){

	network<sequential> nn;
	construct_net(nn);
	gradient_descent optimizer;

	vec_t desired(output_dims, 0.0f);
	vec_t input_image(width*height, 0.0f);

	srand((unsigned int)time(0));

	while (1)
	{
		float max_error = 0.0f;

		for (int r = 0; r < width*height*100; r++)
		{
			// Create training dataset 
			const int rand_i = rand() % width;
			const int rand_j = rand() % height;

			// 점하나만 찍을 때 
			input_image[rand_i + width* rand_j] = dotted; 

			// 십자가 그림 이미지
			//input_image[rand_i - 1 + width* rand_j] 	= dotted;
			//input_image[rand_i + width* rand_j] 		= dotted;
			//input_image[rand_i + 1 + width* rand_j] 	= dotted;
			//input_image[rand_i + width* (rand_j - 1)] = dotted;
			//input_image[rand_i + width* (rand_j + 1)] = dotted;

			// point (x,y) 
			desired[0] = (float)rand_i / (float)width;
			desired[1] = (float)rand_j / (float)height;

			size_t batch_size = 1;
			size_t epochs = 1;

			// prepare training set 
			std::vector<vec_t> train_input, desired_output;
			train_input.push_back(input_image);
			desired_output.push_back(desired);

			nn.fit<mse>(optimizer, train_input, desired_output, batch_size, epochs);

			vec_t result = nn.predict(input_image);

			const float linferror = getLinfNormError(result, desired);
			max_error = std::max(linferror, max_error);

			// Image buffer reset 
			input_image[rand_i + width* rand_j] = 0.0f;
		}

		std::cout << "Max error = " << max_error << endl;

		if (max_error < 0.00015) {
			nn.save("Fake-Image-CNN-Model");
			return 0;
		}
	}

	return 0;
}

// end of file 
