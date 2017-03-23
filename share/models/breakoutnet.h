#pragma once

using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

namespace TDNN_Models {

	// Neural Network for Command-Line Breakout
	void cl_breakout_net(network<sequential>& nn, 
						const int& height,
						const int& width, 
  						const int& input_channel_nums, 
  						const int& output_nums){


		typedef convolutional_layer<tan_h> conv;
		typedef max_pooling_layer<relu> pool;

		core::backend_t backend_type = core::default_engine();

		//const int height = 20;
	    //const int width = 20;
	    //const int input_channel_nums = 4;
	    //const int output_nums = 3;

		const int kernel_size = 5;					// 5*5 kernel
		const serial_size_t n_fmaps = height;		//< number of feature maps for upper layer
		const serial_size_t n_fmaps2 =height*2;	//< number of feature maps for lower layer
		const serial_size_t n_fc =height*2;		//< number of hidden units in fully-connected layer

		nn	<< conv(height, width, kernel_size, input_channel_nums, n_fmaps, padding::same, true, 1, 1, backend_type)
			 << pool(height, width, n_fmaps, 2, backend_type)
			 << conv(height/2, width/2, kernel_size, n_fmaps, n_fmaps2, padding::same, true, 1, 1, backend_type)
			 << pool(height/2, width/2, n_fmaps2, 2, backend_type)
			 << fully_connected_layer<relu>(height/4 * width/4 * n_fmaps2, n_fc, true, backend_type)
			 << fully_connected_layer<leaky_relu>(n_fc, output_nums, true, backend_type);
	}  

}  // namespace TDNN_Models


// end of file
