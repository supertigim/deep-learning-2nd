#pragma once 

using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

namespace TDNN_Models {

	// Neural Network for Self Driving Car 
	void self_driving_car_net(network<sequential>& nn, 
								const int& input_num,
		  						const int& output_num){

    //const int input_nums_ = 13;
    //const int output_nums_ = 3;

	assert(input_num > 0 && output_num > 0);

	const int fc1_n_c = input_num ;
	const int fc2_n_c = input_num ;
	const int fc3_n_c = input_num ;

	nn	<< fully_connected_layer<relu>(input_num,fc1_n_c) 
		<< fully_connected_layer<leaky_relu>(fc1_n_c,fc2_n_c) 
		<< fully_connected_layer<leaky_relu>(fc2_n_c,fc3_n_c) 
		<< fully_connected_layer<leaky_relu>(fc3_n_c,fc3_n_c) 
		//<< fully_connected_layer<leaky_relu>(fc3_n_c,fc3_n_c) 
		//<< fully_connected_layer<softmax>(fc3_n_c,num_game_actions_);
		<< fully_connected_layer<leaky_relu>(fc3_n_c,output_num);
	} 

	// Alternative Neural Network for Self Driving Car 
	void self_driving_car_alt1_net(network<sequential>& nn, 
								const int& input_num,
		  						const int& output_num){

		assert(input_num > 0 && output_num > 0);
	    //const int input_nums_ = 13;
	    //const int output_nums_ = 3;

		const int fc1_n_c = input_num ;
		const int fc2_n_c = input_num ;
		const int fc3_n_c = input_num ;

		nn	<< fully_connected_layer<tan_h>(input_num,fc1_n_c) 
			<< fully_connected_layer<tan_h>(fc1_n_c,fc2_n_c) 
			<< fully_connected_layer<tan_h>(fc2_n_c,fc3_n_c) 
			<< fully_connected_layer<tan_h>(fc2_n_c,fc3_n_c) 
			<< fully_connected_layer<tan_h>(fc3_n_c,output_num);
		} 

	// Neural Network for Self Driving Car 
	void self_driving_car_alt2_net(network<sequential>& nn, 
								const int& input_num,
		  						const int& output_num){

		assert(input_num > 0 && output_num > 0);
	    //const int input_nums_ = 13;
	    //const int output_nums_ = 3;

		const int fc1_n_c = (input_num+output_num)/3+1;
		const int fc2_n_c = (input_num+output_num)/3+1;
		const int fc3_n_c = (input_num+output_num)/3+1;

		nn	<< fully_connected_layer<elu>(input_num,fc1_n_c) 
			<< fully_connected_layer<elu>(fc1_n_c,fc2_n_c) 
			<< fully_connected_layer<elu>(fc2_n_c,fc3_n_c) 
			//<< fully_connected_layer<elu>(fc3_n_c,fc3_n_c) 
			//<< fully_connected_layer<elu>(fc3_n_c,fc3_n_c) 
			//<< fully_connected_layer<elu>(fc3_n_c,fc3_n_c) 
			//<< fully_connected_layer<softmax>(fc3_n_c,output_num);
			<< fully_connected_layer<elu>(fc3_n_c,output_num);
		}

	// Neural Network for Command-Line Breakout
	void self_driving_car_conv_net(network<sequential>& nn, 
						const int& height,
						const int& width, 
  						const int& input_channel_nums, 
  						const int& output_nums){


		assert(height > 0 && width > 0 && input_channel_nums > 0 && output_nums > 0 );

		typedef convolutional_layer<tan_h> conv;
		typedef max_pooling_layer<relu> pool;

		core::backend_t backend_type = core::default_engine();

		//const int height = 4;
	    //const int width = 10;
	    //const int input_channel_nums = 4;
	    //const int output_nums = 4;

		const int kernel_size = 3;					// 5*5 kernel
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
