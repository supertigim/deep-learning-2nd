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

	const int fc1_n_c = input_num ;
	const int fc2_n_c = input_num ;
	const int fc3_n_c = input_num ;

	nn	<< fully_connected_layer<leaky_relu>(input_num,fc1_n_c) 
		<< fully_connected_layer<leaky_relu>(fc1_n_c,fc2_n_c) 
		<< fully_connected_layer<leaky_relu>(fc2_n_c,fc3_n_c) 
		<< fully_connected_layer<leaky_relu>(fc3_n_c,fc3_n_c) 
		<< fully_connected_layer<leaky_relu>(fc3_n_c,fc3_n_c) 
		//<< fully_connected_layer<softmax>(fc3_n_c,num_game_actions_);
		<< fully_connected_layer<leaky_relu>(fc3_n_c,output_num);
	} 

	// Alternative Neural Network for Self Driving Car 
	void self_driving_car_alt1_net(network<sequential>& nn, 
								const int& input_num,
		  						const int& output_num){

    //const int input_nums_ = 13;
    //const int output_nums_ = 3;

	const int fc1_n_c = input_num ;
	const int fc2_n_c = input_num ;
	const int fc3_n_c = input_num ;

	nn	<< fully_connected_layer<tan_h>(input_num,fc1_n_c) 
		<< fully_connected_layer<tan_h>(fc1_n_c,fc2_n_c) 
		<< fully_connected_layer<tan_h>(fc2_n_c,fc3_n_c) 
		<< fully_connected_layer<tan_h>(fc3_n_c,output_num);
	} 

	// Neural Network for Self Driving Car 
	void self_driving_car_alt2_net(network<sequential>& nn, 
								const int& input_num,
		  						const int& output_num){

    //const int input_nums_ = 13;
    //const int output_nums_ = 3;

	const int fc1_n_c = input_num ;
	const int fc2_n_c = input_num ;
	const int fc3_n_c = input_num ;

	nn	<< fully_connected_layer<relu>(input_num,fc1_n_c) 
		<< fully_connected_layer<relu>(fc1_n_c,fc2_n_c) 
		<< fully_connected_layer<relu>(fc2_n_c,fc3_n_c) 
		<< fully_connected_layer<relu>(fc3_n_c,fc3_n_c) 
		<< fully_connected_layer<relu>(fc3_n_c,fc3_n_c) 
		//<< fully_connected_layer<softmax>(fc3_n_c,num_game_actions_);
		<< fully_connected_layer<relu>(fc3_n_c,output_num);
	}     

}  // namespace TDNN_Models


// end of file  
