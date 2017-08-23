#include "BoNet.h"

using namespace tiny_dnn;

BoNet::BoNet(const std::string &name
            ,const int& height
            ,const int& width 
            ,const int& input_channel
            ,const int& output_dim
    ) : tiny_dnn::network<tiny_dnn::sequential>(name) {

    assert(height > 0 && width > 0 && input_channel > 0 && output_dim > 0 );
    
    using conv_l        = tiny_dnn::layers::conv;
    using max_pool_l    = tiny_dnn::layers::max_pool;
    using fc_l          = tiny_dnn::layers::fc;

    using relu_a        = tiny_dnn::activation::relu;
    using tanh_a        = tiny_dnn::activation::tanh;

    const int kernel_size   = 5;        // first kernel size (3*3 5*5 or 1*1)
    const int kernel_size_2 = 3;        // second kernel size (3*3 5*5 or 1*1)
    const int n_fmaps       = 32;//height;   //< number of feature maps for upper layer
    const int n_fmaps2      = height*2; //< number of feature maps for lower layer
    const int n_fc          = height*2; //< number of hidden units in fully-connected layer

    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

    *this   << conv_l(height, width, kernel_size, input_channel, n_fmaps, padding::same, true, 1, 1, backend_type)
            << max_pool_l(height, width, n_fmaps, 2)
            << relu_a()
            << conv_l(height/2, width/2, kernel_size_2, n_fmaps, n_fmaps2, padding::same, true, 1, 1, backend_type)
            << max_pool_l(height/2, width/2, n_fmaps2, 2)
            << relu_a()
            << fc_l(height/4 * width/4 * n_fmaps2, n_fc, true, backend_type)
            << relu_a()
            << fc_l(n_fc, output_dim, true, backend_type)
            << tanh_a()
    ;

}

// end of file 
