#include "AutoCarNet.h"

AutoCarNet::AutoCarNet( const std::string &name, 
                                const int& input_dim,
                                const int& output_dim
    ) : tiny_dnn::network<tiny_dnn::sequential>(name) {

    using conv_l        = tiny_dnn::layers::conv;
    using max_pool_l    = tiny_dnn::layers::max_pool;
    using input_l       = tiny_dnn::layers::input;
    using fc_l          = tiny_dnn::layers::fc;

    using relu_a     = tiny_dnn::activation::relu;
    using elu_a     = tiny_dnn::activation::elu;

    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

    assert(input_dim > 0 && output_dim > 0);

    const int fc1_n_c = (input_dim+output_dim);///3+1;
    const int fc2_n_c = (input_dim+output_dim)/3+1;
    const int fc3_n_c = (input_dim+output_dim)/3+1;

    *this   << input_l(input_dim)
            << fc_l(input_dim,fc1_n_c,    true, backend_type) << relu_a()
            << fc_l(fc1_n_c,  fc2_n_c,    true, backend_type) << relu_a()
            << fc_l(fc2_n_c,  output_dim, true, backend_type) << relu_a()
    ;
}

// end of file 
