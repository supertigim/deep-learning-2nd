#pragma once 

#include <string>
#include "tiny_dnn/tiny_dnn.h"

class AutoCarNet : public tiny_dnn::network<tiny_dnn::sequential> {
 public:
  explicit AutoCarNet(	const std::string &name = ""
  						,const int& input_dim = 0
		  				,const int& output_dim = 0
		  				);
};

// End of file  
