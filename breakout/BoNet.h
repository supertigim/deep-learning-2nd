#pragma once 

#include <string>
#include "tiny_dnn/tiny_dnn.h"

class BoNet : public tiny_dnn::network<tiny_dnn::sequential> {
 public:
  explicit BoNet(const std::string &name = ""
  				,const int& height = 0
				,const int& width = 0
  				,const int& input_channel = 0
		  		,const int& output_dim = 0
		  		);
};

// End of file  
