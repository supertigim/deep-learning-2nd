
#pragma once

#include "Object.h"

class LineObj : public Object
{
public:
	using Object::vertices;
	using Object::vertexbuffer;
	using Object::genVertexBuffer;

	void update(const std::vector<glm::vec3>& vertices_input) {
		showOff();
		
		for(auto itr : vertices_input) {
			vertices.push_back(itr);
		}
		genVertexBuffer();
	}

	void showOff(){
		if (vertices.size() > 0){
			glDeleteBuffers(1, &vertexbuffer);	// free memory !!!!!!!!!!!!!!!!
			vertices.clear();
		}
	}
};

// end of file
