
#pragma once

#include "Object.h"

class LineObj : public Object
{
public:
	using Object::vertices;
	using Object::vertexbuffer;
	using Object::genVertexBuffer;

	void update(const std::vector<glm::vec3>& vertices_input) {
		vertices.clear();

		for(auto itr : vertices_input) {
			vertices.push_back(itr);
		}

		genVertexBuffer();
	}
};

// end of file
