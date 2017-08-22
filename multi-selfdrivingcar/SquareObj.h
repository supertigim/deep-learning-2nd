
#pragma once

//#include "common.h"
#include "Object.h"

/**
 *   Square Object Class 
 */
class SquareObj : public Object
{
public:
	using Object::vertices;
	using Object::vertexbuffer;
	using Object::genVertexBuffer;

	SquareObj() {}
	SquareObj(const glm::vec3& center, const float& half_dx, const float& half_dy);

	void update(const glm::vec3& center, const float& half_dx, const float& half_dy);
	bool isInside(const glm::vec3& pt);
protected:	
	float x_min_, y_min_,x_max_,y_max_;
};

// end of file 
