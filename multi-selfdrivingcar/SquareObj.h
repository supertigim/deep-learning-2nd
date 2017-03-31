
#pragma once

#include "Common.h"
#include "Object.h"

class SquareObj : public Object
{
public:
	using Object::vertices;
	using Object::vertexbuffer;
	using Object::genVertexBuffer;

	//Box<F> box_;

	//union {
	//	struct {T x_min_, y_min_,x_max_,y_max_;};
	//	struct {T i_start_,j_start_,i_end_,j_end_;};
	//};
	F x_min_, y_min_,x_max_,y_max_;

	SquareObj() {}
	SquareObj(const glm::vec3& center, const F& half_dx, const F& half_dy);

	void update(const glm::vec3& center, const F& half_dx, const F& half_dy);

	bool isInside(const glm::vec3& pt);
};

// end of file 
