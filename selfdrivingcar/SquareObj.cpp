#include "SquareObj.h"

SquareObj::SquareObj(const glm::vec3& center, const F& half_dx, const F& half_dy)
{
	update(center, half_dx, half_dy);
}

void SquareObj::update(const glm::vec3& center, const F& half_dx, const F& half_dy)
{
	center_ = center;

	x_min_ = center.x - half_dx;
	y_min_ = center.y - half_dy;
	x_max_ = center.x + half_dx;
	y_max_ = center.y + half_dy;

	const glm::vec3 v0(center.x - half_dx, center.y - half_dy, center.z);
	const glm::vec3 v1(center.x + half_dx, center.y - half_dy, center.z);
	const glm::vec3 v2(center.x + half_dx, center.y + half_dy, center.z);
	const glm::vec3 v3(center.x - half_dx, center.y + half_dy, center.z);

	vertices.clear();
	vertices.push_back(v0);
	vertices.push_back(v1);
	vertices.push_back(v2);
	vertices.push_back(v3);

	genVertexBuffer();
}

bool SquareObj::isInside(const glm::vec3& pt)
{
	const glm::mat4 inv_m = glm::inverse(model_matrix_);
	const glm::vec4 pt4(pt.x, pt.y, pt.z, 1.0f);
	const glm::vec4 pt4_inv = inv_m * pt4;

	//return box_.isInside(pt4_inv.x, pt4_inv.y);
	if(pt4_inv.x < x_min_) return false;
	else if(pt4_inv.x > x_max_) return false;

	if(pt4_inv.y < y_min_) return false;
	else if(pt4_inv.y > y_max_) return false;

	return true;
}

// end of file
