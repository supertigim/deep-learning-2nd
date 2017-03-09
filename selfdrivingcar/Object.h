#pragma once

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <vector>

class Object
{
public:
	glm::vec3 center_;
	glm::mat4 model_matrix_;

	//std::vector<float> vertices;
	std::vector<glm::vec3> vertices;
	GLuint vertexbuffer;

public:
	Object() 
		:model_matrix_(glm::mat4(1.0f))
	{}

	~Object();
	glm::vec3 getTransformed(const glm::vec3 v) const;
	void genVertexBuffer();

	void drawLineLoop(const GLint& MatrixID, const glm::mat4 vp);
	void drawLines(const GLint& MatrixID, const glm::mat4 vp);

	void rotateCenteredZAxis(const float& angle_degree);

	bool checkCollisionLoop(const Object& obj2, glm::vec3& col_obj_center);
	bool checkCollisionLoop(const std::vector<std::unique_ptr<Object>>& obj_list, glm::vec3& col_obj_center);
	void checkCollisionLoop(const glm::vec3& ray_start, const glm::vec3& ray_end, int& flag, float& t, glm::vec3& col_pt) const;
	void initCircle(const glm::vec3& center, const float& radius, const int num_segments);
};

// end of file 
