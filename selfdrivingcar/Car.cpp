#include "Car.h"
#include <memory>
#include <glm/gtx/transform.hpp>
#include <math.h>

void Car::init() {

	const float	world_center_x = 0.5f,
				world_center_y = 0.5,
				world_radius = 1.2f,
				car_length = 0.06f;

	float x,y;
	while (true){
		x = uniform_rand(world_center_x - (world_radius - car_length)
							, world_center_x + (world_radius - car_length));
		y = uniform_rand(world_center_y - (world_radius - car_length)
							, world_center_y + (world_radius - car_length));

		//std::cout << "x: " << x << " y: " << y << endl;
		if( world_radius > std::sqrt((x-world_center_x)*(x-world_center_x)
									 + (y-world_center_y)*(y-world_center_y)) &&
			0.8f < std::sqrt((x-world_center_x)*(x-world_center_x)
									 + (y-world_center_y)*(y-world_center_y))
			)
			break;
	}

	x = 0.6f;
	y = -0.4f;

	body_.update(glm::vec3(x, y, 0.0f), car_length, car_length);

	float dir = uniform_rand(0.0f,359.9f);
	startDirection(dir);

	int count = 0;
	for (int i = sensor_min; i <= sensor_max; i += sensor_di)
		count++;

	distances_from_sensors_.resize(count);
}

void Car::startDirection(float dir){

	body_.model_matrix_ = glm::mat4();
	dir_ = glm::vec3(01.0f, 0.0f, 0.0f);
	vel_ = glm::vec3(0.01f, 0.0f, 0.0f);

	const glm::mat4 rot_mat = glm::rotate(glm::mat4(), glm::radians(dir), glm::vec3(0, 0, 1));

	glm::vec4 temp(dir_.x, dir_.y, dir_.z, 0.0f);

	temp = rot_mat * temp;

	dir_.x = temp.x;
	dir_.y = temp.y;

	body_.rotateCenteredZAxis(dir);
	vel_ = dir_ * glm::sqrt(glm::dot(vel_, vel_));
}

void Car::turnLeft() {
	
	const glm::mat4 rot_mat = glm::rotate(glm::mat4(), glm::radians(turn_coeff_), glm::vec3(0, 0, 1));

	glm::vec4 temp(dir_.x, dir_.y, dir_.z, 0.0f);

	temp = rot_mat * temp;

	dir_.x = temp.x;
	dir_.y = temp.y;

	body_.rotateCenteredZAxis(turn_coeff_);

	F x = 1.0;

	if (glm::dot(vel_, dir_) < 0.0) x = -1.0;
	
	vel_ = dir_ * glm::sqrt(glm::dot(vel_, vel_)) * x;

}

void Car::turnRight() {

	const glm::mat4 rot_mat = glm::rotate(glm::mat4(), glm::radians(-turn_coeff_), glm::vec3(0, 0, 1));

	glm::vec4 temp(dir_.x, dir_.y, dir_.z, 0.0f);

	temp = rot_mat * temp;

	dir_.x = temp.x;
	dir_.y = temp.y;

	body_.rotateCenteredZAxis(-turn_coeff_);

	F x = 1.0;

	if (glm::dot(vel_, dir_) < 0.0) x = -1.0;

	vel_ = dir_ * glm::sqrt(glm::dot(vel_, vel_)) * x;
}

void Car::accel() {
	vel_ += accel_coeff_ * dir_;
}

void Car::decel() {
	vel_ -= accel_coeff_ * dir_;
}

void Car::update() {
	vel_ *= (1.0f - fric);

	body_.model_matrix_ = glm::translate(vel_) * body_.model_matrix_;

	body_.center_ += vel_; //TODO: update model_matrix AND center?
}
	
void Car::updateSensor(const std::vector<std::unique_ptr<Object>>& obj_list, const bool& update_gl_obj)// parameter -> object list
{
	// sensor sensing_lines (distance from car view point)
	std::vector<glm::vec3> sensor_lines;
	const glm::vec3 center = body_.center_;
	const float radius = sensing_radius;

	for (int i = sensor_min, count = 0; i <= sensor_max; i += sensor_di, count ++) {

		glm::vec4 end_pt = glm::vec4(radius*cos(glm::radians((float)i)), radius*-sin(glm::radians((float)i)), 0.0f, 0.0f);

		//end_pt = body_.model_matrix_ * end_pt;
		// reduce detecting area of sensor beside by 1/2 
		end_pt = body_.model_matrix_ * end_pt * (float)(1 - std::abs(i) * 0.5 / sensor_max);

		const glm::vec3 r = center + glm::vec3(end_pt.x, end_pt.y, end_pt.z);

		int flag = 0;
		glm::vec3 col_pt;
		float t;

		// find closest collision pt
		{
			float min_t = 1e8;

			for (int o = 0; o < obj_list.size(); o++) {
				int flag_temp;
				float t_temp;
				glm::vec3 col_pt_temp;

				obj_list[o]->checkCollisionLoop(center, r, flag_temp, t_temp, col_pt_temp);

				if (flag_temp == 1 && t_temp < min_t) {
					t = t_temp;
					min_t = t_temp;
					col_pt = col_pt_temp;
					flag = flag_temp;
				}
			}
		}
		


		if (flag == 1) {
			sensor_lines.push_back(center);
			sensor_lines.push_back(col_pt);

			// scale value.. always less 1.0f 
			distances_from_sensors_[count] = sqrt(glm::dot(col_pt-center, col_pt-center))
											/ sqrt(glm::dot(r-center, r-center));
			//std::cout << "count: " << count << "| "<<distances_from_sensors_[count] << endl;
		}
		else {
			sensor_lines.push_back(center);
			sensor_lines.push_back(r);

			distances_from_sensors_[count] = 1.0f;
		}
	}

	if(update_gl_obj == true)
		sensing_lines.update(sensor_lines);
	//sensing_lines.center_ = car_body.center_;

}

// end of file 
