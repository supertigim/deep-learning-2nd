#pragma once 

#include "SquareObj.h"
#include "LineObj.h"

class Car {

public:
	SquareObj body_;
	LineObj sensing_lines;
	
	glm::vec3 dir_, vel_;

	F turn_coeff_;
	F accel_coeff_;
	F fric;
	F sensing_radius;

	std::vector<F> distances_from_sensors_;

	int sensor_min, sensor_max, sensor_di;

public:
	Car() 
		: turn_coeff_(2.0f), accel_coeff_(0.0001f), fric(0.02f), sensing_radius(0.25f)
		, sensor_min(-90), sensor_max(90), sensor_di(15)
	{}

	void init();
	void startDirection(float dir);
	void turnLeft();
	void turnRight();
	void accel();
	void decel();
	void update();
	void updateSensor(const std::vector<std::unique_ptr<Object>>& obj_list
					, const bool& update_gl_obj = true);
};

// end of file
