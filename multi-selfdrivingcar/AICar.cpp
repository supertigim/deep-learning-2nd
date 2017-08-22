#include "AICar.h"
#include <memory>
#include <glm/gtx/transform.hpp>
#include <math.h>
#include "SelfDrivingWorld.h"

float AICar::NETWORK_INPUT_NUM;//	= 0.0f;
float AICar::INPUT_FRAME_CNT;//	= 0.0f;

AICar::AICar(int id) 
		: ID_(id), car_length_(0.03f), fric(0.01f), turn_coeff_(2.0f), accel_coeff_(0.0001f), brake_coeff_(0.0003f)
		, sensing_radius(0.25f), sensor_min(-170), sensor_max(180), sensor_di(10)//, keep_turning_(0.0f)
		, batch_size_(2)
{
	for (int i = sensor_min; i <= sensor_max; i += sensor_di) {
		distances_from_sensors_.push_back(i);
	}
	
	// add Speed, Direction, position as additional inputs
	NETWORK_INPUT_NUM = distances_from_sensors_.size()+ 1;	
	INPUT_FRAME_CNT = 1;

	replay_.init(NETWORK_INPUT_NUM, INPUT_FRAME_CNT);
}

void AICar::updateStateVector(){
	vec_t t(NETWORK_INPUT_NUM);

	
	std::copy(distances_from_sensors_.begin(), distances_from_sensors_.end(), t.begin());
	int count = distances_from_sensors_.size();
	 t[count++] = getSpeed(); 
	//t[count++] = direction_degree_/360.0f;	// 넣을 필요 있는지 모르겠음 (범위: 0~1)
	//t[count++] = center_.x;
	//t[count++] = center_.y;	

	past_states_.push_back(t);
	if (past_states_.size() > AICar::INPUT_FRAME_CNT) {
		past_states_.pop_front();
	}
}

bool AICar::getState(vec_t& t){
	if (past_states_.size() < AICar::INPUT_FRAME_CNT) { 
		t.clear();
		return false; 
	}

	for(int i = 0, count = 0; i < AICar::INPUT_FRAME_CNT ; ++i, count += AICar::NETWORK_INPUT_NUM)
		std::copy(past_states_[i].begin(), past_states_[i].end(), t.begin() + count);	
	return true;
}

// reset car status until new position is good enough to do new episode 
void AICar::initialize() {
	const float	world_center_x = 0.5f,
				world_center_y = 0.5,
				world_radius = 1.2f;
	float x,y;
	while (true){
		x = uniform_rand(world_center_x - (world_radius - car_length_)
							, world_center_x + (world_radius - car_length_));
		y = uniform_rand(world_center_y - (world_radius - car_length_)
							, world_center_y + (world_radius - car_length_));

		//if( world_radius < std::sqrt((x-world_center_x)*(x-world_center_x)
		//							 + (y-world_center_y)*(y-world_center_y)) ||
		//	0.8f > std::sqrt((x-world_center_x)*(x-world_center_x)
		//							 + (y-world_center_y)*(y-world_center_y))
		//	)
		if( (x > world_center_x - 0.75f && x < world_center_x + 0.75f) || 
			(y > world_center_y - 0.75f && y < world_center_y + 0.75f) )
		{
			if( (x > world_center_x - 0.05f && x < world_center_x + 0.05f) || 
			(y > world_center_y - 0.05f && y < world_center_y + 0.05f) )
			{} 
			else 
				continue;
		}

		update(glm::vec3(x, y, 0.0f), car_length_, car_length_);
		previous_pos_ = center_;

		direction_degree_ = uniform_rand(0.0f,359.9f);
		setDirection(direction_degree_);

		glm::vec3 col_line_center;
		SelfDrivingWorld& world = SelfDrivingWorld::get();
		// 시작하자마자 거리가 좁혀져 있을 경우 
		if(!isTerminated())	break;
	}

}

void AICar::setDirection(float dir){

	model_matrix_ = glm::mat4();
	dir_ = glm::vec3(01.0f, 0.0f, 0.0f);
	vel_ = glm::vec3(0.002f, 0.0f, 0.0f);

	const glm::mat4 rot_mat = glm::rotate(glm::mat4(), glm::radians(dir), glm::vec3(0, 0, 1));

	glm::vec4 temp(dir_.x, dir_.y, dir_.z, 0.0f);

	temp = rot_mat * temp;

	dir_.x = temp.x;
	dir_.y = temp.y;

	rotateCenteredZAxis(dir);
	vel_ = dir_ * glm::sqrt(glm::dot(vel_, vel_));
}

void AICar::turnLeft() {
	const glm::mat4 rot_mat = glm::rotate(glm::mat4(), glm::radians(turn_coeff_), glm::vec3(0, 0, 1));
	glm::vec4 temp(dir_.x, dir_.y, dir_.z, 0.0f);
	temp = rot_mat * temp;

	dir_.x = temp.x;
	dir_.y = temp.y;

	rotateCenteredZAxis(turn_coeff_);
	direction_degree_ += turn_coeff_;
	if(direction_degree_ > 360.0f) direction_degree_ -= 360.0f;

	float x = (glm::dot(vel_, dir_) < 0.0)? -1.0: 1.0;
	vel_ = dir_ * glm::sqrt(glm::dot(vel_, vel_)) * x;
}

void AICar::turnRight() {
	const glm::mat4 rot_mat = glm::rotate(glm::mat4(), glm::radians(-turn_coeff_), glm::vec3(0, 0, 1));
	glm::vec4 temp(dir_.x, dir_.y, dir_.z, 0.0f);
	temp = rot_mat * temp;

	dir_.x = temp.x;
	dir_.y = temp.y;

	rotateCenteredZAxis(-turn_coeff_);
	direction_degree_ -= turn_coeff_;
	if(direction_degree_ < 0) direction_degree_ = 360.0f - direction_degree_;

	float x = (glm::dot(vel_, dir_) < 0.0)? -1.0: 1.0;
	vel_ = dir_ * glm::sqrt(glm::dot(vel_, vel_)) * x;
}

void AICar::accel() {
	vel_ += accel_coeff_ * dir_;
}

void AICar::decel() {
	if (glm::dot(vel_ - brake_coeff_ * dir_, dir_) > 0.0f )	vel_ -= brake_coeff_ * dir_;	// for fast stop
	else													vel_ -= accel_coeff_ * dir_;	// normally move backward like forward-move
}


void AICar::updateAll(label_t& action, float& reward, vec_t& t) {
	processInput(action);

	vel_ *= (1.0f - fric);	// natural speed reduce 
	model_matrix_ = glm::translate(vel_) * model_matrix_;
	previous_pos_ = center_;
	center_ += vel_;

	updateSensor();
	updateStateVector();

	calculateReward(reward);
	
	if (isTerminated()) {
		reward = -1.0f;		// punishment!!
		initialize();
		t.clear();
	} else {
		getState(t);
	}
}

void AICar::updateSensor()
{
	SelfDrivingWorld& world = SelfDrivingWorld::get();
	std::vector<glm::vec3> sensor_lines;
	const glm::vec3 center = center_;
	const float radius = sensing_radius;

	int count = 0;
	for (int i = sensor_min; i <= sensor_max; i += sensor_di, ++count) {
		glm::vec4 end_pt = glm::vec4(radius*cos(glm::radians((float)i)), radius*-sin(glm::radians((float)i)), 0.0f, 0.0f);

		// reduce detecting area of beside sensors by 1/2 
		//end_pt = model_matrix_ * end_pt * (float)(1 - std::abs(i) * 0.80 / sensor_max);
		//end_pt = model_matrix_ * end_pt;
		end_pt = model_matrix_ * end_pt * ( 0.7f + std::abs(getSpeed()) * 0.7f );
		
		const glm::vec3 r = center + glm::vec3(end_pt.x, end_pt.y, end_pt.z);

		int flag = 0;
		glm::vec3 col_pt;
		float min_t = 1e8;
	
		for (int o = 0; o < world.getObjects().size(); o++) {
			int flag_temp;
			float t_temp;
			glm::vec3 col_pt_temp;

			world.getObjects()[o]->checkCollisionLoop(center, r, flag_temp, t_temp, col_pt_temp);

			if (flag_temp == 1 && t_temp < min_t) {
				min_t = t_temp;
				col_pt = col_pt_temp;
				flag = flag_temp;
			}
		}
		
		for (int o = 0; o < world.getCars().size(); ++o) {
			if(ID_ & 1 << o)	continue;

			int flag_temp;
			float t_temp;
			glm::vec3 col_pt_temp;

			world.getCars()[o]->checkCollisionLoop(center, r, flag_temp, t_temp, col_pt_temp);
			
			if (flag_temp == 1 && t_temp < min_t) {
				min_t = t_temp;
				col_pt = col_pt_temp;
				flag = flag_temp;
			}
		}

		if (flag == 1) {
			sensor_lines.push_back(center);
			sensor_lines.push_back(col_pt);

			// scale value between 0.0 ~ 1.0f 
			distances_from_sensors_[count] = glm::distance(col_pt, center) / glm::distance(r, center);
			//distances_from_sensors_[count] = glm::distance(col_pt, center) / sensing_radius;
		}
		else {
			sensor_lines.push_back(center);
			sensor_lines.push_back(r);

			distances_from_sensors_[count] = 1.0f;
			//distances_from_sensors_[count] = glm::distance(r, center) / sensing_radius;
		}
	}
	if(world.is_training() & ID_)	sensing_lines.update(sensor_lines);
	else							sensing_lines.showOff();
}

void AICar::calculateReward(float& reward){
	
	float head_sensor_stat = 1.0f;
	float collision_warning = 1.0f;
	for (int i = sensor_min, count = 0; i <= sensor_max; i += sensor_di, ++count) {
		if ( -10 <= i && i <= 10){
			head_sensor_stat = std::min (head_sensor_stat, distances_from_sensors_[count]);
		}	
		collision_warning = std::min (collision_warning, distances_from_sensors_[count]);
	}

	// 속도가 빠를 수록, 정면 센서에 장애물이 없을 수록 보상 증가 
	if(collision_warning <= 0.5f)	reward += collision_warning - 1.0f;
	reward += getSpeed(); 	// keeping moving is reward
	if(reward > 0) reward += head_sensor_stat/2;				// addition reward if head sersors do not detect any obstacle
	reward = floorf(reward * 10) / 10.0f;
}

bool AICar::isTerminated(){

	bool ret = false;
	glm::vec3 col_line_center;
	SelfDrivingWorld& world = SelfDrivingWorld::get();

	ret |= checkCollisionLoop(world.getObjects(), col_line_center); // 장애물이나 벽에 부딪힐 때...

	for(int i = 0; i < world.getCars().size(); ++i){
		if(ID_ & 1 << i)	continue;
		ret |= checkCollisionLoop(*world.getCars()[i], col_line_center); // 다른 차와 충돌 체크 
	}
	ret |= (getSpeed() < -0.50f);	// to prevent go backward
	return ret;
}

// range -1.0f ~ 1.0f
float AICar::getSpeed(){
	float ret = 110.0f * glm::distance(previous_pos_, center_ );
	//std::cout << "speed: "<< ret << endl;
	if (ret < 0.1f) 				ret = 0.0f;		// 빙글 빙글 도는 상태를 막기 위해
	if (glm::dot(vel_, dir_) < 0.0) ret *= -1.0f;	// 방향 factor 
	if (ret < -1.0f) 				ret = -1.0f;	// 최소 -1.0
	if (ret > 1.0f) 				ret = 1.0f;		// 최대 +1.0
	return ret ;
}

void AICar::processInput(const int& action){

	switch (action)
	{
	//case AICar::ACT_STAY:
	//	// do nothing
	//	break;
	case AICar::ACT_ACCEL:
		accel();
		break;
	case AICar::ACT_LEFT:
		turnLeft(); 
		accel();
		break;
	case AICar::ACT_RIGHT:
		turnRight(); 
		accel();
		break;
	case AICar::ACT_BRAKE:
		decel();
		break;
	default:
		std::cout << "Wrong action " << endl;
		exit(1);
		break;
	}
}

void AICar::drive(){	
	SelfDrivingWorld& world = SelfDrivingWorld::get();
	DQN& dqn = world.dqn();

	std::unique_ptr<Transition> t_ptr = std::make_unique<Transition>(AICar::NETWORK_INPUT_NUM * AICar::INPUT_FRAME_CNT
																	,0
																	,0.0f
																	,AICar::NETWORK_INPUT_NUM * AICar::INPUT_FRAME_CNT
																	,0.0f);
	vec_t& state 		= std::get<0>(*t_ptr);
	label_t& action 	= std::get<1>(*t_ptr);
	float& reward 		= std::get<2>(*t_ptr);
	vec_t& next_state 	= std::get<3>(*t_ptr);
	float& td 			= std::get<4>(*t_ptr);

	if(getState(state))	action = (world.is_training() & ID_)? dqn.selectAction(state): dqn.selectAction(state, true);
	else 				action = ACT_ACCEL;

	// get reward and next_statte
	updateAll(action, reward, next_state);
		
	if(state.size() && world.is_training() & ID_){
		td = reward;
		replay_.addTransition(std::move(t_ptr));

		if (replay_.size() >= AICar::INPUT_FRAME_CNT)

			//std::cout << "reward:" << reward << endl;
			dqn.update(replay_, batch_size_);
	}
}

void AICar::render(const GLint& MatrixID, const glm::mat4 vp){
	drawLineLoop(MatrixID, vp);
	sensing_lines.drawLineLoop(MatrixID, vp);
}

// end of file 
