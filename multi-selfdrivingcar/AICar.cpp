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
		, loop_count_(0), loop_count_max_ (30), batch_size_(32), training_threshold_nums_(100), reward_sum_(0.0f), reward_max_(0.0f)
{
	for (int i = sensor_min; i <= sensor_max; i += sensor_di) {
		distances_from_sensors_.push_back(i);
	}
	
	// add Speed, Direction, position as additional inputs
	NETWORK_INPUT_NUM = distances_from_sensors_.size() + 4;	
	INPUT_FRAME_CNT = 1;

	replay_.init(NETWORK_INPUT_NUM, INPUT_FRAME_CNT);
}

void AICar::getStateBuffer(vec_t& t){
	assert(t.size() == NETWORK_INPUT_NUM );
	std::copy(distances_from_sensors_.begin(), distances_from_sensors_.end(), t.begin());
	int count = distances_from_sensors_.size();

	t[count++] = getSpeed(); 
	t[count++] = direction_degree_/360.0f;	// 넣을 필요 있는지 모르겠음 (범위: 0~1)
	t[count++] = center_.x;
	t[count++] = center_.y;
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
		passed_pos_ = center_;

		direction_degree_ = uniform_rand(0.0f,359.9f);
		setDirection(direction_degree_);

		glm::vec3 col_line_center;
		SelfDrivingWorld& world = SelfDrivingWorld::get();
		// 시작하자마자 거리가 좁혀져 있을 경우 
		if(!isTerminated())	break;
	}

	passed_pos_obj_list_.clear();
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

void AICar::updateAll() {
	vel_ *= (1.0f - fric);

	model_matrix_ = glm::translate(vel_) * model_matrix_;

	previous_pos_ = center_;
	center_ += vel_; 
	updateSensor();

	//const int skidmark_num = 5;
	//createSkidMark(skidmark_num);
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
		end_pt = model_matrix_ * end_pt * (float)(1 - std::abs(i) * 0.80 / sensor_max);
		//end_pt = model_matrix_ * end_pt;
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

		for (int o = 0; o < passed_pos_obj_list_.size(); o++) {
			int flag_temp;
			float t_temp;
			glm::vec3 col_pt_temp;

			passed_pos_obj_list_[o]->checkCollisionLoop(center, r, flag_temp, t_temp, col_pt_temp);

			if (flag_temp == 1 && t_temp < min_t) {
				min_t = t_temp;
				col_pt = col_pt_temp;
				flag = flag_temp;
			}
		}
		//#include <glm/gtx/string_cast.hpp>
		//std::cout<< i << "' vec4:" << glm::to_string(col_pt) << endl;
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

void AICar::createSkidMark(const int& nums){

	if( glm::distance(passed_pos_, center_ ) > car_length_* 2.9f  && getSpeed() > 0){
		Object *temp = new Object;
		temp->initCircle(passed_pos_, 0.05f, 6);
		passed_pos_obj_list_.push_back(std::move(std::unique_ptr<Object>(temp))); 

		if( passed_pos_obj_list_.size() > nums) {
			passed_pos_obj_list_.pop_front();
		}
		passed_pos_ = center_;
	}
}

void AICar::calculateRewardAndcheckCollision(float& reward, int& is_terminated){
	/*
	is_terminated = 0;
	reward = 0.0f;
	float sensor_reward = 0.0f;
	
	if(getSpeed() <= 0.0f) reward = -1.0;
	else{
		reward = getSpeed();
		//int cnt;
		//for (int i = sensor_min; i <= sensor_max; i += sensor_di, ++cnt) {
		//	if(-40 < i && i < 40){
		//		sensor_reward += distances_from_sensors_[cnt]/sensing_radius * 0.5;
		//	}
		//}
		//sensor_reward = sensor_reward/cnt;
		//sensor_reward = sensor_reward == 1? 0.5f: 0.0f;

		//reward = getSpeed() * 0.5 
		//		+ sensor_reward * 0.5;
		//reward = std::min(1.0f , reward);
	}
	*/
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
	
	
	if (isTerminated()) {
		reward = -2.0f;	// punishment!!
		is_terminated = 1;	// terminal 
		initialize();
	}
}

bool AICar::isTerminated(){

	bool ret = false;
	glm::vec3 col_line_center;
	SelfDrivingWorld& world = SelfDrivingWorld::get();

	ret |= checkCollisionLoop(world.getObjects(), col_line_center); // 장애물이나 벽에 부딪힐 때...
	ret |= checkCollisionLoop(passed_pos_obj_list_, col_line_center);// 스키드마크에 충돌

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

//void AICar::makeImage(vec_t& t){	
void AICar::makeImage(){	// CNN으로 입력 받기 위해 이미지 작업 하다 냅둔 것

	for (int i = sensor_min, count = 0; i <= sensor_max; i += sensor_di, ++count) {
		//float r = 36.0f * distances_from_sensors_[count]; 
		const float r = distances_from_sensors_[count]; 
		//const float x = r*cos((float)i);
		//const float y = r*sin((float)i);
		//std::cout<< i << "' x:" << x << " y:" << y << endl;
		std::cout<< i << "' r:" << r << endl;
	}
	std::cout << endl ;
}

//const int skip_frame_rate = 3;		// 테스트로 넣어봄~ 프레임이 너무 많은것 같아서~ 
//int skip_count = 0;
void AICar::drive(){	
	SelfDrivingWorld& world = SelfDrivingWorld::get();
	DQN& dqn = world.dqn();

	Transition transition(AICar::NETWORK_INPUT_NUM, 0, 0.0f,0);
	vec_t& input_to_replay = std::get<0>(transition);

	//if(world.is_training() & ID_)
	//	makeImage();
	getStateBuffer(input_to_replay);

	//if(skip_count == skip_frame_rate)
		past_states_.push_back(input_to_replay);

	label_t& action = std::get<1>(transition);
	if (past_states_.size() < AICar::INPUT_FRAME_CNT) {
		action = 0; // stay because it doesn't have enough past states to make input to the net
	}
	else {
		if (past_states_.size() > AICar::INPUT_FRAME_CNT) {
			past_states_.pop_front();
		}

		vec_t input_to_nn(AICar::NETWORK_INPUT_NUM * AICar::INPUT_FRAME_CNT);
		for(int i = 0, count = 0; i < AICar::INPUT_FRAME_CNT ; ++i, count += AICar::NETWORK_INPUT_NUM){
			std::copy(past_states_[i].begin(), past_states_[i].end(), input_to_nn.begin() + count);	
		}
		action = (world.is_training() & ID_)? dqn.selectAction(input_to_nn): dqn.selectAction(input_to_nn, true);
		//if( loop_count_ > loop_count_max_ )
			//dqn.printQValues(input_to_nn);
	}
	processInput(action);					

	float& reward = std::get<2>(transition);
	int& isTerminated = std::get<3>(transition);	// 0 : continue, 1 : terminate

	updateAll();
	calculateRewardAndcheckCollision(reward,isTerminated);

	if(world.is_training() &ID_){
		//if(skip_count == skip_frame_rate)
			replay_.push_back(transition); // store transition 	
		
		reward_sum_ += getSpeed(); // represent the distance to travel

		if(isTerminated) {
			
			if (reward_max_ < reward_sum_) {
				reward_max_ = reward_sum_;
				//std::cout 	<< "**************************" << endl
				//				<< "**[" << ID_ << "] New Record : " << reward_max_ << " **" << endl
				//				<< "**************************" << endl;
			}
			reward_sum_ = 0.0f;
		}
		if(replay_.size() > training_threshold_nums_) {
			if( loop_count_ > loop_count_max_ ){
				dqn.update(replay_, batch_size_);
				loop_count_ = 0;
			}
		}
	}
	//if(skip_count == skip_frame_rate)	skip_count = 0;
	//else 								++skip_count;
	++loop_count_;
}

void AICar::render(const GLint& MatrixID, const glm::mat4 vp){
	drawLineLoop(MatrixID, vp);
	sensing_lines.drawLineLoop(MatrixID, vp);
	for(int i = 0 ; i <  passed_pos_obj_list_.size(); ++i ) {
		passed_pos_obj_list_[i]->drawLineLoop(MatrixID, vp);
	}
}

// end of file 
