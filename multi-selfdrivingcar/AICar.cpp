#include "AICar.h"
#include <memory>
#include <glm/gtx/transform.hpp>
#include <math.h>
#include "SelfDrivingWorld.h"

float AICar::NETWORK_INPUT_NUM	= 0.0f;
float AICar::INPUT_FRAME_CNT	= 0.0f;

AICar::AICar(int id) 
		: ID_(id), car_length_(0.03f), fric(0.01f), turn_coeff_(2.0f), accel_coeff_(0.0001f), brake_coeff_(0.0002f)
		, sensing_radius(0.20f), sensor_min(-180), sensor_max(180), sensor_di(15)
		, loop_count_(0), loop_count_max_ (30), batch_size_(10), training_threshold_nums_(100), reward_sum_(0.0f), reward_max_(0.0f)
{
	int count = 0;
	for (int i = sensor_min; i <= sensor_max; i += sensor_di)
		count++;
	distances_from_sensors_.resize(count);

	NETWORK_INPUT_NUM = count + 2;	// add Speed as an input
	INPUT_FRAME_CNT = 1;

	setTrainingAlgorithm();
}

void AICar::setTrainingAlgorithm(){
	SelfDrivingWorld& world = SelfDrivingWorld::get();
	dqn_ = std::make_unique<DQN>(world.getGlobalNetwork(),
								NETWORK_INPUT_NUM,
								INPUT_FRAME_CNT, 
								ACT_MAX);
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

		//std::cout << "x: " << x << " y: " << y << endl;
		if( world_radius < std::sqrt((x-world_center_x)*(x-world_center_x)
									 + (y-world_center_y)*(y-world_center_y)) ||
			0.8f > std::sqrt((x-world_center_x)*(x-world_center_x)
									 + (y-world_center_y)*(y-world_center_y))
			)
			continue;

		update(glm::vec3(x, y, 0.0f), car_length_, car_length_);
		previous_pos_ = center_;
		passed_pos_ = center_;

		direction_degree_ = uniform_rand(0.0f,359.9f);
		setDirection(direction_degree_);

		glm::vec3 col_line_center;
		SelfDrivingWorld& world = SelfDrivingWorld::get();
		//updateSensor(world.getObjects(), !world.is_training()); // 시작하자마자 거리가 좁혀져 있을 경우 
		if (checkCollisionLoop(world.getObjects(), col_line_center) == false)
			break;
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

static float keep_left = 0.0f;
void AICar::turnLeft() {

	if( getSpeed() >= -0.1f && getSpeed() <= 0.1f)	keep_left += turn_coeff_; 
	else 											keep_left = 0.0f;

	const glm::mat4 rot_mat = glm::rotate(glm::mat4(), glm::radians(turn_coeff_), glm::vec3(0, 0, 1));
	glm::vec4 temp(dir_.x, dir_.y, dir_.z, 0.0f);
	temp = rot_mat * temp;

	dir_.x = temp.x;
	dir_.y = temp.y;

	rotateCenteredZAxis(turn_coeff_);
	direction_degree_ += turn_coeff_;
	if(direction_degree_ > 360.0f) direction_degree_ -= 360.0f;

	F x = 1.0;
	if (glm::dot(vel_, dir_) < 0.0) x = -1.0;
	vel_ = dir_ * glm::sqrt(glm::dot(vel_, vel_)) * x;
}

static float keep_right = 0.0f;
void AICar::turnRight() {

	if( getSpeed() >= -0.1f && getSpeed() <= 0.1f)	keep_right += turn_coeff_; 
	else							 				keep_right = 0.0f;

	const glm::mat4 rot_mat = glm::rotate(glm::mat4(), glm::radians(-turn_coeff_), glm::vec3(0, 0, 1));
	glm::vec4 temp(dir_.x, dir_.y, dir_.z, 0.0f);
	temp = rot_mat * temp;

	dir_.x = temp.x;
	dir_.y = temp.y;

	rotateCenteredZAxis(-turn_coeff_);
	direction_degree_ -= turn_coeff_;
	if(direction_degree_ < 0) direction_degree_ = 360.0f - direction_degree_;

	F x = 1.0;
	if (glm::dot(vel_, dir_) < 0.0) x = -1.0;
	vel_ = dir_ * glm::sqrt(glm::dot(vel_, vel_)) * x;
}

void AICar::accel() {
	vel_ += accel_coeff_ * dir_;
}

void AICar::decel() {
	if (glm::dot(vel_ - brake_coeff_ * dir_, dir_) > 0.0f )
		vel_ -= brake_coeff_ * dir_;	// for fast stop
	else
		vel_ -= accel_coeff_ * dir_;	// normally move backward like forward-move
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
	// sensor sensing_lines (distance from car view point)
	std::vector<glm::vec3> sensor_lines;
	const glm::vec3 center = center_;
	const float radius = sensing_radius;

	for (int i = sensor_min, count = 0; i <= sensor_max; i += sensor_di, count ++) {

		glm::vec4 end_pt = glm::vec4(radius*cos(glm::radians((float)i)), radius*-sin(glm::radians((float)i)), 0.0f, 0.0f);

		// reduce detecting area of beside sensors by 1/2 
		end_pt = model_matrix_ * end_pt * (float)(1 - std::abs(i) * 0.80 / sensor_max);
		//end_pt = model_matrix_ * end_pt;
		const glm::vec3 r = center + glm::vec3(end_pt.x, end_pt.y, end_pt.z);

		int flag = 0;
		glm::vec3 col_pt;
		float min_t = 1e8;
	
		// find closest collision pt
		for (int o = 0; o < world.getObjects().size(); o++) {
			int flag_temp;
			float t_temp;
			glm::vec3 col_pt_temp;

			world.getObjects()[o]->checkCollisionLoop(center, r, flag_temp, t_temp, col_pt_temp);

			if (flag_temp == 1 && t_temp < min_t) {
				//t = t_temp;
				min_t = t_temp;
				col_pt = col_pt_temp;
				flag = flag_temp;
			}
		}
		
		for (int o = 0; o < world.getCars().size(); ++o) {
			if(ID_ == o)	continue;

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
				//t = t_temp;
				min_t = t_temp;
				col_pt = col_pt_temp;
				flag = flag_temp;
			}
		}

		if (flag == 1) {
			sensor_lines.push_back(center);
			sensor_lines.push_back(col_pt);

			//distances_from_sensors_[count] = 10.0f * sqrt(glm::dot(col_pt-center, col_pt-center));
			// scale value.. always less 1.0f 
			distances_from_sensors_[count] = 10.0f * sqrt(glm::dot(col_pt-center, col_pt-center))
										/ sqrt(glm::dot(r-center, r-center));
		}
		else {
			sensor_lines.push_back(center);
			sensor_lines.push_back(r);

			//distances_from_sensors_[count] = 10.0f * sqrt(glm::dot(r-center, r-center));
			distances_from_sensors_[count] = 1.0f;
		}
	}

	if(!world.is_training())
		sensing_lines.update(sensor_lines);
}


void AICar::createSkidMark(const int& nums){

	//if( glm::distance(passed_pos_, body_.center_ ) > car_length_* 2.9f  && getSpeed() > 0){
	if( glm::distance(passed_pos_, center_ ) > car_length_* 2.9f  && getSpeed() > 0){

		Object *temp = new Object;
		temp->initCircle(passed_pos_, 0.05f, 6);
		passed_pos_obj_list_.push_back(std::move(std::unique_ptr<Object>(temp))); 

		if( passed_pos_obj_list_.size() > nums) {
			passed_pos_obj_list_.pop_front();
		}
		//passed_pos_ = body_.center_;
		passed_pos_ = center_;
	}
}

void AICar::calculateRewardAndcheckCollision(float& reward, int& is_terminated){
	is_terminated = 0;
	//reward = (getSpeed() == 0) ? -1.0f: getSpeed();
	reward = getSpeed();

	//if(getSpeed() > 0.6f) 		reward = 1.0f;
	//else if( getSpeed() > 0.3f) 	reward = 0.0f;
	//else 							reward = -0.8f;

	int rayNums = distances_from_sensors_.size();
	float sensorReward = 0.0f; 
	float discount = 1.0f;

	for( int i = 0 ; i < rayNums ; ++i) {
		sensorReward += distances_from_sensors_[i];
	}
	sensorReward /= (rayNums * discount);
	//std::cout << "Sensor Reward: " << sensorReward << endl; 

	reward =  reward 	  * 0.90f 
			+ sensorReward * 0.10f;

	// collision check
	glm::vec3 col_line_center;
	SelfDrivingWorld& world = SelfDrivingWorld::get();

	if (isTerminated() || keep_right > 180.0f || keep_left > 180.0f) {
		reward = -1.0f;		// no reward
		is_terminated = 1;	// terminal 
		initialize();
		keep_right = 0.0f;
		keep_left = 0.0f;
	}
}

bool AICar::isTerminated(){

	bool ret = false;
	// collision check
	glm::vec3 col_line_center;
	SelfDrivingWorld& world = SelfDrivingWorld::get();

	ret |= checkCollisionLoop(world.getObjects(), col_line_center);
	ret |= checkCollisionLoop(passed_pos_obj_list_, col_line_center);

	for(int i = 0; i < world.getCars().size(); ++i){
		if(ID_ == i)	continue;
		ret |= checkCollisionLoop(*world.getCars()[i], col_line_center);
	}

	return ret;
}

// range -1.0f ~ 1.0f
float AICar::getSpeed(){
	//float ret = 110.0f * glm::distance(previous_pos_, body_.center_ );
	float ret = 110.0f * glm::distance(previous_pos_, center_ );
	if (ret < 0.1f) ret = 0.0f;
	if (glm::dot(vel_, dir_) < 0.0) ret *= -1.0f;
	if (ret < -1.0f) ret = -1.0f;
	if (ret > 1.0f) ret = 1.0f;
	//std::cout << "Speed: " << ret << endl;
	return ret ;
}

void AICar::render(const GLint& MatrixID, const glm::mat4 vp){
	// draw
	drawLineLoop(MatrixID, vp);
	sensing_lines.drawLineLoop(MatrixID, vp);

	for(int i = 0 ; i <  passed_pos_obj_list_.size(); ++i ) {
		passed_pos_obj_list_[i]->drawLineLoop(MatrixID, vp);
	}
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
		break;
	case AICar::ACT_RIGHT:
		turnRight(); 
		break;
	case AICar::ACT_BRAKE:
		decel();
		break;
	default:
		std::cout << "Wrong action " << endl;
		exit(1);
		break;
	}
	//accel();
}

void AICar::getStateBuffer(vec_t& t){
	assert(t.size() == NETWORK_INPUT_NUM );
	std::copy(distances_from_sensors_.begin(), distances_from_sensors_.end(), t.begin());
	int count = distances_from_sensors_.size();

	t[count++] = getSpeed(); //std::cout << t[distances_from_sensors_.size()] << endl;
	t[count++] = direction_degree_/360.0f;
}


void AICar::drive(){	
	SelfDrivingWorld& world = SelfDrivingWorld::get();

	Transition transition(AICar::NETWORK_INPUT_NUM, 0, 0.0f,0);
	vec_t& input_to_replay = std::get<0>(transition);

	getStateBuffer(input_to_replay);
	past_states_.push_back(input_to_replay);

	//for( int i = 0 ; i < input_to_replay.size() ; ++i){
	//	std::cout << "input: "<<input_to_replay[i] << "|";
	//}

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
		
		action = (world.is_training() == true)?
				dqn_->selectAction(input_to_nn):
				dqn_->selectAction(input_to_nn, 0.0f);

		if( loop_count_ > loop_count_max_ ){
			dqn_->printQValues(input_to_nn);
			//loop_count_ =0;	
		}
	}
	
	processInput(action);					

	float& reward = std::get<2>(transition);
	int& isTerminated = std::get<3>(transition);	// 0 : continue, 1 : terminate

	
	updateAll();
	calculateRewardAndcheckCollision(reward,isTerminated);

	if(world.is_training()){
		// store transition 	
		dqn_->addTransition(transition);
		
		reward_sum_ += 0.1; // represent the distance to travel

		// start state replay training at terminal state
		// this is terminal state
		if(isTerminated) {
			
			if (reward_max_ < reward_sum_) {
				reward_max_ = reward_sum_;
				std::cout 	<< "**************************" << endl
							<< "**[" << ID_ << "] New Record : " << reward_max_ << " **" << endl
							<< "**************************" << endl;
			}
			
			std::cout << "(Max:" << reward_max_ << ") " << "Reward sum " << reward_sum_ << endl;

			reward_sum_ = 0.0f;
		}
		if(dqn_->replay_memory_size() > training_threshold_nums_) {
			if( loop_count_ > loop_count_max_ ){
				dqn_->update(batch_size_);
				loop_count_ = 0;
			}
		}
	}
	++loop_count_;
}

// end of file 
