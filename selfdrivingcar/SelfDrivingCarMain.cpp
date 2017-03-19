#include "TestDrivingScene.h"
#include "Agent.h"
#include "ctime"

int main(int argc, char** argv) {
	set_random_seed(1);
	
	TestDrivingScene scene;
	Agent agent(&scene);
	
	scene.init();
	agent.init();
	
	// agent runs TestDrivingScene::run(), so you don't need to call the function here
	agent.driveCar();
	
	return 0;
}

// end of file 
