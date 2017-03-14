#include "TestDrivingScene.h"
#include "Agent.h"

int main(int argc, char** argv) {
	set_random_seed(3);
	
	TestDrivingScene scene;
	Agent agent(&scene);
	
	scene.init();
	agent.init();
	
	// agent runs TestDrivingScene::run(), so you don't need to call the function here
	agent.driveCar();
	
	return 0;
}

// end of file 
