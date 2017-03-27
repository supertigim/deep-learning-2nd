
#include "SelfDrivingWorld.h"
#include "ctime"

int main(int argc, char** argv) {
	set_random_seed(1);
	
	//TestDrivingScene scene;
	SelfDrivingWorld& world = SelfDrivingWorld::get();

	world.initialize();
	world.run();
	
	return 0;
}


// end of file 
