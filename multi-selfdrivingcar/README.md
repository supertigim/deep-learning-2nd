## Introduction    

This project has moved to [autonomous vehicles](https://github.com/supertigim/autonomous-vehicles) repository as tensorflow/keras provides more chance to adapt other algorithms and techonologies with ease. Anyhow, this application will be also updated along with that, because tiny-dnn is also a great asset for my future project. :)     

**[Video](https://www.youtube.com/watch?v=lzlVKUNpIoc)**  
      
## How to do

You can flip the scene between training mode and normal mode by pressing space bar for one car or **"a" for all**.  
  
![](https://preview.ibb.co/dOACMa/multi_self_driving_cars.png)  

## Build Envrionment  

See CMakeLists.txt and change for your environment. Oherwise, You can refer to [this page](https://github.com/supertigim/autonomous-vehicles/tree/master/opengl-env) that has all dependencies are in external folder.  

OpenGL related dependancies are quite annoying to me. Sorry for my unconvenient description. :(  
  
## How to build  

	$ mkdir build 
	$ cd build

	build$ cmake ..
	build$ make 

	// run 
	build$ ./multi_self_driving_car

## Reference  

1. [Deep Q-Networks and Beyond 한글버전](http://ishuca.tistory.com/396)  
2. [Double Dueling DQN tutorial in Python](https://gist.github.com/awjuliani/fffe41519166ee41a6bd5f5ce8ae2630)  
3. [Chainer로 구현된 D-DQN](https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/agent.py)  
4. [Chainer로 구현된 Prioritized Replay Memory](https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py)  
5. [Chainer로 구현된 Dueling DQN](https://github.com/musyoku/dueling-network/blob/master/dueling_network.py)  
6. [Prioritized Replay Memory Tutorial](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/)   

