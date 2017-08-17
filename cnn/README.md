## Introduction  

This app makes a dot randomly on 10*10 screen and predicts the location of the dot with tiny-dnn. It takes more than 30 minutes with my old Mac-pro to train in order to reach less than 0.00015 error rate, but definitely works well just with the 2-layer convolutional network.   

## How to build & run

	// build  
	mkdir build  
	cd build  
	cmake .. 
	make   
	  
	//run  
	./cnn  

![](https://preview.ibb.co/fOf9HF/run_cnn_exam.png)  
