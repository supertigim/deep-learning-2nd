## Introduction  

[All previous neural net studies](https://github.com/supertigim/deep.learning) are rewritten by using tiny-dnn. It will help me to test an advanced algorithm easily. Also, there is python source code with tensorflow in python-tf directory which comes from online lectures by professor [Sunghun Kim](http://www.cse.ust.hk/~hunkim/) at HKUST.

## Changes in tiny-DNN  

**Random generator in random.h**  
	
	class random_generator {  
	public:  
		static random_generator &get_instance() {  
		static random_generator instance;  
		return instance;  
	}  

	std::mt19937 &operator()() {   
		set_seed(rd_());  // ADD!!!
		return gen_;  
	}    

	void set_seed(unsigned int seed) { gen_.seed(seed); }  

	private:  
		// avoid gen_(0) for MSVC known issue  
		// https://connect.microsoft.com/VisualStudio/feedback/details/776456  
		random_generator() : gen_(3) {}  
		std::mt19937 gen_;  
		std::random_device rd_;	// ADD!!!  
	};

**Uncomment defines in config.h**   

	#define CNN_USE_SSE  
	#define CNN_USE_OMP  
	#define CNN_USE_GCD  
	#define CNN_USE_EXCEPTIONS  
	#define CNN_USE_STDOUT  
  
## Reference  

1.[tiny-dnn online manual](http://tiny-dnn.readthedocs.io/en/latest/index.html)  
2.[comparison between neural network libraries](https://github.com/tiny-dnn/tiny-dnn/tree/v1.0.0a3#comparison-with-other-libraries)  
3.[Solving XOR problem using tiny-DNN](http://linerocks.blogspot.kr/2017/02/solving-xor-problem-using-tiny-dnn_89.html)  
4.[딥러닝 활용 추천 시스템 개발](https://www.buzzvil.com/2017/02/22/buzzvil-techblog-tensorflow-deeplearning/)  
5.[구글 AI Experiments](https://aiexperiments.withgoogle.com/)  
6.[AI Duet이라는 들은 음악에 맞춰 자동 연주](https://github.com/googlecreativelab/aiexperiments-ai-duet)  
7.[Caffe로 구현한 breakout with DQN](https://github.com/muupan/dqn-in-the-caffe)  
8.[python으로 구현한 self driving car](https://github.com/musyoku/self-driving-cars)  
9.[8번 설명 사이트](http://deeplearningstudy.github.io/doc_caffe_intro.html)   
9.[Caffe c++ 스터디](https://github.com/DeepLearningStudy)  

## license   
  
**MIT**
