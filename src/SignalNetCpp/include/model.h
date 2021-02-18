#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>

#include <iostream>
struct SignalNetImpl : public torch::nn::Module {
  
  SignalNetImpl(int64_t channels, int64_t height, int64_t width) :
    conv1(torch::nn::Conv2dOptions(channels, 32, 4).stride(1).padding(1)),
	conv2(torch::nn::Conv2dOptions(32, 64, 4).stride(1)),
    n(GetConvOutput(channels, height, width)),
	fc1(n, 256),   
    fc2(256, 10)    
  {
   // register_module() is needed if we want to use the parameters() method later on
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("fc1", fc1);
   register_module("fc2", fc2);
  }
  
  torch::Tensor forward(torch::Tensor x) {
    
    //std::cout << x.max() << std::endl;
    //std::cout << x.min() << std::endl;
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 4));
    //std::cout << "1" << std::endl;
    x = torch::relu(torch::max_pool2d(conv2->forward(x), 4));
	//std::cout << "2" << std::endl;
    x = x.view({-1, n});
    //std::cout << "3" << std::endl;
    x = torch::relu(fc1->forward(x));
    //std::cout << "4" << std::endl;
    x = torch::dropout(x, 0.2, is_training());
    //std::cout << "5" << std::endl;
    x = torch::log_softmax(fc2->forward(x), 1);
	
	return x;
  }
  
  int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width) {
      torch::Tensor x = torch::zeros({1, channels, height, width});
      //std::cout << x.max() << std::endl;
      x = torch::max_pool2d(conv1->forward(x), 4);
      x = torch::max_pool2d(conv2->forward(x), 4);
      return x.numel();
  }
  
  torch::nn::Conv2d conv1, conv2;
  int64_t n;
  torch::nn::Linear fc1, fc2;
  
};

TORCH_MODULE(SignalNet);

#endif MODEL_H