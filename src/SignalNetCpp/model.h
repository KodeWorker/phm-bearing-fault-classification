#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>

struct SignalNetImpl : torch::nn::Module {
	
  SignalNetImpl() :
    Conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 32, 4).stride(1).padding(1))),
	//MaxPool1(torch::nn::FractionalMaxPool2d(torch::nn::FractionalMaxPool2dOptions(4).stride(2))),
	MaxPool1(torch::nn::FractionalMaxPool2d(torch::nn::FractionalMaxPool2dOptions(4))),
	Conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(1))),
	//MaxPool2(torch::nn::FractionalMaxPool2d(torch::nn::FractionalMaxPool2dOptions(4).stride(2))),
	MaxPool2(torch::nn::FractionalMaxPool2d(torch::nn::FractionalMaxPool2dOptions(4))),
	FC1(torch::nn::Linear(torch::nn::LinearOptions(9216, 256))),
	dropout(torch::nn::Dropout(torch::nn::DropoutOptions().p(0.2))),
	FC2(torch::nn::Linear(torch::nn::LinearOptions(256, 10)))
  {
   // register_module() is needed if we want to use the parameters() method later on
   register_module("Conv1", Conv1);
   register_module("MaxPool1", MaxPool1);
   register_module("Conv2", Conv2);
   register_module("MaxPool2", MaxPool2);
   register_module("FC1", FC1);
   register_module("dropout", dropout);
   register_module("FC2", FC2);
  }
  
  torch::Tensor forward(torch::Tensor x) {
    
	auto in_size = x.size(0);
	
	x = torch::nn::functional::relu(MaxPool1(Conv1(x)));
	x = torch::nn::functional::relu(MaxPool2(Conv2(x)));
	x = x.view({in_size, -1});
	x = torch::nn::functional::relu(FC1(x));
	x = dropout(x);
	x = FC2(x);
	x = torch::nn::functional::log_softmax(x, torch::nn::functional::LogSoftmaxFuncOptions(1));
	
	return x;
  }
  
  torch::nn::Conv2d Conv1, Conv2;
  torch::nn::FractionalMaxPool2d MaxPool1, MaxPool2;
  torch::nn::Linear FC1, FC2;
  torch::nn::Dropout dropout;
  
};

TORCH_MODULE(SignalNet);

#endif MODEL_H