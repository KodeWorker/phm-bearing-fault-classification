#include <iostream>
#include <string>
#include <stdexcept>
#include <torch/torch.h>
#include "model.h"
//#include "dataset.h"
#include "utils.h"

int main() {
	
	printf("+++ SignalNet Initialized +++\n");
	
	std::string strDataPath = "../../../data/mafaulda";
    std::string strSaveModelPath = "../../../data/SignalNet_demo.pth";
    int nTrainBatchSize = 1024;
    int nTestBatchSize = 1024;
    int nEpochs = 25;
    float dValRatio = 0.2;
	
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	
	printf("+++ Load Dataset +++\n");
	
	//auto data_set = MafauldaDataset(strDataPath).map(torch::data::transforms::Stack<>());
	double array[] = { 1, 2, 3, 4, 5};
	auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, 1);
	torch::Tensor tharray = torch::from_blob(array, {5}, options);
	std::cout << tharray << std::endl;

	printf("+++ Build Model +++\n");
	
	//SignalNet model = SignalNet();	
	//model->to(device);
	
	printf("+++ Done +++\n");
}