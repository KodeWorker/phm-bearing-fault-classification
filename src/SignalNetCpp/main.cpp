#include <iostream>
#include <string>
#include <stdexcept>
#include <torch/torch.h>
#include "model.h"
//#include "dataset.h"
#include "utils.h"


using namespace std;

int main() {
	
	printf("+++ SignalNet Initialized +++\n");
	
	string strDataPath = "../../../data/mafaulda";
    string strSaveModelPath = "../../../data/SignalNet_demo.pth";
    int nTrainBatchSize = 1024;
    int nTestBatchSize = 1024;
    int nEpochs = 25;
    float dValRatio = 0.2;
	
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	
	printf("+++ Load Dataset +++\n");
	
	auto data_set = MafauldaDataset(strDataPath).map(torch::data::transforms::Stack<>());
	
	
	printf("+++ Build Model +++\n");
	
	//SignalNet model = SignalNet();	
	//model->to(device);
	
	printf("+++ Done +++\n");
}