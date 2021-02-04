#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <torch/torch.h>
#include "model.h"
#include "dataset.h"
#include "utils.h"

int main() {
	
	printf("+++ SignalNet Initialized +++\n");
	
	std::string strDataPath = "../../../data/MAFAULDA_XX";
    std::string strSaveModelPath = "../../../data/SignalNet_demo.pth";
    int64_t nTrainBatchSize = 1024;
    int64_t nTestBatchSize = 1024;
    int64_t nEpochs = 25;
    float dValRatio = 0.2;
	
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	
	printf("+++ Load Dataset +++\n");
	
    std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> npy = ReadNpy(strDataPath);
    
    int64_t nTrainSamples = (int64_t)(npy.size() * (1 - dValRatio));
    std::cout << "nTrainSamples: " << nTrainSamples << std::endl;
    
    std::pair<std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> ,
              std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> > npyPair = SplitVector<std::tuple<std::string /*file location*/, int64_t /*label*/>>(npy, nTrainSamples);
    
	auto dataset = MafauldaDataset(npy).map(torch::data::transforms::Stack<>());
    
    /*
    auto data_loader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(nTrainBatchSize).workers(2));

    for (torch::data::Example<>& batch : *data_loader) {
        std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    }
    */
	printf("+++ Build Model +++\n");
	
	//SignalNet model = SignalNet();	
	//model->to(device);
	
	printf("+++ Done +++\n");
}