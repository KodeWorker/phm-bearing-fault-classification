#include <iostream>
#include <string>
#include <vector>
#include <chrono>       // std::chrono::system_clock
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
	float dLearningRate = 1e-3;
	
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	
	printf("+++ Load Dataset +++\n");
	
    std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> npy = ReadNpy(strDataPath);
    
    int64_t nTrainSamples = (int64_t)(npy.size() * (1 - dValRatio));
	int64_t nTestSamples = (int64_t)(npy.size() - nTrainSamples);
    
	// set random split dataset
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		
    std::pair<std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>>* ,
              std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>>* > npyPair = 
			  SplitVector<std::tuple<std::string /*file location*/, int64_t /*label*/>>(npy, nTrainSamples, seed);
    
	std::cout << "nTrainSamples: " << npyPair.first->size() << std::endl;
    std::cout << "nTestSamples: " << npyPair.second->size() << std::endl;
	
	auto trainDataset = MafauldaDataset(*npyPair.first).map(torch::data::transforms::Stack<>());
    auto testDataset = MafauldaDataset(*npyPair.second).map(torch::data::transforms::Stack<>());
	
	
    auto trainDataLoader = torch::data::make_data_loader(
    std::move(trainDataset),
    torch::data::DataLoaderOptions().batch_size(nTrainBatchSize).workers(2));
	
	auto testDataLoader = torch::data::make_data_loader(
    std::move(testDataset),
    torch::data::DataLoaderOptions().batch_size(nTestBatchSize).workers(2));

	printf("+++ Build Model +++\n");
	
	SignalNet model = SignalNet();	
	model->to(device);
	
	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(dLearningRate));
	torch::nn::CrossEntropyLoss criterion;
	
	int dataset_size = trainDataset.size().value();
	for (int64_t epoch = 1; epoch <= nEpochs; ++epoch) {
		size_t batch_idx = 0;
		for (torch::data::Example<>& batch : *trainDataLoader) {
			auto features = batch.data;
			auto labels = batch.target;
			
			features.to(device);
			labels.to(device);
			
			optimizer.zero_grad();
			auto output = model(features);
			auto loss = criterion(output, labels);
			
			loss.backward();
            optimizer.step();
						
			// monitor training phase
			
			auto outputTuple = torch::max(output, 1);
			float accuracy = Accuracy(std::get<1>(outputTuple) , labels); /// temp
			
			printf("\rTrain Epoch: %lld/%lld [%5zd/%5d] Loss: %.4f Acc.: %.2f %%", 
				   epoch,
				   nEpochs,
				   ++batch_idx * batch.data.size(0),
				   dataset_size,
				   loss.template item<float>(),
				   accuracy * 100);
		}
	}
	
	printf("+++ Verify Model +++\n");
	
	/*
	for (torch::data::Example<>& batch : *testDataLoader) {
        std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    }
	*/
	
	printf("+++ Done +++\n");
}