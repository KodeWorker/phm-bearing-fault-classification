#include <iostream>
#include <string>
#include <vector>
#include <chrono>       // std::chrono::system_clock
#include <stdlib.h>     // atof 
#include <torch/torch.h>
#include <model.h>
#include <dataset.h>
#include <utils.h>
#include <SimpleIni.h>

int main(int argc, char* argv[]) {
	
    const char* charConfigPath;
    
    for(int i = 1; i < argc; i++){
		if(!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help") ){
			printf("Usage: App <options>\nOptions are:\n");
			printf("-c, --config : path to config file\n");
			exit(0);
		}else if(!strcmp(argv[i], "-c") || !strcmp(argv[i], "--config" )){
            charConfigPath = argv[i+1];
			printf("Read configuration: %s\n", charConfigPath);
		}else{
			if(i == argc-1){
				break;
			}
            printf("ERROR: Invalid Command Line Option Found: \"%s\".\n", argv[i]);
		}
	}
    
	printf("+++ SignalNet Initialized +++\n");
    
	CSimpleIniA ini;
	ini.SetUnicode();

	SI_Error rc = ini.LoadFile(charConfigPath);
    int seed = atoi(ini.GetValue("Torch", "seed", "-1"));
    const char* charDataPath = ini.GetValue("Dataset", "DataPath", "");
    const std::string strDataPath(charDataPath);
    const char* charSaveModelPath = ini.GetValue("Dataset", "SaveModelPath", "");
    const std::string strSaveModelPath(charSaveModelPath);
    const float dValRatio = atof(ini.GetValue("Dataset", "ValRatio", "0.2"));
    const size_t nTrainBatchSize = atoi(ini.GetValue("Dataset", "TrainBatchSize", "512"));
    const size_t nTestBatchSize = atoi(ini.GetValue("Dataset", "TestBatchSize", "512"));
    const size_t nWorkers = atoi(ini.GetValue("Dataset", "Workers", "2"));    
    const size_t nEpochs = atoi(ini.GetValue("Model", "Epochs", "25"));
    const float dLearningRate = atof(ini.GetValue("Model", "LearningRate", "1e-3"));
    const int64_t nModelChannels = atoi(ini.GetValue("Model", "Channels", "8"));
    const int64_t nModelHeight = atoi(ini.GetValue("Model", "Height", "64"));
    const int64_t nModelWidth = atoi(ini.GetValue("Model", "Width", "64"));
    
    printf("*** Torch ***\n");
    printf("seed: %d\n", seed);
    
    printf("*** Dataset ***\n");
    printf("DataPath: %s\n", strDataPath.c_str());
    printf("SaveModelPath: %s\n", strSaveModelPath.c_str());
    printf("ValRatio: %.2f\n", dValRatio);
    printf("TrainBatchSize: %zd\n", nTrainBatchSize);
    printf("TestBatchSize: %zd\n", nTestBatchSize);
    printf("Workers: %zd\n", nWorkers);    
    
    printf("*** Model ***\n");
    printf("Epochs: %zd\n", nEpochs);
    printf("LearningRate: %.4f\n", dLearningRate);
    printf("Model Size: (%zd, %zd, %zd)\n", nModelChannels, nModelHeight, nModelWidth);
    
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	printf("+++ Load Dataset +++\n");
	
    std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> npy = ReadNpy(strDataPath);
    
    size_t nTrainSamples = (size_t)(npy.size() * (1 - dValRatio));
	size_t nTestSamples = (size_t)(npy.size() - nTrainSamples);
    
	// set random split dataset
    if(seed==-1)
        seed = std::chrono::system_clock::now().time_since_epoch().count();
	torch::manual_seed(seed);
    
    std::pair<std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>>* ,
              std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>>* > npyPair = 
			  SplitVector<std::tuple<std::string /*file location*/, int64_t /*label*/>>(npy, nTrainSamples, seed);
    
	std::cout << "nTrainSamples: " << npyPair.first->size() << std::endl;
    std::cout << "nTestSamples: " << npyPair.second->size() << std::endl;
	
	auto trainDataset = MafauldaDataset(*npyPair.first).map(torch::data::transforms::Stack<>());
    auto testDataset = MafauldaDataset(*npyPair.second).map(torch::data::transforms::Stack<>());
	
    auto trainDataLoader = torch::data::make_data_loader(
    std::move(trainDataset),
    torch::data::DataLoaderOptions().batch_size(nTrainBatchSize).workers(nWorkers));
	
	auto testDataLoader = torch::data::make_data_loader(
    std::move(testDataset),
    torch::data::DataLoaderOptions().batch_size(nTestBatchSize).workers(nWorkers));
    
	printf("+++ Build Model +++\n");
	SignalNet model(nModelChannels, nModelHeight, nModelWidth);
    model->to(device);
    
	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(dLearningRate));
	torch::nn::CrossEntropyLoss criterion;
	
    for (size_t epoch = 1; epoch <= nEpochs; ++epoch) {
		size_t batch_idx = 0;
        model->train();
        for (torch::data::Example<>& batch : *trainDataLoader) {
            
            auto features = batch.data.to(torch::kFloat32).to(device);
            auto labels = batch.target.to(torch::kLong).to(device);
            optimizer.zero_grad();
			auto output = model->forward(features);
            labels = torch::squeeze(labels, 1);
			auto loss = criterion(output, labels);
            loss.backward();
            optimizer.step();
            
            // monitor training phase			
			auto outputTuple = torch::max(output, 1);
            float accuracy = Accuracy(std::get<1>(outputTuple) , labels);
            
            printf("\rTrain Epoch: %lld/%lld [%5zd / %5zd] Loss: %.4f Acc.: %.2f %%", 
				   epoch,
				   nEpochs,
				   ++batch_idx * batch.data.size(0),
				   nTrainSamples,
				   loss.template item<float>(),
				   accuracy * 100);
		}
	}
    printf("\n");
    
    printf("+++ Verify Model +++\n");    
    model->eval();
    
    torch::Tensor predTensor;
    torch::Tensor trueTensor;
    bool isInitialized = false;
    
	for (torch::data::Example<>& batch : *testDataLoader) {
        
        auto features = batch.data.to(torch::kFloat32).to(device);
        auto labels = batch.target.to(torch::kLong).to(device);
        labels = torch::squeeze(labels, 1);
        
        auto output = model->forward(features);
        auto outputTuple = torch::max(output, 1);
        auto preds = std::get<1>(outputTuple);
       
        if(!isInitialized){
            predTensor = preds;
            trueTensor = labels;
            isInitialized = true;
        }else{
            predTensor = torch::cat({predTensor, preds}, 0);
            trueTensor = torch::cat({trueTensor, labels}, 0);
        }
        
    }
    
	float accuracy = Accuracy(predTensor , trueTensor);
    printf("Acc.: %.2f %%\n", accuracy * 100);
    
    printf("+++ Save Model +++\n");
    torch::save(model, strSaveModelPath);
    
	printf("+++ Done +++\n");
    
}