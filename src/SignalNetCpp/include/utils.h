#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <tuple>
#include <sys/stat.h>
#include <unordered_set>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <map>
#include <boost/filesystem.hpp>
#include <torch/torch.h>


using namespace boost::filesystem;

//#include <iostream>
std::pair<std::map<int, int>, std::map<int, int>> ClassificationReport(torch::Tensor predTensor, torch::Tensor trueTensor){
    trueTensor = trueTensor.to(torch::kLong);
    predTensor = predTensor.to(torch::kLong);
    
    std::map<int, int> mapCorrectCount;
    std::map<int, int> mapTotalCount;
    
    for (int i=0; i<trueTensor.numel(); i++){
        //std::cout << trueTensor[i] << "," << predTensor[i] << std::endl;
        
        if(mapTotalCount.count(trueTensor[i].item<int>())==0){
            mapTotalCount[trueTensor[i].item<int>()] = 0;
            mapCorrectCount[trueTensor[i].item<int>()] = 0;
        }
        
        mapTotalCount[trueTensor[i].item<int>()] += 1;
        if(trueTensor[i].eq(predTensor[i]).item<int>())
            mapCorrectCount[trueTensor[i].item<int>()] += 1;
    }
    
    return std::make_pair(mapCorrectCount, mapTotalCount);
}

float Accuracy(torch::Tensor predTensor, torch::Tensor trueTensor){
    trueTensor = trueTensor.to(torch::kLong);
    predTensor = predTensor.to(torch::kLong);
	auto correct = (trueTensor.eq(predTensor)).sum();
    //std::cout << correct << std::endl;
    //std::cout << correct.item<float>() << std::endl;
    //std::cout << predTensor.numel() << std::endl;
    return correct.item<float>()/predTensor.numel();
}

template< class T > 
std::pair<std::vector<T>* , std::vector<T>* > SplitVector(const std::vector< T >& vecIn, int nbIn, unsigned seed)
{
    auto vecC = vecIn;
    std::shuffle(vecC.begin(), vecC.end(), std::default_random_engine(seed));
    auto vec1 = new std::vector<T>(vecC.begin(), vecC.begin() + nbIn);
    auto vec2 = new std::vector<T>(vecC.begin() + nbIn, vecC.end());
    return std::make_pair(vec1, vec2);
}

bool IsDirExists(const std::string &path) {
    struct stat info;
    if (stat(path.c_str(), &info) == 0 && info.st_mode & S_IFDIR) {
        return true;
    }
    return false;
}

bool HasSuffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

auto ReadNpy(const std::string& location) -> std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> {

    std::vector<std::tuple<std::string, int64_t>> npy;
	std::vector<std::string> pathVector;
	std::vector<std::string> dirnameVector;
	
    //std::cout << location << std::endl;
    directory_iterator end_itr;
    for (directory_iterator itr(location); itr != end_itr; ++itr){
        if (is_directory(itr->path())) {
            std::string dirname = itr->path().filename().string();
            //std::cout << dirname << std::endl;
            
            std::string category_directory = itr->path().string();
            directory_iterator end_citr;
            for (directory_iterator citr(category_directory); citr != end_citr; ++citr){
                std::string path = citr->path().string();
                pathVector.push_back(path);
                dirnameVector.push_back(dirname);
                //std::cout << " - " + path << std::endl;
            }
        }
    }
    
    /*
	for(auto& p: std::filesystem::recursive _directory_iterator(location)) {
		std::string path{p.path().u8string()};
		std::string dirname{p.path().parent_path().filename().u8string()};
		if(HasSuffix(path, ".npy")){
			pathVector.push_back(path);
			dirnameVector.push_back(dirname);
		}
	}
	*/
    
	std::unordered_set<std::string> dirnameSet;
	for (std::string dirname : dirnameVector)
		dirnameSet.insert(dirname);
	std::map<std::string, int64_t> mapLabel;
	int64_t label = 0;
	for (const auto& elem: dirnameSet) {
		mapLabel[elem] = label;
		label += 1;
	}
	for(int i=0; i<pathVector.size(); i++){
		std::string path = pathVector[i];
        int64_t label = mapLabel[dirnameVector[i]];
        npy.push_back(std::make_tuple(path, label));
	}
    return npy;
}

#endif UTILS_H