#ifndef UTILS_H
#define UTILS_H

/*
#include <windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>
#pragma comment(lib, "User32.lib")
*/
#include <vector>
#include <tuple>
#include <sys/stat.h>
#include <unordered_set>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <torch/torch.h>

#include <iostream>
float Accuracy(torch::Tensor predTensor, torch::Tensor trueTensor){
	std::cout << predTensor.sizes() << std::endl;
	std::cout << trueTensor.sizes() << std::endl;
	return 0;
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

auto ReadNpy(std::string& location) -> std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> {

    std::vector<std::tuple<std::string, int64_t>> npy;
	std::vector<std::string> pathVector;
	std::vector<std::string> dirnameVector;
	
    std::cout << location << std::endl;
    
    
    // walk directory tree
    /*
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    
    location += "\\normal\\*";
    
    hFind = FindFirstFile(location.c_str(), &ffd);
    do
    {
        if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
         _tprintf(TEXT("  %s   <DIR>\n"), ffd.cFileName);
        }
        else
        {
         _tprintf(TEXT("  %s   \n"), ffd.cFileName);
        }
    }
    while (FindNextFile(hFind, &ffd) != 0);
 
    FindClose(hFind);
    */
    
    /*
	for(auto& p: std::filesystem::recursive_directory_iterator(location)) {
		std::string path{p.path().u8string()};
		std::string dirname{p.path().parent_path().filename().u8string()};
		if(HasSuffix(path, ".npy")){
			pathVector.push_back(path);
			dirnameVector.push_back(dirname);
		}
	}
	
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
	*/
    return npy;
}

#endif UTILS_H