#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <tuple>
#include <filesystem>
#include <sys/stat.h>
#include <unordered_set>

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
	
    return npy;
}

#endif UTILS_H