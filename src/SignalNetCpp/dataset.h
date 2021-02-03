#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <tuple>
#include <torch/torch.h>
#include "utils.h"
#include "cnpy.h"

class MafauldaDataset : public torch::data::Dataset<MafauldaDataset>
{
	private:
        std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> npy_;
	
	public:
		explicit CustomDataset(std::string& data_dir) : npy_(ReadNpy(data_dir)){
			
		};
		
		torch::data::Example<> get(size_t index) override {
			
			std::string file_location = std::get<0>(npy_[index]);
            int64_t label = std::get<1>(npy_[index]);
			
			cnpy::NpyArray arr = cnpy::npy_load(file_location);
			
			std::cout << arr.shape[0] << "," << arr.shape[1] << "," << arr.shape[2] << std::endl;
			
			torch::Tensor img_tensor = torch::from_blob(arr.data, {arr.shape[0], arr.shape[1], arr.shape[2]}, torch::kByte).clone()// read from npy
			img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW
			
			torch::Tensor label_tensor = torch::full({1}, label);

            return {img_tensor, label_tensor};
		};
		
		// Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {

            return npy_.size();
        };
		
}

#endif DATASET_H