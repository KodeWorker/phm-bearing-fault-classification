call conda activate pytorch-1.7.1-cuda-10.1
call python src/FaultNet/main.py --data_path data/CWRU/CWRU_dataset.npy ^
--label_path data/CWRU/CWRU_labels.npy ^
--save_model_path data/FailtNet_demo.pth