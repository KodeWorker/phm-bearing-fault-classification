call conda activate pytorch-1.7.1-cuda-10.1
call python preprocessing_1.py --data_path ../../data/MAFAULDA^
--output_dir ../../data/MAFAULDA_X^
--n_segments 4096
call python preprocessing_2.py --data_path ../../data/MAFAULDA_X^
--output_dir ../../data/MAFAULDA_XX