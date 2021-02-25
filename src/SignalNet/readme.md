# SignalNet
Simplied FaultNet implementation without using additional features.

The classification accuracy is around 99% on Mafaulda dataset .

# Project Structure
```
.
+-- dataset.py
+-- eval.py
+-- main.py
+-- model.py
+-- preprocessing_1.py
+-- preprocessing_2.py
+-- readme.py
+-- runEvalSignalNet.bat
+-- runPreprocessing.bat
+-- runTrainSignalNet.bat
```

- dataset.py: dataset implementation
- eval.py:: validation flow
- main.py:: training flow
- model.py: model implementation
- preprocessing_1.py: devide signal files(*.csv) into segmented signal files(*.csv)
- preprocessing_2.py: reshape signal files into CxHxW format and convert to *.npy
- runEvalSignalNet: batch file for running SignalNet validation flow
- runPreprocessing.bat: batch file for running preprocessing flow
- runTrainSignalNet.bat: batch file for running SignalNet training flow

# Instructions
1. Install CUDA (in my case: cuda 10.1)
2. Setup python environment
```
conda create --name pytorch-1.7.1-cuda-10.1 python=3.8
conda activate pytorch-1.7.1-cuda-10.1
pip install -r requirements.txt
```
3. Install corresponding PyTorch version
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
4. Download dataset from Mafaulda dataset website and name the 6-category folder as following:
    - normal
    - imbalance
    - horizontal-misalignment
    - vertical-misalignment
    - underhang
    - overhang
5. Run runPreprocessing.bat
6. Run runTrainSignalNet.bat.bat
7. Run runEvalSignalNet.bat.bat

# References
- https://github.com/joshuadickey/FaultNet
- http://www02.smt.ufrj.br/~offshore/mfs/page_01.html