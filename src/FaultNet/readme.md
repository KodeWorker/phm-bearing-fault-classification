# FaultNet
Another FaultNet implementation modified from original [repository](https://github.com/joshuadickey/FaultNet)
The classification accuracy is 98%, which is similar to the paper results(95~08%) on CWRU dataset.

# Project Structure
```
.
+-- feature.py
+-- main.py
+-- model.py
+-- preprocessing.py
+-- readme.md
+-- runFaultNet.bat
+-- runPreprocessing.bat
```

- feature.py: contains three features(mean, median, original signal)
- main.py:: training/validation flow
- model.py: model implementation
+-- preprocessing.py: independent data preprocessing flow
+-- runFaultNet.bat: batch file for running training/validation flow
+-- runPreprocessing.bat: batch file for running data preprocessing flow

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
4. Download dataset from CWRU website and name the 10-category folder as following:
    - Normal
    - B007
    - B014
    - B021
    - IR007
    - IR021
    - OR007
    - OR014
    - OR021
5. Run runPreprocessing.bat
6. Run runFaultNet.bat

# References
- https://github.com/joshuadickey/FaultNet
- https://csegroups.case.edu/bearingdatacenter/pages/download-data-file
