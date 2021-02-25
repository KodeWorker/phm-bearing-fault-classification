# SignalNet C++ Implementation
This is C++ implementation of SignalNet using Libtorch

# Project Structure

# Instructions

## Dependencies
- Libtorch
- Boost C++ Library

## Build executable
W
```
mkdir build
cd build
cmake -G"Visual Studio 16 2019"  ..
cmake --build . --config Release
```

## Run executable
```
Release\signalnet.exe -c ../config.ini
```

# Refereces
- https://pytorch.org/tutorials/advanced/cpp_frontend.html
- https://discuss.pytorch.org/t/libtorch-how-to-use-torch-datasets-for-custom-dataset/34221/2
- https://github.com/mhubii/libtorch_custom_dataset
- https://github.com/llohse/libnpy.git
- https://shengyu7697.github.io/blog/2019/12/22/Boost-use-cmake-visual-studio-in-windows/
- https://gist.github.com/UnaNancyOwen/d879a41710e9c05025f8
- https://github.com/brofield/simpleini