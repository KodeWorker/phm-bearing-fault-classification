# SignalNet C++ Implementation
This is C++ implementation of SignalNet using Libtorch.

The classification accuracy is around 99% on Mafaulda dataset (generated npy from python implementation).

# Project Structure
```
.
+-- include/
|    +-- ConvertUTF.c
|    +-- ConvertUTF.h
|    +-- dataset.h
|    +-- model.h
|    +-- npy.hpp
|    +-- SimpleIni.h
|    +-- utils.h
+-- src/
|    +-- main.cpp
+-- CMakeLists.txt
+-- config.ini
+-- readme.md
```

- ConvertUTF.c/ConvertUTF.h/SimpleIni.h: for reading config file , from Github [repository](https://github.com/brofield/simpleini)
- npy.hpp: for reading npy using C++, from Github [repository](https://github.com/llohse/libnpy.git)
- dataset.h: libtorch dataset implementation
- model.h: libtorch SignalNet model implementation
- utils.h: utility functions
- main.cpp: main program entry
- CMakeLists.txt: CMakeLists for cmake
- config.ini: config for execution

# Instructions

## Dependencies
- CMake > 3.0
- Visual Studio 16 2019
- Libtorch
- Boost C++ Library

## Build executable
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