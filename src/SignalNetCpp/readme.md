cmake -DCMAKE_PREFIX_PATH=D:\kelvinwu\code\libtorch -DBOOST_ROOT:PATH=D:\kelvinwu\code\boost_1_75_0 -DBOOST_INCLUDEDIR:PATH=D:\kelvinwu\code\boost_1_75_0 -DBOOST_LIBRARYDIR:PATH=D:\kelvinwu\code\boost_1_75_0\libs ..
cmake --build . --config Release

# Refereces
- https://pytorch.org/tutorials/advanced/cpp_frontend.html
- https://discuss.pytorch.org/t/libtorch-how-to-use-torch-datasets-for-custom-dataset/34221/2
- https://github.com/mhubii/libtorch_custom_dataset
- https://github.com/llohse/libnpy.git
- https://shengyu7697.github.io/blog/2019/12/22/Boost-use-cmake-visual-studio-in-windows/
