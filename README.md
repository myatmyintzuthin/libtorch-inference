# torchlib-inference

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Torch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Cmake](https://img.shields.io/badge/CMake-064F8C?style=for-the-badge&logo=cmake&logoColor=white)

This repo consists of C++ libtorch inference implementation for pytorch classification model.

### Install Dependencies

1. Download libtorch distribution from [official website](https://pytorch.org/get-started/locally/). libtorch version used in this repo [link](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip).
2. Unzip the downloaded file and store `libtorch` folder in `external_packages` folder. If `external_packages` folder doesn't exit, create one.
3. Install OpenCV: follow the installation instructions in this [guide](https://github.com/myatmyintzuthin/extract-table/blob/C%2B%2B/Installation_Guide.md).

### Run inference
- Before running inference, the pytorch model should be converted to script model. Reference, [trace_model.py](https://github.com/myatmyintzuthin/jpfood-classification-pytorch/blob/master/trace_model.py).
- Build cmake:
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```
- run inference, inside `build` folder: 
```
./libtorch --model_path ../script_model/resnet_food_script.pt --image_path ../test-images/sushi.jpg
```

### Results

```
Finish model loading ... 
Finish image processing ... 
---------------------------------------
Predicted class = Sushi | confidence = 0.999214
```

