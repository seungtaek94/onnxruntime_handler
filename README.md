# onnxruntime_handler

Simple `c++` onnxruntime handler. 

## Build Status
|     | Windows-x64                                                                                                                            | Linux-x64                                                                                                                                |
|-----|----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| CPU | ![Build Status](https://github.com/seungtaek94/onnxruntime_handler/actions/workflows/build_orthandler_win-x64-cpu_debug.yml/badge.svg) | ![Build Status](https://github.com/seungtaek94/onnxruntime_handler/actions/workflows/build_orthandler_linux-x64-cpu_debug.yml/badge.svg) |
| GPU |                                                                                                                                        | ![Build Status](https://github.com/seungtaek94/onnxruntime_handler/actions/workflows/build_orthandler_linux-x64-gpu_debug.yml/badge.svg) |                                                                                                                                     

## Implementation status

- [X] Inference without batch.
- [X] Inference on the CPU.
- [X] Support `float32` data type.
- [X] Support multiple output.
- [X] Inference on the GPU.
- [ ] Dynamic input shape.
- [ ] Batch inference.
- [ ] Support for various data types


## Requirements
- cmake >= 3.16.x
- onnxruntime
- `MSVC 16.xx(Visual Studio 2019)` for windows
- `unzip` for windows

## Tested System
- win-x64
  - Tested on Windows 10, Windows server 2019 with visudal studio 2019
- linux-x64
  - Tested on ubuntu 20.04

## Build
### Common
- Install CMake that version upper then 3.16.x

### Linux

```Bash
git clone https://github.com/seungtaek94/onnxruntime_handler.git
cd onnxruntime_handler
./build.sh
```

### Windows
`Warning` - `MSVC 16.xx` and `unzip` should be installed before run the `.\build.bat`

```Bash
git clone https://github.com/seungtaek94/onnxruntime_handler.git
cd onnxruntime_handler
.\build.bat
```

## How to use

1. Load model and initialize Ort::Handler.
2. Make tensor from data array.
   - Currently, this function only `supports 3-channel single image`.
3. Run.

```c++
#include "OrtHandler.h"

// 1. Load model and initialize Ort::Handler.
InferenceOption inferenceOption;
std::unique_ptr<Ort::Handler> orthandler = Ort::Handler::LoadModel(
        "./assets/models/conv1x1.onnx", inferenceOption);

// 2. Make tensor from data array.
// Currently, this function only supports 3-channel single image.
float input_data[6][3] = {
            {1.1f, 1.2f, 1.3f},{2.1f, 2.2f, 2.3f},
            {3.1f, 3.2f, 3.3f},{4.1f, 4.2f, 4.3f},
            {5.1f, 5.2f, 5.3f},{6.1f, 6.2f, 6.3f},
    };
Tensor<float> input_tensor  = Ort::Handler::ToTensor((float*)input_data, 2, 3);

// 3. Run.
std::vector<Tensor<float>> output_tensor = orthandler->Run(input_tensor);
```

### Data from opencv cv::Mat

If using `OpenCV` then you can make a tensor like the below:

1. Load image.
2. Convert cv::Mat type to `CV_32FC3`.
3. Make tensor.

```c++

cv::Mat src = cv::imread("/path/for/image");
src.convertTo(src, CV_32FC3);

Tensor<float> input_tensor  = Ort::Handler::ToTensor((float*)src.data, src.rows, src.cols);
```