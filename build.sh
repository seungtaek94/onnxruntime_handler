#!/bin/bash

cd onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz
tar -zxvf onnxruntime-linux-x64-1.12.1.tgz
cd ../

echo "@@@@@@@@@@@@"
ls -al

git clone -b release-1.12.1 https://github.com/google/googletest.git

cmake -DCMAKE_BUILD_TYPE=Debug -S ./ -B ./cmake-build-debug-linux

cmake --build cmake-build-debug-linux/ --target orthandler -j 2

cmake --build cmake-build-debug-linux/ --target test_orthandler -j 2