#!/bin/bash

USE_GPU=false

while (("$#"))
do
        case "$1" in
                --gpu)
                        echo "use gpu"
                        USE_GPU=true
                        shift
                        ;;
                -*|--*)
                        echo "[Error] Unsupported flag: $1" >&2
                        exit 1
                        ;;
                *)
                        echo "[Error] Arguments with not proper flag: $1" >&2
                        exit 1
                        ;;
        esac
done

if [ "$USE_GPU" = true ]
then
  TAG_GPU="-gpu"
else
  TAG_GPU=""
fi

mkdir onnxruntime
cd onnxruntime
rm -rf ./*
wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64${TAG_GPU}-1.12.1.tgz
tar -zxvf onnxruntime-linux-x64${TAG_GPU}-1.12.1.tgz
cd ../

git clone -b release-1.12.1 https://github.com/google/googletest.git

cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_GPU=${USE_GPU} -S ./ -B ./cmake-build-debug-linux

cmake --build cmake-build-debug-linux/ --target orthandler -j 2
cmake --build cmake-build-debug-linux/ --target test_orthandler -j 2