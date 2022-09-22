mkdir onnxruntime
cd onnxruntime
dir
powershell Invoke-WebRequest https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-win-x64-1.12.1.zip
dir
unzip .\onnxruntime-win-x64-1.12.1.zip
dir
cd ../

git clone -b release-1.12.1 https://github.com/google/googletest.git

cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Debug -S .\ -B .\cmake-build-debug-vs2019

cmake --build .\cmake-build-debug-vs2019 --target orthandler --config Debug
cmake --build .\cmake-build-debug-vs2019 --target test_orthandler --config Debug