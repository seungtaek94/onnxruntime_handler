# onnxruntime_handler

Now this repository has only sample code for CI.

## Build Status
| Windows-x64                                                                                                                        | Linux-x64                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| ![Build Status](https://github.com/seungtaek94/onnxruntime_handler/actions/workflows/build_orthandler_win-x64_debug.yml/badge.svg) | ![Build Status](https://github.com/seungtaek94/onnxruntime_handler/actions/workflows/build_orthandler_linux-x64_debug.yml/badge.svg) |

## Requirements
- cmake 3.23.x
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
- Install CMake that version upper then 3.23.x

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
   