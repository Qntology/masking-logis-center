@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MSVC_BIN=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
set "PYTHONIOENCODING=utf-8"

:: 실행 시 넘겨준 인자(cuda 또는 rocm)를 받습니다. 기본값은 cuda입니다.
set TARGET_GPU=%1
if "%TARGET_GPU%"=="" set TARGET_GPU=cuda

if "%TARGET_GPU%"=="cuda" (
    echo [ENV] Setting up CUDA Environment...
    set "NVCC_CCBIN=%MSVC_BIN%\cl.exe"
    set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
    set "CUDA_PATH=%CD%\src-tauri\dlls\cuda"
    set "CUDA_ROOT=%CD%\src-tauri\dlls\cuda"
    set "CUDA_TOOLKIT_ROOT_DIR=%CD%\src-tauri\dlls\cuda"
    set "PATH=%MSVC_BIN%;%CD%\src-tauri\dist_cuda;%CD%\src-tauri\dlls\cuda\bin;%CD%\src-tauri\dlls\cuda\lib;%CD%\src-tauri\dlls\cuda\include;%PATH%"
)

if "%TARGET_GPU%"=="rocm" (
    echo [ENV] Setting up AMD ROCm Environment...
    set "CANDLE_HIP=1"
    set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
    set "PATH=%MSVC_BIN%;%CD%\src-tauri\dist_rocm;%HIP_PATH%\bin;%PATH%"
)

cd src-tauri

echo [BUILD] Compiling and bundling Tauri application for %TARGET_GPU%...
cargo tauri build -- --features %TARGET_GPU%