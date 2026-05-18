@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
chcp 65001 > nul
set "PYTHONIOENCODING=utf-8"
set "LC_ALL=C.UTF-8"

:: [AMD ROCm/HIP]
set "CANDLE_HIP=1"
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"

cd src-tauri

rem Add DirectStorage DLLs to PATH
set "PATH=%PATH%;%CD%\microsoft.direct3d.directstorage.1.3.0\native\bin\x64"

echo [RUN] Compiling and starting application (UTF-8 Mode)...
cargo run --features cuda --features cuda > ..\run_result.txt 2>&1
type ..\run_result.txt
