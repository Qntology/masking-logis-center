@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"

:: [AMD ROCm/HIP]
set "CANDLE_HIP=1"
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"

cd src-tauri

rem Add DirectStorage DLLs to PATH
set "PATH=%PATH%;%CD%\microsoft.direct3d.directstorage.1.3.0\native\bin\x64"

echo [BUILD] Compiling and bundling Tauri application (Release and UTF-8 Mode)...

:: cargo build 대신 cargo tauri build를 사용하여 .msi 번들링까지 수행합니다.
:: 기존 컴파일 캐시를 재사용하기 위해 -- --features cuda 옵션을 그대로 유지합니다.
cargo tauri build -- --features cuda