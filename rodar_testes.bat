@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Script Windows para compilar e executar testes do HC8 OpenCL
REM - Compila: main.cpp + OpenCL
REM - Executa: exemplos (dna/english/proteins) e um teste individual opcional
REM ============================================================================

cd /d "%~dp0"
echo [INFO] Diretorio atual: %CD%

set "EXE=hc8_opencl_test.exe"
set "BUILD_OK=0"

REM --------------------------------------------------------------------------
REM 1) Tentativa com MSVC (cl.exe)
REM --------------------------------------------------------------------------
where cl >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [INFO] cl.exe encontrado. Tentando compilar com MSVC...

    REM Ajuste OPENCL_SDK para o caminho do seu SDK OpenCL no Windows, se necessario.
    REM Exemplo:
    REM set OPENCL_SDK=C:\Program Files (x86)\OCL_SDK_Light

    if defined OPENCL_SDK (
        set "OCL_INC=/I"%OPENCL_SDK%\include""
        set "OCL_LIB=/LIBPATH:"%OPENCL_SDK%\lib\x64""
        echo [INFO] Usando OPENCL_SDK=%OPENCL_SDK%
    ) else (
        set "OCL_INC="
        set "OCL_LIB="
        echo [WARN] OPENCL_SDK nao definido. Tentando include/lib padrao do sistema.
    )

    cl /nologo /O2 /std:c++17 /EHsc main.cpp !OCL_INC! /link !OCL_LIB! OpenCL.lib /OUT:%EXE%
    if %ERRORLEVEL%==0 (
        set "BUILD_OK=1"
    ) else (
        echo [WARN] Falha na compilacao com MSVC.
    )
)

REM --------------------------------------------------------------------------
REM 2) Fallback com g++ (MinGW/MSYS2)
REM --------------------------------------------------------------------------
if "%BUILD_OK%"=="0" (
    where g++ >nul 2>&1
    if %ERRORLEVEL%==0 (
        echo [INFO] Tentando compilar com g++...

        REM Se necessario, defina OPENCL_SDK para ajudar no include/lib:
        REM set OPENCL_SDK=C:\Program Files (x86)\OCL_SDK_Light

        if defined OPENCL_SDK (
            g++ -std=c++17 -O2 main.cpp -I"%OPENCL_SDK%\include" -L"%OPENCL_SDK%\lib\x64" -lOpenCL -o %EXE%
        ) else (
            g++ -std=c++17 -O2 main.cpp -lOpenCL -o %EXE%
        )

        if %ERRORLEVEL%==0 (
            set "BUILD_OK=1"
        ) else (
            echo [ERROR] Falha na compilacao com g++.
        )
    ) else (
        echo [ERROR] Nem cl.exe nem g++ foram encontrados no PATH.
    )
)

if "%BUILD_OK%"=="0" (
    echo.
    echo [FALHA] Nao foi possivel compilar %EXE%.
    echo [DICA] Instale Build Tools (MSVC) ou MinGW e configure OPENCL_SDK, se preciso.
    exit /b 1
)

echo.
echo [OK] Compilacao concluida: %EXE%

REM --------------------------------------------------------------------------
REM 3) Execucao dos exemplos benchmark (bd/*.100MB)
REM --------------------------------------------------------------------------
echo.
echo [INFO] Executando exemplos (benchmark) com exportacao CSV...
%EXE% --run-examples --chunk-size 262144 --max-results 1000000 --csv-dir results-csv\windows
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Execucao dos exemplos falhou.
    exit /b 1
)

REM --------------------------------------------------------------------------
REM 4) Execucao adicional (teste individual)
REM --------------------------------------------------------------------------
echo.
echo [INFO] Executando teste individual (dna)...
%EXE% --text bd/dna.100MB --pattern-len 32 --pattern-offset 1000000 --chunk-size 262144 --max-results 1000000 --csv-dir results-csv\windows
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Teste individual falhou.
    exit /b 1
)

echo.
echo [SUCESSO] Todos os testes finalizaram.
echo [INFO] CSVs em: %CD%\results-csv\windows
exit /b 0
