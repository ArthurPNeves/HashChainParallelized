@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  Coleta FINAL dos benchmarks do TCC (Hash Chain HC8 OpenCL)
REM  Faz tudo de uma vez: build -> run (31 reps) -> figuras -> verificacao.
REM
REM  Pre-requisitos (ja devem estar prontos na maquina):
REM    - bd\dna.100MB, bd\english.100MB, bd\proteins.100MB
REM    - MSVC (cl.exe) OU MinGW (g++) no PATH
REM    - Driver/SDK OpenCL da AMD (defina OPENCL_SDK se headers/lib nao estiverem no PATH)
REM    - Python 3 com pandas/matplotlib/seaborn (ou venv em ..\.venv)
REM
REM  Saidas:
REM    results-csv\final\timings_<UTC>.csv  (fonte da Tabela 1)
REM    results-csv\final\stdout.txt         (log completo + verificacao de regressao)
REM    figuras-benchmark\fig1/2/3.pdf
REM ============================================================================

cd /d "%~dp0"
echo [INFO] Diretorio: %CD%

set "EXE=hc8_opencl_test.exe"
set "OUTDIR=results-csv\final"
set "BUILD_OK=0"

REM ---------------------------------------------------------------------------
REM 0) Pre-checagem do corpus
REM ---------------------------------------------------------------------------
for %%F in (dna.100MB english.100MB proteins.100MB) do (
    if not exist "bd\%%F" (
        echo [FALHA] Arquivo do corpus ausente: bd\%%F
        echo [DICA]  Copie os 3 arquivos do Pizza ^& Chilli Corpus para a pasta bd\.
        exit /b 1
    )
)
echo [OK] Corpus bd\*.100MB presente.

REM ---------------------------------------------------------------------------
REM 1) Build: MSVC -> fallback g++
REM ---------------------------------------------------------------------------
where cl >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [INFO] Compilando com MSVC (cl.exe)...
    if defined OPENCL_SDK (
        set "OCL_INC=/I"%OPENCL_SDK%\include""
        set "OCL_LIB=/LIBPATH:"%OPENCL_SDK%\lib\x64""
    ) else (
        set "OCL_INC="
        set "OCL_LIB="
    )
    cl /nologo /O2 /std:c++17 /EHsc main.cpp !OCL_INC! /link !OCL_LIB! OpenCL.lib /OUT:%EXE%
    if !ERRORLEVEL!==0 ( set "BUILD_OK=1" ) else ( echo [WARN] Falha no MSVC. )
)

if "%BUILD_OK%"=="0" (
    where g++ >nul 2>&1
    if !ERRORLEVEL!==0 (
        echo [INFO] Compilando com g++...
        if defined OPENCL_SDK (
            g++ -std=c++17 -O2 main.cpp -I"%OPENCL_SDK%\include" -L"%OPENCL_SDK%\lib\x64" -lOpenCL -o %EXE%
        ) else (
            g++ -std=c++17 -O2 main.cpp -lOpenCL -o %EXE%
        )
        if !ERRORLEVEL!==0 ( set "BUILD_OK=1" ) else ( echo [ERROR] Falha no g++. )
    ) else (
        echo [ERROR] Nem cl.exe nem g++ encontrados no PATH.
    )
)

if "%BUILD_OK%"=="0" (
    echo [FALHA] Build de %EXE% falhou. Veja as mensagens acima e o INSTRUCOES_TESTES_AMIGO.md.
    exit /b 1
)
echo [OK] Build concluido: %EXE%

REM ---------------------------------------------------------------------------
REM 2) Run da bateria final (31 reps = 1 aquecimento + 30 amostras)
REM    Escreve em diretorio dedicado; o log vai para stdout.txt.
REM ---------------------------------------------------------------------------
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
echo [INFO] Rodando a bateria (pode levar alguns minutos)... log em %OUTDIR%\stdout.txt
%EXE% --run-examples --repeat 30 --warmup 1 --chunk-size 262144 --max-results 1000000 --csv-dir "%OUTDIR%" > "%OUTDIR%\stdout.txt" 2>&1
set "RUN_RC=!ERRORLEVEL!"
type "%OUTDIR%\stdout.txt"

if not "%RUN_RC%"=="0" (
    echo [FALHA] A execucao do benchmark retornou erro (%RUN_RC%). Veja %OUTDIR%\stdout.txt.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM 3) Gate de regressao (correcao CPU vs GPU)
REM ---------------------------------------------------------------------------
findstr /C:"REGRESSION-FAIL" "%OUTDIR%\stdout.txt" >nul 2>&1
if !ERRORLEVEL!==0 (
    echo [FALHA] Houve [REGRESSION-FAIL]: a GPU divergiu da CPU em algum cenario. NAO use estes dados.
    exit /b 1
)
echo [OK] Regressao: CPU e GPU concordaram em todos os cenarios.

REM ---------------------------------------------------------------------------
REM 4) Figuras + verificacao de sanidade (--strict)
REM ---------------------------------------------------------------------------
set "PY=python"
if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"

echo [INFO] Gerando figuras e verificando sanidade...
"%PY%" analise_benchmarks.py --input "%OUTDIR%" --out-dir figuras-benchmark --strict
if not "%ERRORLEVEL%"=="0" (
    echo [WARN] A analise/verificacao falhou. Causas comuns:
    echo        - faltam libs Python: "%PY%" -m pip install pandas matplotlib seaborn numpy
    echo        - inconsistencia detectada pelo --strict ^(kernel ^> wall^): investigar antes de usar.
    echo [INFO] O CSV de timings ja foi salvo em %OUTDIR% e pode ser analisado depois.
    exit /b 1
)

echo.
echo [SUCESSO] Coleta final concluida.
echo   - CSV de timings : %CD%\%OUTDIR%\
echo   - Log completo   : %CD%\%OUTDIR%\stdout.txt
echo   - Figuras (PDF)  : %CD%\figuras-benchmark\
echo.
echo [PROXIMO PASSO] Copie a "Tabela - regime WARM/COLD" impressa acima (e os 3 PDFs e o
echo   timings_*.csv) para atualizar a Tabela 1 e as Figuras 1-3 do tcc.tex.
exit /b 0
