# INSTRUÇÕES — Rodar os benchmarks finais do TCC (máquina com GPU AMD)

> **Para o chat do Claude que abrir aqui:** esta sessão roda na máquina de um amigo, que tem a
> **GPU AMD Radeon RX 9060 XT (RDNA 4)** em **Windows**. O objetivo é **só coletar os benchmarks
> finais** do Hash Chain (HC8) já com as correções metodológicas aplicadas, gerar as figuras e
> verificar a sanidade. **Leia primeiro o `CLAUDE.md`** (na pasta acima) — ele tem o contexto do
> projeto e os anti-padrões. Todo o código já foi preparado; em geral basta **rodar um script**.

## O que mudou em relação à versão anterior (por que rerodar)

A coleta anterior tinha defeitos que estas mudanças corrigem:
- **C1/C4** — a Tabela 1 vinha de um `rglob` que misturava CSVs de lotes antigos e tirava média entre eles
  (tempos saíam pela metade; `kernel` chegava a passar do `wall`). Agora o `main.cpp` grava **um único
  `timings_*.csv`** já com mediana+CV% por cenário, e o `analise_benchmarks.py` lê **apenas o mais recente**
  do diretório do run final (`results-csv\final`), com um `--strict` que aborta se `kernel+h2d+d2h > wall`.
- **C2** — agora há **31 repetições** (1 aquecimento + 30 amostras) por cenário; reporta-se **mediana e CV%**.
- **C3** — no regime *cold*, a **leitura do arquivo (100 MB) é cobrada também da CPU** (antes só da GPU),
  tornando a comparação justa.
- O regime **cold/warm** vira coluna explícita no CSV.

## Pré-checagem (rode cada comando; tudo deve passar)

No terminal, dentro de `...\TCC\HashChainParallelized`:

```bat
cd /d <caminho>\TCC\HashChainParallelized
dir bd\*.100MB            REM deve listar dna.100MB, english.100MB, proteins.100MB
where cl                 REM MSVC; se vazio, tente:  where g++
echo %OPENCL_SDK%         REM opcional: caminho do SDK OpenCL, se headers/lib nao estiverem no PATH
python --version          REM ou:  py --version
```

- Se `where cl` e `where g++` estiverem ambos vazios: abrir o **"x64 Native Tools Command Prompt for VS"**
  (traz o `cl.exe` no PATH) **ou** instalar MinGW.
- Se a build reclamar de `CL/cl.h` ou `OpenCL.lib`: `set OPENCL_SDK=C:\caminho\do\sdk` (geralmente o SDK da AMD
  ou o `OCL_SDK_Light`) e rode de novo.

## Caminho feliz (uma tecla)

```bat
rodar_tcc_final.bat
```

Esse script: confere o corpus → compila (`main.cpp` → `hc8_opencl_test.exe`, MSVC com *fallback* g++) →
roda a bateria (`--run-examples --repeat 30 --warmup 1`, gravando em `results-csv\final`) → checa regressão →
gera as 3 figuras e roda a verificação `--strict`.

**Sucesso =** terminar com `[SUCESSO]`. Confira no log:
- `[OK] Regressao: CPU e GPU concordaram em todos os cenarios.` (nenhum `[REGRESSION-FAIL]`).
- `[VERIFICAÇÃO] OK` no passo do Python (sem `kernel > wall`).
- A `Tabela - regime WARM/COLD` foi impressa no final.

**Onde ficam as saídas:**
- `results-csv\final\timings_<UTC>.csv` ← fonte da nova Tabela 1.
- `results-csv\final\stdout.txt` ← log completo.
- `figuras-benchmark\fig1_speedup.pdf`, `fig2_tempos_absolutos.pdf`, `fig3_memoria_gpu.pdf`.

## Fallback manual (se o `.bat` falhar em algum passo)

```bat
REM 1) Build (MSVC)
cl /nologo /O2 /std:c++17 /EHsc main.cpp /I"%OPENCL_SDK%\include" /link /LIBPATH:"%OPENCL_SDK%\lib\x64" OpenCL.lib /OUT:hc8_opencl_test.exe
REM    ... ou g++:
g++ -std=c++17 -O2 main.cpp -I"%OPENCL_SDK%\include" -L"%OPENCL_SDK%\lib\x64" -lOpenCL -o hc8_opencl_test.exe

REM 2) Run
hc8_opencl_test.exe --run-examples --repeat 30 --warmup 1 --chunk-size 262144 --max-results 1000000 --csv-dir results-csv\final

REM 3) Figuras + verificacao (use o venv se existir; senao instale as libs)
..\.venv\Scripts\python.exe -m pip install pandas matplotlib seaborn numpy
..\.venv\Scripts\python.exe analise_benchmarks.py --input results-csv\final --out-dir figuras-benchmark --strict
REM    ... ou com o Python do sistema:
python -m pip install pandas matplotlib seaborn numpy
python analise_benchmarks.py --input results-csv\final --out-dir figuras-benchmark --strict
```

## Troubleshooting

- **`CL_DEVICE_NOT_FOUND`** → driver AMD/OpenCL não visível; confirme a GPU e o driver. Em alguns casos o
  ICD só aparece fora do WSL — rode no **Windows nativo**, não no WSL.
- **Falha de build do kernel** (mensagem com log do `clBuildProgram`) → leia o log; não altere `-cl-std=CL1.2`
  (anti-padrão #7 do CLAUDE.md) nem as constantes `ALPHA/Q` sem sincronizar `main.cpp` e `kernel.cl`.
- **CV% alto (`[AVISO] ... cv > 15%`)** → algo concorria pela GPU. Feche outros apps de GPU e rode de novo
  (anti-padrão #14: GPU ociosa durante a medição).
- **`overflow=1`** → algum cenário passou de `--max-results`; aumente `--max-results` e rerode.
- **Python sem libs** → `pip install pandas matplotlib seaborn numpy` (o CSV de timings já fica salvo de qualquer forma).

## Regras herdadas do `CLAUDE.md` (não violar)

- **Não** ler/alterar `bd/` nem `src/` (somente leitura/referência).
- **Não** rodar a análise com `rglob` sobre toda a `results-csv/` — sempre `--input results-csv\final`.
- **Não** apagar CSVs antigos; o run novo escreve em `results-csv\final` (o script de análise usa o `timings_*.csv` mais recente).
- Manter as strings de log/CLI em **PT-BR**.
- **Não** rodar benchmark junto com outras cargas de GPU.

## Depois de rodar (o que trazer / reportar)

1. Cole de volta neste chat (ou leve os arquivos) a **`Tabela - regime WARM`** e a **`Tabela - regime COLD`**
   impressas, mais os 3 PDFs e o `timings_*.csv`.
2. De volta em casa, esses números atualizam: **Tabela 1**, **Figuras 1–3** e o texto de **§4.3/§5** do `tcc.tex`
   (agora com mediana±CV%, regime explícito e o *cold* justo).
