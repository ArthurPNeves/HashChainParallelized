# Paralelização do algoritmo Hash Chain em GPU via OpenCL

Trabalho de Conclusão de Curso (TCC) — **Arthur Patrocínio Neves**, Pontifícia
Universidade Católica de Minas Gerais (PUC Minas).

Este repositório contém a **paralelização em GPU** do algoritmo de busca exata de
strings **Hash Chain** (variante `hc8`, Q = 8), proposto por Palmer, Faro e
Scafiti (SEA 2024). A versão paralela usa o padrão aberto **OpenCL** sob um
modelo híbrido CPU/GPU, e é comparada contra a implementação sequencial original
em CPU sobre o *Pizza & Chilli Corpus* de 100 MB.

## O algoritmo em um parágrafo

O Hash Chain constrói um filtro estilo Bloom de 4096 entradas sobre q-gramas do
padrão, onde cada entrada guarda a impressão digital do q-grama *seguinte* na
cadeia. A busca lê um q-grama do texto, consulta o filtro e, se a entrada estiver
marcada, percorre a cadeia para trás (Q bytes por vez) validando as impressões
encadeadas; só então faz a verificação final por `memcmp`. O deslocamento em caso
de incompatibilidade é `m - Q + 1` (sublinear no caso médio).

**Modelo híbrido adotado:** a CPU (host) faz o pré-processamento leve (constrói o
filtro `F` e o hash final `Hm`) e a GPU (device) executa apenas o laço de busca,
massivamente paralelo, sobre o texto particionado em blocos de 256 KiB
(um *work-item* por bloco, com overlap de `m-1` para tratar fronteiras).

## Estrutura do repositório

```
HashChainParallelized/
├── main.cpp                 # Código host (C++17): CLI, OpenCL, impl. sequencial de referência, benchmarking
├── kernel.cl               # Código device (OpenCL C 1.2): kernel de busca paralela
├── rodar_tcc_final.bat     # Build + coleta final (31 reps) + figuras + verificação (Windows)
├── rodar_testes.bat        # Script antigo de build + smoke test (Windows)
├── CppProperties.json      # Configuração de build do Visual Studio
├── analise_benchmarks.py   # Pós-processamento: lê results-csv/ e gera os PDFs
├── results-csv/            # Dados dos benchmarks (a coleta final fica em results-csv/final/)
└── figuras-benchmark/      # Figuras geradas (fig1_speedup.pdf … fig5_colisoes.pdf)
```

> **Constantes do algoritmo** (`ALPHA=12`, `Q=8`, `ASIZE=4096`, …) estão
> duplicadas por design em `main.cpp` e `kernel.cl` e **devem permanecer em
> sincronia** — host precisa delas como `constexpr`, device como `#define`.

## Pré-requisitos

- Compilador C++17 (MSVC no Windows; `g++` no Linux).
- SDK/headers OpenCL e um *ICD loader* (`OpenCL.lib` no Windows;
  `ocl-icd-libopencl1` no Linux) com uma plataforma instalada (AMD/Intel/NVIDIA).
  O código prefere uma plataforma AMD se houver; senão, usa a primeira GPU.
- Python 3 com `pandas`, `matplotlib`, `seaborn`, `numpy` (para a análise).

## Corpus (não incluído no repositório)

Os arquivos de 100 MB do *Pizza & Chilli Corpus* **não** são versionados (excedem
o limite do GitHub). Baixe-os em <http://pizzachili.dcc.uchile.cl/texts.html> e
coloque-os em uma pasta `bd/` na raiz do projeto:

```
bd/dna.100MB
bd/english.100MB
bd/proteins.100MB
```

## Build

**Windows (alvo principal do TCC):**

```
rodar_tcc_final.bat
```

Faz o build (MSVC `cl /O2 /std:c++17 /EHsc … OpenCL.lib`, com *fallback* para
`g++ -std=c++17 -O2 … -lOpenCL`), roda a varredura completa, valida a regressão
CPU×GPU e gera as figuras. Defina `OPENCL_SDK` se os headers/libs não estiverem
no caminho padrão. Binário gerado: `hc8_opencl_test.exe`.

**Linux (desenvolvimento):**

```
g++ -std=c++17 -O2 main.cpp -lOpenCL -o hc8_opencl_test
```

## Execução

```
./hc8_opencl_test --run-examples --repeat 30
```

Flags principais:

| Flag | Padrão | Significado |
|---|---|---|
| `--run-examples` | off | Roda os 15 cenários sobre `bd/*.100MB` (3 bases × m = 8/16/32/64/128). |
| `--repeat <n>` | 30 | Repetições medidas por cenário (regime *warm*). Reporta mediana + CV%. |
| `--warmup <n>` | 1 | Repetições de aquecimento descartadas. |
| `--text <path>` | — | Arquivo de texto para execução única. |
| `--pattern <str>` | — | Padrão literal. |
| `--pattern-len <n>` | — | Amostra um padrão de tamanho `n` do próprio texto. |
| `--chunk-size <n>` | 262144 | Bytes por *work-item*. |
| `--csv-dir <dir>` | `results-csv` | Onde gravar os CSVs (use `results-csv/final` na coleta final). |

A corretude (CPU × GPU: `same_total ∧ same_overflow ∧ same_indices`) é checada em
**toda** repetição; qualquer divergência imprime `[REGRESSION-FAIL]`.

## Análise dos benchmarks

```
python analise_benchmarks.py --input results-csv/final --out-dir figuras-benchmark --strict
```

Lê o `timings_*.csv` mais recente do diretório indicado e gera
`fig1_speedup.pdf`, `fig2_tempos_absolutos.pdf`, `fig3_memoria_gpu.pdf`,
`fig4_vazao.pdf` e `fig5_colisoes.pdf`, além de tabelas em Markdown. `--strict`
falha se algum registro violar `kernel+h2d+d2h ≤ wall`.

## Resultados (resumo)

Avaliação em uma **AMD Radeon RX 9060 XT (RDNA 4)**, corpus de 100 MB:

- **Regime *warm*** (texto residente em VRAM): a GPU venceu a CPU sequencial em
  **todos os 15 cenários**, com *speedups* de **1,33× a 3,45×**.
- **Regime *cold*** (consulta única isolada, cobrando a leitura de disco dos dois
  lados): a GPU perde por margem estreita (**0,92×–0,98×**), pois o *upload* único
  RAM→VRAM, pago só pela GPU e não amortizado, supera sua vantagem por consulta.

Ou seja, a paralelização compensa sempre que o custo único de carga do texto é
amortizado sobre múltiplas consultas (serviços de indexação, *pipelines* de
bioinformática).

## Créditos

- **Algoritmo original Hash Chain:** Matthew N. Palmer, Simone Faro e Stefano
  Scafiti — *Efficient Exact Online String Matching Through Linked Weak Factors*,
  SEA 2024 ([arXiv:2310.15711](https://arxiv.org/abs/2310.15711),
  [DROPS/LIPIcs](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2024.24)).
- **Paralelização em GPU/OpenCL e este TCC:** Arthur Patrocínio Neves (PUC Minas).

## Agradecimentos

Agradeço à PUC Minas pela formação e pelo apoio. Registro ainda o uso das
ferramentas de inteligência artificial **Claude Opus 4.8** e **4.7** (Anthropic)
como apoio ao desenvolvimento do software e à escrita e formatação do texto; as
decisões metodológicas, a análise dos resultados e a revisão final são de
responsabilidade do autor.
