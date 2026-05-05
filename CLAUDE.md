# CLAUDE.md — TCC: Paralelização do algoritmo Hash Chain

## 1. Project Overview

This repository is the codebase for the undergraduate thesis (TCC) of Arthur Patrocínio Neves at PUC Minas:
**"Paralelização do algoritmo Hash Chain para busca exata de strings"**.

**Goal.** Parallelize the **Hash Chain** exact string matching algorithm (Palmer, Faro, Scafiti — arXiv:2310.15711, SEA 2024) on a GPU using **OpenCL**, and compare the parallel version against the original sequential C implementation.

**Hardware target.** AMD Radeon RX 9060 XT (RDNA 4) on Windows. Code is portable C/C++17 + OpenCL 1.2 (host runs on Linux too for development).

**Scientific question.** When does the GPU implementation actually beat the sequential CPU version? The preliminary results already answer this: GPU wins consistently on natural-language text (English, up to ~5× speedup at m=128), but loses on small alphabets (DNA, Proteins) at small `m` because the H2D PCIe transfer of the 100 MB text dominates a kernel that already runs in milliseconds (memory-bound regime).

**Algorithm in one paragraph.** Hash Chain (the `hc8` variant — Q=8) builds a 4096-entry Bloom-style filter `F` over q-grams of the pattern, where each entry stores a fingerprint of the *next* q-gram in the chain. The search reads a q-gram from the text, looks it up in `F`, and if the bit is set walks back chain-by-chain (Q bytes at a time) verifying linked fingerprints. On a chain match it falls back to `memcmp` against the pattern. The shift on miss is `m - Q + 1` (sublinear on average).

## 2. Repository Layout

```
TCC/
├── HashChain/                       # (mostly empty stub — ignore)
│   └── src/Experimental/AnchorHashChain/   (empty)
│
├── HashChainParallelized/           # ← THE ACTUAL PROJECT
│   ├── main.cpp                     # Host code: CLI, OpenCL setup, CPU reference impl, benchmarking
│   ├── kernel.cl                    # Device code: hc8_search_kernel (the parallel search)
│   ├── rodar_testes.bat             # Windows build+run script (MSVC or g++ fallback)
│   ├── analise_benchmarks.py        # Post-processing: reads results-csv/, emits PDFs in figuras-benchmark/
│   ├── README.md                    # Original Palmer/Faro README about the HashChain family
│   ├── .gitignore                   # main.exe, main.obj, .vs/
│   │
│   ├── src/                         # ← ORIGINAL SEQUENTIAL C CODE (the reference being parallelized)
│   │   ├── HashChain/               # The canonical algorithm — hc1.c … hc8.c (Q=1..8)
│   │   │   ├── hc8.c                # ← The 8-byte q-gram variant we actually parallelize
│   │   │   └── include/             # define.h, main.h, timer.h, stats.h, AUTOMATON.h, GRAPH.h, log2.h
│   │   ├── LinearHashChain/         # Linear-WFR variant (lhc1..lhc8) — not yet parallelized
│   │   ├── SentinelHashChain/       # Sentinel hack variant (shc1..shc8) — not parallelized
│   │   ├── WeakerHashChain/         # No re-scan variant (whc1..whc8) — not parallelized
│   │   └── Experimental/            # Research variants from the original authors
│   │       ├── AnchorHashChain/     # ahc1..ahc8
│   │       ├── FastHashChain/       # fhc1..fhc8
│   │       ├── HashChainQVerify/    # hc2-qverify..hc8-qverify
│   │       ├── LinearHashChain/     # lhc2, lhc3-{full,kmp,scan,worstcase}
│   │       ├── RollingHashChain/    # rhc1..rhc8
│   │       └── vTestHashChain/      # hc1-vtest..hc8-vtest
│   │
│   ├── bd/                          # ← Pizza & Chilli corpus (100 MB each). DO NOT explore/read this folder.
│   │                                #   Files: dna.100MB, english.100MB, proteins.100MB
│   │
│   ├── SEA2024-benchmarks/          # Reference benchmark results from the SEA 2024 paper
│   │   ├── pizza-dna/               # EXP*.csv/html/txt — measurements, statistics, report
│   │   ├── pizza-english/
│   │   └── pizza-protein/
│   │
│   ├── results-csv/                 # CSVs produced by main.exe (cpu/gpu, per-pattern-len)
│   │                                #   Naming: benchmark-example_<base>(-extra-<m>)?-(cpu|gpu)_<UTC>.csv
│   ├── figuras-benchmark/           # PDFs produced by analise_benchmarks.py
│   │                                #   fig1_speedup.pdf, fig2_tempos_absolutos.pdf, fig3_memoria_gpu.pdf
│   │
│   └── .git/                        # The Parallelized folder is the only git repo
│
└── .venv/                           # Python venv for analise_benchmarks.py (pandas, matplotlib, seaborn)
```

`src/` is the legacy SMART-tool C code (Faro/Lecroq research framework). It is the *reference* — the parallel work lives at the top level (`main.cpp` + `kernel.cl`), not inside `src/`.

## 3. Build, Run, Test

All commands assume CWD = `HashChainParallelized/`.

### 3.1 Build

**Windows (primary target — what the thesis runs on):**
```
rodar_testes.bat
```
Tries MSVC (`cl /O2 /std:c++17 /EHsc main.cpp ... OpenCL.lib`) first, falls back to `g++ -std=c++17 -O2 main.cpp -lOpenCL`. Set `OPENCL_SDK=C:\Program Files (x86)\OCL_SDK_Light` (or equivalent) if headers/libs are not on the default path. Output: `hc8_opencl_test.exe`.

**Linux (development):**
```
g++ -std=c++17 -O2 main.cpp -lOpenCL -o hc8_opencl_test
```
Requires an OpenCL ICD loader (`ocl-icd-libopencl1`) and an AMD/Intel/NVIDIA platform installed. The host code prefers an AMD platform if present, otherwise picks the first GPU.

### 3.2 Run

`main.exe` (or `./hc8_opencl_test`) accepts:

| Flag | Default | Meaning |
|---|---|---|
| `--run-examples` | off | Run the canned 7 scenarios over `bd/*.100MB` (DNA m=16, English m=8/16/32/64/128, Proteins m=32) |
| `--text <path>` | — | Single-run text file |
| `--pattern <str>` | — | Literal pattern bytes |
| `--pattern-len <n>` | — | Sample a pattern of length n from the text instead |
| `--pattern-offset <n>` | text/3 | Where to sample the pattern from |
| `--chunk-size <n>` | 262144 | Bytes per work-item (global = ceil(N/chunk)) |
| `--max-results <n>` | 1000000 | Capacity of the device results buffer |
| `--csv-dir <dir>` | `results-csv` | Where CSVs are written |
| `--no-csv` | (csv on) | Disable CSV export |

Each run prints CPU and GPU timings, an indices-preview, and a `[COMPARE]` block (correctness diff + speedup). Both versions always run, so correctness is validated implicitly on every invocation.

### 3.3 Analysis pipeline

```
source ../.venv/bin/activate          # pandas, matplotlib, seaborn
python analise_benchmarks.py --input results-csv --out-dir figuras-benchmark
```
Produces `fig1_speedup.pdf`, `fig2_tempos_absolutos.pdf`, `fig3_memoria_gpu.pdf` plus a Markdown summary table on stdout.

### 3.4 Tests

There is **no unit test framework**. Correctness validation is structural:

1. `run_hc8_cpu` and `run_hc8_gpu` are run on the same `(text, pattern)` and `compare_and_print` checks `same_total`, `same_overflow`, `same_indices`. Any divergence is a regression.
2. Both implementations sort their indices before comparison, so order is not part of the contract.
3. `bd/*.100MB` files act as the regression corpus.

## 4. Architectural Patterns & Conventions

### 4.1 Algorithm constants (must stay in sync between host and device)

`main.cpp` `namespace { … }` and `kernel.cl` both hard-code:
```
ALPHA = 12   Q = 8   S = ALPHA/Q = 1   ASIZE = 4096   TABLE_MASK = 0xFFF
Q2 = 16   END_FIRST_QGRAM = 7   END_SECOND_QGRAM = 15
LINK_HASH(H) = 1u << (H & 0x1F)
```
**These two definitions must match exactly.** If you change Q or ALPHA, change both.

### 4.2 Host/Device split (the thesis's chosen model)

- **CPU (host) does:** read file, build `F` (Bloom filter) and `Hm` (final hash) via `preprocessing_hc8`, set up OpenCL, allocate buffers, enqueue H2D writes, launch kernel, read results back, sort, compare.
- **GPU (device) does:** the search loop only. It receives `text`, `pattern` (for verification), `F`, `Hm`, `chunk_size`, `num_chunks`, `max_results`.
- **Why split there:** preprocessing is small (~4 KB filter, O(m) work) and inherently sequential — Amdahl says don't bother. The text scan is the embarrassingly-parallel part. This matches the "hybrid CPU/GPU" pattern argued for in the thesis literature review.

### 4.3 Data-parallel decomposition

- Text is partitioned into `chunk_size` blocks (default 256 KiB). One work-item processes one chunk.
- Boundary handling = **overlap by m-1**: each work-item's `max_pos = min(n-1, core_end + m - 2)` so a match starting near the end of a chunk is still found.
- A match is *recorded* only when `match_start ∈ [chunk_start, core_end)` — this prevents a match from being recorded twice (once by the chunk that owns it, once by the previous chunk's overlap region).
- **Local memory optimization:** `F_local` (4 KB) is loaded cooperatively by all work-items in a workgroup before the search loop. Reads in the hot loop hit local memory, not global. This is critical for AMD wavefront throughput.
- Workgroup size = 256, fixed. Global size is rounded up to a multiple of `local_size`.

### 4.4 Result aggregation

- `results[]` (device buffer, sized `max_results * sizeof(int)`), `result_count` (atomic counter), `overflow_flag` (set when capacity exceeded).
- Each match does `atomic_inc(result_count)`; if `idx < max_results` the index is written, otherwise overflow is OR'd in. **No order guarantee** — host sorts after readback.

### 4.5 Profiling discipline

- Command queue is created with `CL_QUEUE_PROFILING_ENABLE`.
- Every enqueue captures a `cl_event`; `event_millis` reads `CL_PROFILING_COMMAND_START/END`.
- Host reports H2D, kernel, D2H **separately** and a derived `gpu_total = h2d + kernel + d2h`. The thesis's central finding (PCIe is the bottleneck for small-alphabet small-m runs) depends on this separation — never collapse them into a single wall-clock number.

### 4.6 CSV schema (consumed by `analise_benchmarks.py`)

GPU CSV columns:
`run_tag, text_file, pattern_len, chunk_size, max_results, found_total, returned_indices, overflow, h2d_ms, kernel_ms, d2h_ms, gpu_total_ms, index_order, index_value`

CPU CSV columns:
`run_tag, text_file, pattern_len, max_results, found_total, returned_indices, overflow, preprocessing_ms, search_ms, index_order, index_value`

If `indices` is empty a single row with empty `index_order/index_value` is still emitted (so every run produces ≥1 row). The Python script joins CPU↔GPU on `(text_file, pattern_len)` and computes `speedup = cpu_total_ms / gpu_total_ms`.

### 4.7 Naming & language

- Code is in **C++17** (host) and **OpenCL C 1.2** (device, `-cl-std=CL1.2`). The reference C code in `src/` is C99-ish.
- Comments and CLI strings are in **Portuguese** (PT-BR) — error messages, log tags, and the .bat script. Variable names and function names are in **English**. Keep this convention.
- Use `snake_case` for functions and locals (`run_hc8_gpu`, `chain_hash8`, `preprocessing_hc8`); `PascalCase` for structs (`OclContext`, `RunResult`, `CpuRunResult`); `SCREAMING_CASE` for constants.

### 4.8 OpenCL error handling

Every CL call goes through `check_cl(err, "where")`, which throws `std::runtime_error`. `cl_error_to_string` only knows a handful of codes — extend it if you hit a new one rather than swallowing the error. Build failures dump `CL_PROGRAM_BUILD_LOG` into the exception message.

## 5. Anti-Patterns — Things NOT to do

1. **Do not modify files under `bd/`** and do not enumerate or read its contents. The corpus files are large (100 MB each) and are not relevant to source changes. The user explicitly excluded this folder.
2. **Do not edit the legacy code in `src/`.** It is the reference implementation (Palmer/Faro SMART tool) — the parallelization lives in `main.cpp` + `kernel.cl` at the top level. Treat `src/` as read-only documentation of the algorithm being parallelized.
3. **Do not desync ALPHA/Q between `main.cpp` and `kernel.cl`.** They are duplicated by design (host needs them as `constexpr`, device needs them as `#define`). If you change one, change both — and update `link_hash`, `chain_hash8`, `chain_hash8_global` consistently.
4. **Do not collapse the GPU timing into a single number.** The thesis's argument hinges on showing H2D vs kernel vs D2H separately. `gpu_total_ms` is fine to *also* report; never replace the breakdown with it.
5. **Do not "optimize" by skipping the CPU run when measuring.** Every invocation runs both CPU and GPU and uses `compare_and_print` for correctness — that's the regression harness. Removing the CPU path silently disables correctness checking.
6. **Do not swallow `--max-results` overflow.** When `overflow_flag` is set the run is invalid for indices comparison; preserve the flag in the CSV (`overflow` column) and surface it in the comparison.
7. **Do not change the OpenCL std flag without reason.** `-cl-std=CL1.2` is intentional for AMD ROCm/PAL compatibility on RDNA 4. Going to 2.0 may break the build on some drivers.
8. **Do not introduce non-deterministic ordering in match recording.** Work-items are inherently unordered, so always sort `indices` host-side before comparing or writing CSV. Don't rely on insertion order anywhere downstream.
9. **Do not write to global memory for every q-gram lookup.** The `F_local` cooperative load + `barrier(CLK_LOCAL_MEM_FENCE)` is the hot-path optimization. Refactors that re-introduce `F_global` reads inside the inner loop will tank performance and invalidate prior benchmark numbers.
10. **Do not regenerate or delete `results-csv/*.csv` from prior runs casually.** Those CSVs are the data behind figures already cited in the thesis (Figures 1 and 2 in TCC1). Add new runs alongside; archive before overwriting.
11. **Do not commit `main.exe`, `main.obj`, or `.vs/`** — already in `.gitignore`. Don't commit benchmark artifacts (`figuras-benchmark/*.pdf`) without intent — they are derived from CSVs.
12. **Do not add a README/doc file unless asked.** The existing `README.md` is the upstream Palmer/Faro README and should be left intact as attribution.
13. **Do not change Portuguese log/CLI strings to English** (or vice versa) unless asked. The audience is the PUC Minas TCC committee, and screenshots of stdout/figures may already be embedded in the thesis text.
14. **Do not run benchmarks in parallel with other GPU workloads.** Profiling numbers are only comparable across runs when the GPU is otherwise idle; the kernel timings reported in TCC1 assume that.
15. **Do not "fix" the chunk overlap by extending the recording window.** A match must be claimed by exactly one chunk (the one that owns its `match_start`). Loosening the `match_start ∈ [chunk_start, core_end)` guard reintroduces duplicates.