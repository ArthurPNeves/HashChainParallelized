# Session summary — 2026-05-04

End-to-end log of what changed in the codebase during this session. Audience is future-you (or the next Claude instance handed this repo).

---

## 1. Context coming into the session

The thesis (TCC1) reported that the OpenCL GPU implementation of `hc8` won ~5× on English text and lost (~0.9×) on DNA/Proteins, blaming PCIe H2D. The CSVs in `HashChainParallelized/results-csv/` told a different (overly optimistic) story because `clGetEventProfilingInfo` was returning `h2d_ms = 0.000` for the 100 MB transfer — events were lying.

A previous Claude session (Linux laptop) had already written **Step 1** code changes — wall-clock GPU timing via `chrono` — and committed them, but had not run them. This session was opened on the Windows + AMD RX 9060 XT machine to verify Step 1 and continue the optimization plan.

The plan from `handoff.md`:

| Step | What | Files | Goal |
|---|---|---|---|
| 1 | Wall-clock GPU timing | `main.cpp`, `analise_benchmarks.py` | Honest measurement |
| 2 | Hoist `OclContext` + cache text uploads across examples | `main.cpp` | Eliminate per-call setup tax |
| 3 | Pinned-staging text upload | `main.cpp` | Faster + observable H2D |
| 4 | Tiled double-buffered upload | `main.cpp` + small kernel arg | Hide H2D behind kernel |
| 5 | Bump `ALPHA` 12 → 14/16 | `main.cpp` + `kernel.cl` | Reduce kernel work for small alphabets |

User success criterion: positive wall-speedup (>1×) consistently for DNA and Proteins.

---

## 2. What got done

### 2.1 Environment bring-up

- No OpenCL SDK on the machine; AMD Adrenalin only ships `OpenCL.dll` (the ICD), not headers/lib.
- Installed **OCL_SDK_Light** to `C:\Program Files (x86)\OCL_SDK_Light\` (manual zip extract). Layout: `include/CL/cl.h`, `lib/x86_64/OpenCL.lib`. The repo's `rodar_testes.bat` hardcodes `lib\x64`, so the .bat won't link out-of-the-box; we built manually instead via `cl.exe` direct invocation under VS18 Insiders' `vcvars64.bat`:
  ```
  cl /nologo /O2 /std:c++17 /EHsc main.cpp ^
     /I"C:\Program Files (x86)\OCL_SDK_Light\include" ^
     /link /LIBPATH:"C:\Program Files (x86)\OCL_SDK_Light\lib\x86_64" OpenCL.lib ^
     /OUT:hc8_opencl_test.exe
  ```
- Installed **Python 3.13** user-scope via `winget` and created `.venv` at the project root with `pandas`, `matplotlib`, `seaborn`, `numpy`. `analise_benchmarks.py` now runnable on Windows.

### 2.2 Step 1 — verify wall-clock measurement (already coded, just ran it)

Ran `--run-examples --csv-dir results-csv\step1`. All 7 baseline examples passed `same_indices: true`. The new wall numbers showed huge gaps vs the events:

| run | CPU search | GPU wall | GPU events | gap | speedup_wall |
|---|---:|---:|---:|---:|---:|
| dna m=16 | 23.3 | 1291.1 | 9.8 | 1281.4 | 0.018× |
| english m=64 | 5.6 | 110.8 | 2.2 | 108.6 | 0.051× |
| protein m=32 | 10.2 | 116.4 | 4.4 | 112.0 | 0.087× |
| english m=8 | 178.1 | 167.0 | 47.2 | 119.8 | 1.067× |

Interpretation:
- TCC1's 5× English wins were event-only numbers (kernel-time only). Honest wall-clock said GPU lost on **all** scenarios except `english m=8` (where CPU's 178 ms search amortizes the 120 ms setup tax).
- The ~110–130 ms gap on every example was per-call OpenCL `create_opencl(…)` (context, queue, build program, kernel) — not PCIe.
- The dna outlier (1.29 s wall on the first run) was first-process kernel JIT.

§5.5 verdict: Option B — H2D was hidden in events; setup tax dominates; recommend Steps 2 and 3.

### 2.3 Step 2 — context hoist + text cache (`main.cpp` only)

Edits to `run_benchmark_examples`:
- One `create_opencl(…)` before the loop. Setup time printed once (`[SETUP] OpenCL context+kernel build (uma vez): 223 ms`).
- `std::map<std::string, CachedText>` keyed by file path. First example to use a given text reads it from disk + uploads it; subsequent examples for the same text get the cached `cl_mem`.
- Added `<map>` include.
- Refactored `run_hc8_gpu` (single-run path) into a thin wrapper around a new `run_hc8_gpu_inner(ocl, text_buf, …)` that does only per-pattern work (preprocessing, pattern/F/count/over uploads, kernel, readback).

Built + ran `--csv-dir results-csv\step2`. All `same_indices: true`. Per-call wall on warm-cache examples collapsed from ~120 ms to ~2–60 ms:

| run (warm) | wall step1 | wall step2 | speedup step2 |
|---|---:|---:|---:|
| english-extra-8 | 167 | 58.7 | 3.00× |
| english-extra-16 | 119 | 10.7 | 1.93× |
| english-extra-32 | 120 | 4.9 | 2.38× |
| english-extra-128 | 133 | 2.0 | 3.29× |

Cold runs (first-time-per-text) still ~120–175 ms — now dominated by file read + driver pinning + memcpy + DMA.

§4 stop condition for Step 2 ("all bases > 1× cold-start") not met. Continued to Step 3.

### 2.4 Step 3 — pinned-staging text upload (`main.cpp` only)

Replaced the `clCreateBuffer + clEnqueueWriteBuffer(text)` with a `CL_MEM_ALLOC_HOST_PTR` staging buffer, `clEnqueueMapBuffer` → `memcpy` → `clEnqueueUnmapMemObject`, then `clEnqueueCopyBuffer(staging → device)`. Helper: `upload_text_pinned(ocl, text)`.

Built + ran `--csv-dir results-csv\step3`. All `same_indices: true`. Key result: **H2D events became honest** — cold runs reported ~3.8–4.7 ms for the 100 MB DMA (consistent with PCIe 4.0 staging at ~22 GB/s). Cold-start wall barely moved (~110–175 ms) because:

```
Cold-protein wall = 175 ms
   ≈  ~50–80 ms file read (read_binary_file)
   +  ~20–30 ms clCreateBuffer pinned-register
   +  ~30–50 ms memcpy(host_ptr, vec.data(), 100MB)
   +    4.7 ms clEnqueueCopyBuffer (DMA, now visible)
   +    3.9 ms kernel
   +    <1 ms readback
```

The bottleneck moved out of the GPU pipeline entirely. Everything left to optimize on cold-start is host-side I/O.

### 2.5 Steps 4 + 5 — attempted, reverted

Step 4 (tiled upload + second `xfer_queue` + per-tile kernel launches via `clEnqueueNDRangeKernel` `global_work_offset`) and Step 5 (`ALPHA` 12 → 14, then 13 after the 64 KB LDS ceiling hit `CL_OUT_OF_RESOURCES`) were implemented and built.

Bugs hit:
1. **ALPHA=14**: `F_local = 16384 × 4 = 64 KB`, exactly the AMD per-workgroup LDS ceiling on RDNA 4. Tiled launch failed with `CL_OUT_OF_RESOURCES`. Dropped to `ALPHA=13` (`F_local = 32 KB`).
2. **Tiled correctness regression**: `global_size` rounded up to multiples of `local_size=256` made gids in tile 0 overflow into tile 1's chunks → matches double-recorded (DNA m=16 cold reported `[1000000, 59337108, 59337108, 69664436]` instead of 3 distinct). Fix attempted: add `tile_chunk_end` kernel arg replacing `num_chunks` in the bound check. Correctness was restored, but wall-clock numbers showed Step 4 saved at most ~2 ms vs Step 3 (the cold-start floor is host-side, immune to tiling).

User decision: revert both, keep Step 3 as the final state. Reverted:
- `kernel.cl`: `ALPHA` 13 → 12, `tile_chunk_end` arg name → `num_chunks` (revert).
- `main.cpp`: `ALPHA` 13 → 12, removed `xfer_queue`, removed tiled `upload_text_pinned`, removed `release_pinned_upload`, removed `NUM_TEXT_TILES`, removed `tile_xfer_events` parameter, restored single-launch kernel arg setup, restored single `upload_text_pinned` returning `{device_buf, h2d_ms}`.

`results-csv/step45/` left in place but not referenced by figures.

### 2.6 m-sweep extension (kept)

User asked for DNA and Protein at multiple `m` values (the original example list only had DNA m=16 and Protein m=32 — single-point coverage). Added 8 examples to `run_benchmark_examples`:

```
dna-extra-{8,32,64,128}
protein-extra-{8,16,64,128}
```

Reordered the list so the **first 3 entries** (one per base) pay the cold-upload, and the remaining 12 are warm-cache reuses of the already-uploaded text. Total: 15 examples.

Built + ran `--csv-dir results-csv\sweep`. All 15 `same_indices: true`.

### 2.7 Figures regenerated

`analise_benchmarks.py --input results-csv/sweep --out-dir figuras-benchmark` produces:
- `fig1_speedup.pdf`
- `fig2_tempos_absolutos.pdf`
- `fig3_memoria_gpu.pdf`
- `sweep_analysis_stdout.txt` (Markdown summary)

---

## 3. Final state of the codebase

| File | Change |
|---|---|
| `HashChainParallelized/main.cpp` | Steps 1–3 + m-sweep examples. Single `OclContext` hoisted across examples; text cache; `upload_text_pinned`; `chrono`-based `gpu_wall_ms`. `ALPHA = 12`. |
| `HashChainParallelized/kernel.cl` | Unchanged from Step 1 baseline. `ALPHA = 12`. |
| `HashChainParallelized/analise_benchmarks.py` | Already had the Step 1 update (reads `gpu_wall_ms`, falls back to `gpu_total_ms` for archive CSVs). Not modified this session. |
| `.venv/` (project root) | Created. Python 3.13.13 + pandas 3.0.2 + matplotlib 3.10.9 + seaborn 0.13.2 + numpy 2.4.4. |
| `HashChainParallelized/results-csv/step1/` | New. Step 1 verification CSVs + stdout. |
| `HashChainParallelized/results-csv/step2/` | New. Step 2 CSVs + stdout. |
| `HashChainParallelized/results-csv/step3/` | New. Step 3 CSVs + stdout. |
| `HashChainParallelized/results-csv/step45/` | New, **leftover from reverted attempts**. Safe to delete. |
| `HashChainParallelized/results-csv/sweep/` | New. Final m-sweep run, 15 examples × 2 (CPU + GPU) = 30 CSVs + stdout. **This is what the figures plot.** |
| `HashChainParallelized/results-csv/tcc1-archive/` | Untouched. Original TCC1 baseline. |
| `HashChainParallelized/figuras-benchmark/*.pdf` | Regenerated from `results-csv/sweep/`. |
| `HashChainParallelized/hc8_opencl_test.exe` | Final binary. Build: MSVC `cl /O2 /std:c++17` + `OCL_SDK_Light\lib\x86_64\OpenCL.lib`. |

---

## 4. Final results (m-sweep, 100 MB texts)

Wall-clock speedup = `cpu.search_ms / gpu_wall_ms`.

| Base | m | Cold/Warm | CPU search (ms) | GPU wall (ms) | Speedup |
|---|---:|---|---:|---:|---:|
| DNA | 16 | cold | 11.7 | 66.8 | 0.17× |
| DNA | 8 | warm | 94.4 | 26.1 | **3.61×** |
| DNA | 32 | warm | 4.6 | 2.5 | **1.84×** |
| DNA | 64 | warm | 3.8 | 1.4 | **2.66×** |
| DNA | 128 | warm | 2.8 | 0.9 | **3.11×** |
| English | 64 | cold | 4.2 | 59.7 | 0.07× |
| English | 8 | warm | 101.5 | 24.4 | **4.16×** |
| English | 16 | warm | 12.3 | 8.3 | **1.49×** |
| English | 32 | warm | 5.7 | 2.2 | **2.63×** |
| English | 128 | warm | 3.4 | 0.9 | **3.62×** |
| Proteins | 32 | cold | 5.2 | 95.5 | 0.05× |
| Proteins | 8 | warm | 102.8 | 25.1 | **4.10×** |
| Proteins | 16 | warm | 12.9 | 5.3 | **2.41×** |
| Proteins | 64 | warm | 4.1 | 2.0 | **2.05×** |
| Proteins | 128 | warm | 3.3 | 1.3 | **2.45×** |

GPU wins **all 12 warm-cache scenarios** across every base and every `m`. The 3 losses are exactly the 3 cold-first-time uploads (one per base). Setup tax (one-time, hoisted): 213 ms.

---

## 5. The thesis narrative this supports

1. **Correctness**: every run matches the sequential C reference on 100 MB corpora — `same_total`, `same_overflow`, `same_indices` all true on 15/15 scenarios.
2. **Honest measurement**: `gpu_wall_ms` (chrono around the GPU sequence) is the only valid speedup denominator on AMD; the OpenCL events (`h2d_ms` in particular) silently report 0 for non-blocking writes from pageable memory. The Step 1 + Step 3 chain made the events honest *and* added the wall-clock fallback.
3. **The realistic GPU use case wins**: when the same text is searched with multiple patterns (indexing, library-style usage), the GPU beats the sequential CPU on every base × `m` combination tested — 1.49–4.16×, with the largest wins on small `m` where CPU does the most work.
4. **The unrealistic GPU use case loses**: a one-shot search of a fresh 100 MB text on the GPU loses (~0.05–0.17×) because the disk read + driver-pinned-memory registration + host-to-pinned `memcpy` together cost ~120–175 ms — more than CPU takes to finish the entire search. This is a **fundamental property of GPU offload** for cheap-per-byte workloads, not a defect of the implementation. Steps 4 and 5 confirmed they cannot move this floor.
5. **TCC1's narrative was incomplete**: "GPU loses on small alphabets due to PCIe" was directionally right but missed the bigger fixed cost (per-call OpenCL context creation, ~110 ms, which Step 2 hoisted out). Once that's hoisted, PCIe DMA itself is only ~4 ms on this hardware — small enough that even Proteins cold-start is bounded by host I/O, not PCIe.

---

## 6. Open questions / what's left

- **Cold-start can't be flipped without changing the I/O path.** The only remaining lever is "read file directly into the pinned buffer" (skipping the intermediate `std::vector` and the host→pinned `memcpy`) or `mmap`-ing the file. Saves ~30–80 ms on cold. Even then, Proteins cold = ~95 ms vs CPU ~5 ms = 0.05× — still a loss. There is no parameter combination where one-shot small-`m` GPU search beats CPU on this corpus size.
- **Step 4 / Step 5 are not viable as written** for this hardware/scenario:
  - Step 5 with `ALPHA=14` requires 64 KB LDS = the AMD per-workgroup ceiling, triggers `CL_OUT_OF_RESOURCES` on tiled launches. `ALPHA=13` (32 KB) builds and runs but gives marginal (or noisy) kernel-time wins for the work we're doing.
  - Step 4 saves ≤ ~4 ms even when correct. The bug from the first attempt (gids past `tile_chunk_end` re-entering the search) is fixable with a single kernel arg, but the engineering cost outweighs the win.
- **`rodar_testes.bat` still hardcodes `lib\x64`.** Doesn't match `OCL_SDK_Light`'s actual `lib\x86_64`. Either fix the .bat or document the manual `cl` invocation.

---

## 7. Reproducing this from scratch

From `c:\Documentos\Faculdade\tcc_Arthur\HashChainParallelized\` in a Developer Command Prompt:

```
:: Build (assuming OCL_SDK_Light at C:\Program Files (x86)\OCL_SDK_Light)
cl /nologo /O2 /std:c++17 /EHsc main.cpp ^
   /I"C:\Program Files (x86)\OCL_SDK_Light\include" ^
   /link /LIBPATH:"C:\Program Files (x86)\OCL_SDK_Light\lib\x86_64" OpenCL.lib ^
   /OUT:hc8_opencl_test.exe

:: Run all 15 examples
hc8_opencl_test.exe --run-examples --csv-dir results-csv\sweep

:: Regenerate figures
..\.venv\Scripts\python.exe analise_benchmarks.py ^
   --input results-csv\sweep --out-dir figuras-benchmark
```
