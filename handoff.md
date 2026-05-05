# Handoff to a fresh Claude Code instance running on the Windows + AMD GPU machine

Read this whole file before doing anything. Context first, then the task.

---

## 1. Who you are and what's going on

You are a fresh Claude Code session opened by Arthur on his **Windows desktop with an AMD Radeon RX 9060 XT (RDNA 4)**. Another Claude session, running on his Linux laptop, already wrote code changes for **Step 1** of a multi-step optimization plan and committed them to disk. Arthur switched machines because the Linux laptop has no GPU.

The repo is the codebase for his undergraduate thesis (TCC) at PUC Minas — **"Paralelização do algoritmo Hash Chain para busca exata de strings"**. He is parallelizing the `hc8` variant of Hash Chain (Palmer/Faro/Scafiti, SEA 2024) on AMD GPU via OpenCL 1.2 and comparing against the original sequential C reference.

**Read these first**, in this order, to understand the project and constraints:

1. `CLAUDE.md` (project root) — full project context, layout, build/run/test, conventions, anti-patterns. **The anti-pattern section is binding** — especially: don't touch `bd/` (corpus, 100 MB files), don't touch `src/` (legacy reference C code), don't desync `ALPHA/Q` between `main.cpp` and `kernel.cl`, don't translate Portuguese log strings, don't rerun in parallel with other GPU workloads.
2. `HashChainParallelized\main.cpp` — host code (C++17, OpenCL 1.2). The Step 1 changes already live here.
3. `HashChainParallelized\kernel.cl` — device code. **You will not modify this in Step 1.**

The plan you are executing the verification half of:
`C:\Users\<arthur>\.claude\plans\i-wanted-to-make-logical-map.md` (or wherever Claude Code stores plans on Windows). If it's not there, the plan is summarized in section 4 below — that's enough.

---

## 2. The scientific question and why Step 1 exists

The thesis (TCC1) reports that the GPU implementation **wins ~5× on English text but loses (~0.9×) on DNA and Proteins**, blaming the H2D PCIe transfer of the 100 MB corpus.

But the existing CSVs in `HashChainParallelized\results-csv\` (now archived in `HashChainParallelized\results-csv\tcc1-archive\`) **contradict that narrative**:

| Base | m | CPU search (ms) | GPU kernel (ms) | h2d_ms (CSV) | gpu_total_ms | Implied speedup |
|---|---|---|---|---|---|---|
| DNA | 16 | 19.745 | 9.768 | **0.000** | 9.779 | 2.0× |
| Proteins | 32 | 7.932 | 4.174 | **0.000** | 4.185 | 1.9× |
| English | 64 | 6.825 | 2.400 | **0.000** | 2.413 | 2.8× |

`h2d_ms = 0.000` is impossible for a 100 MB Gen4 transfer (~5–6 ms minimum). The most likely cause: AMD's `clGetEventProfilingInfo` returns equal `START`/`END` timestamps for the events on non-blocking `clEnqueueWriteBuffer`, so the H2D silently zeroes out and the implicit `gpu_total = h2d + kernel + d2h` becomes "kernel only". The TCC1 figures were probably taken with wall-clock timing around the whole submit-and-wait sequence, which is why they tell a different (more negative) story.

**Step 1's purpose is to settle this** — measure honestly with `chrono` wall-clock around the GPU sequence, then look at the gap.

---

## 3. What's already been done (don't redo)

In `HashChainParallelized\main.cpp`:

- `RunResult` now has `double gpu_wall_ms`.
- A `std::chrono::high_resolution_clock` `wall_start` snapshot is taken right before `create_opencl(...)` inside `run_hc8_gpu`, and `wall_end` is taken right after the last `clWaitForEvents` for the results readback (before host-side `assign`/`sort` and before `clReleaseMemObject`). The duration is stored in `out.gpu_wall_ms`.
- `print_log` prints a new line `GPU_wall(ms): ...`.
- `compare_and_print` prints a second speedup line: `[COMPARE] speedup(search_cpu / gpu_wall): X.YYYx`.
- The GPU CSV writer adds a `gpu_wall_ms` column (between `gpu_total_ms` and `index_order`). Old CSVs in `tcc1-archive/` don't have this column — the Python analyzer was updated to handle both.

In `HashChainParallelized\analise_benchmarks.py`:

- Reads `gpu_wall_ms` if present, falls back to `gpu_total_ms` for archived CSVs.
- Speedup is now `cpu_total_ms / gpu_wall_ms` (was `/ gpu_total_ms`).
- Markdown output table has both `GPU parede` and `GPU eventos` columns.

In `HashChainParallelized\results-csv\tcc1-archive\`:

- 17 CSVs from the TCC1 baseline run, preserved before any new runs overwrite the working folder. **Do not delete or move these.** They back the figures already cited in TCC1.

**Do not edit any of the above.** Your job is to build the binary and capture verification data.

---

## 4. The full plan (just for orientation — you only execute Step 1 verification)

| Step | What it does | Files touched | Expected effect on DNA/Protein |
|---|---|---|---|
| **1 (DONE in code, you VERIFY)** | Wall-clock GPU timing | `main.cpp`, `analise_benchmarks.py` | Measurement honesty |
| 2 (next) | Hoist `OclContext` out of per-call path; cache text buffers by file path across the 7 benchmark examples | `main.cpp` | Eliminates re-upload of `english.100MB` for examples 2–5; eliminates redundant `clBuildProgram` |
| 3 | Replace text upload with `CL_MEM_ALLOC_HOST_PTR` + `clEnqueueMapBuffer` (pinned host memory) | `main.cpp` | ~1.5–2× faster H2D |
| 4 (only if needed) | Tiled double-buffered upload — overlap H2D with kernel on a second queue | `main.cpp`, small kernel arg added | Hides H2D entirely behind kernel work |
| 5 (last resort) | Bump `ALPHA` from 12 to 14/16 to drop DNA false-positive rate | both `main.cpp` AND `kernel.cl` (must stay in sync) | Reduces kernel work for small alphabets |

User's success criterion: **just a positive speedup (>1×) consistently** for DNA and Proteins. Anything beyond that is bonus.

Stop conditions:
- After **Step 1**: if `gpu_wall_ms ≈ gpu_total_ms` (gap < 1 ms) AND DNA/Proteins already > 1.0× → only do Step 2 for the realistic-usage win, then stop entirely.
- After **Step 2**: if all bases > 1.0× cold-start → stop.
- After **Step 3**: if Proteins cold-start ≥ 1.1× → stop, skip Steps 4 and 5.

---

## 5. Your task: build and verify Step 1

### 5.1 Build via Visual Studio

The user is opening this in **Visual Studio**. The repo doesn't currently ship a `.sln` or `.vcxproj` — the existing build path is `rodar_testes.bat` (MSVC `cl.exe` direct invocation, with g++ fallback).

Two options. Pick whichever matches what the user already has in front of them.

#### Option A — They opened the folder via `File → Open → Folder`

VS treats it as a "Folder View" / Open Folder workspace. There is no project to set properties on. Two sub-options:

- **A.1 — Use the existing batch file (simplest):** open the **Developer Command Prompt for VS 2022** (Start menu) inside the repo folder and run:
  ```
  cd /d "C:\path\to\TCC\HashChainParallelized"
  rodar_testes.bat
  ```
  The batch file calls `cl /O2 /std:c++17 /EHsc main.cpp ... OpenCL.lib /OUT:hc8_opencl_test.exe` and runs the examples. If `cl` errors with "Cannot open include file 'CL/cl.h'", set `OPENCL_SDK` first:
  ```
  set OPENCL_SDK=C:\Program Files (x86)\OCL_SDK_Light
  rodar_testes.bat
  ```
  (OCL_SDK_Light: download from https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases — the user may already have it; the AMD Adrenalin driver alone does **not** include the OpenCL headers.)

  This already runs `--run-examples` — go straight to **section 5.3 Capture**.

- **A.2 — Create a `CMakeLists.txt` so VS Folder View can drive the build.** Only do this if the user explicitly asks. The simpler path is A.1.

#### Option B — They want a real `.sln`

Create one yourself only if the user asks for it explicitly. The `.sln`/`.vcxproj` files are **not** in `.gitignore` and would be committed by accident; ask before generating. If the user insists:

1. `File → New → Project → Empty Project (C++)`. Name it `hc8_opencl_test`. Save the `.sln` **inside** `HashChainParallelized\` so relative paths to `bd/` and `kernel.cl` work at runtime.
2. Right-click the project → `Add → Existing Item` → select `main.cpp`.
3. Project Properties (Configuration: **Release**, Platform: **x64**):
   - **C/C++ → General → Additional Include Directories** → `$(OPENCL_SDK)\include` (or the literal `C:\Program Files (x86)\OCL_SDK_Light\include`)
   - **C/C++ → Language → C++ Language Standard** → `/std:c++17`
   - **C/C++ → Optimization** → `/O2`
   - **Linker → General → Additional Library Directories** → `$(OPENCL_SDK)\lib\x64`
   - **Linker → Input → Additional Dependencies** → prepend `OpenCL.lib;`
   - **Debugging → Working Directory** → `$(ProjectDir)` (so `bd/`, `kernel.cl`, and `results-csv/` resolve)
   - **Debugging → Command Arguments** → `--run-examples --csv-dir results-csv\step1`
4. **Build → Build Solution** (Ctrl+Shift+B). Output: `hc8_opencl_test.exe` somewhere under `x64\Release\`.
5. Add `*.sln`, `*.vcxproj`, `*.vcxproj.*`, `x64/`, `Debug/`, `Release/` to `HashChainParallelized\.gitignore` before they get accidentally committed.

### 5.2 Run

Whichever build path you took, the run is:
```
hc8_opencl_test.exe --run-examples --csv-dir results-csv\step1
```
from `HashChainParallelized\` as CWD.

This invokes `run_benchmark_examples`, which loops over 7 scenarios:

1. dna m=16 (`bd/dna.100MB`)
2. english m=64 (`bd/english.100MB`)
3. protein m=32 (`bd/proteins.100MB`)
4. english m=8
5. english m=16
6. english m=32
7. english m=128

Each prints a `[COMPARE]` block. Crucially, the new line is:
```
[COMPARE] speedup(search_cpu / gpu_wall): X.YYYx
```
plus a `GPU_wall(ms): ...` printed in the per-run block above it.

**Do not run any other GPU workload during the benchmark** (anti-pattern #14). Close any game/browser GPU compositor work if you can.

### 5.3 Capture

Two outputs to gather:

1. **Stdout** — the user can pipe it to a file, or you can redirect when launching:
   ```
   hc8_opencl_test.exe --run-examples --csv-dir results-csv\step1 > results-csv\step1\stdout.txt 2>&1
   ```
2. **CSVs** — `results-csv\step1\` will contain ~14 files (one CPU + one GPU per example).

### 5.4 Sanity check

In the stdout, find each `[COMPARE]` block. For **every** example, all three of these must be `true`:
```
[COMPARE] same_total: true
[COMPARE] same_overflow: true        (or both false — they just have to match)
[COMPARE] same_indices: true
```

If **any** example shows `same_indices: false`, **stop immediately**. The Step 1 timing changes shouldn't have affected correctness (they only added measurements), so a `false` would mean something else is broken — possibly an unrelated kernel/driver issue, or that `kernel.cl` got modified accidentally. In that case:
- Check `git status` inside `HashChainParallelized\`. If `kernel.cl` is dirty, revert it: `git checkout -- kernel.cl`.
- Re-run.
- If `same_indices: false` persists, do not write a follow-up edit; report this back to the user verbatim with the failing run tag and the first 5 indices from CPU vs GPU.

### 5.5 Report back

Build a Markdown table from the stdout. For each of the 7 runs:

| run_tag | CPU search (ms) | GPU_wall (ms) | GPU_total (ms) | gap (wall − total) | speedup vs gpu_wall |
|---|---|---|---|---|---|
| benchmark-example:dna | ... | ... | ... | ... | ... |
| benchmark-example:english | ... | ... | ... | ... | ... |
| benchmark-example:protein | ... | ... | ... | ... | ... |
| benchmark-example:english-extra-8 | ... | ... | ... | ... | ... |
| benchmark-example:english-extra-16 | ... | ... | ... | ... | ... |
| benchmark-example:english-extra-32 | ... | ... | ... | ... | ... |
| benchmark-example:english-extra-128 | ... | ... | ... | ... | ... |

Plus a one-line verdict:
- If **all gaps < 1 ms** AND DNA m=16 and Proteins m=32 both have `speedup_vs_gpu_wall > 1` → "Step 1 settled it: events were honest, kernel really is the bottleneck. Recommend: do Step 2 (context hoist) for amortized wins on `--run-examples`, then stop."
- If **at least one gap ≥ 4 ms** AND/OR DNA/Protein speedup_vs_gpu_wall < 1 → "Step 1 confirms TCC1: H2D is hidden in the events and dominates for small alphabets. Recommend: continue to Steps 2 and 3 (context hoist + pinned host memory)."
- Anything else → describe what you see; don't pick a recommendation.

### 5.6 What NOT to do

- **Do not start Step 2.** That's for the next session, after the user reads your report and confirms the direction.
- **Do not edit `main.cpp`, `kernel.cl`, or `analise_benchmarks.py`.** Step 1's code is final unless verification reveals a bug.
- **Do not delete or overwrite anything in `results-csv\tcc1-archive\`.**
- **Do not commit anything.** The user commits manually. Do not run `git add` / `git commit` even if asked indirectly.
- **Do not enumerate or read `bd/`.** It's the 100 MB corpus; it's been pre-validated; the program reads it at runtime.
- **Do not "fix" things you weren't asked to fix** — e.g., don't rename Portuguese log strings, don't reorganize the CSV schema beyond the Step 1 column, don't add unit tests.
- **Do not build in Debug.** Release / O2 is required for the timings to be representative.
- **Do not run with other GPU workloads active.** Compositor effects, browsers with hardware accel, games — close them. The kernel times are sub-10 ms and very sensitive to contention.

---

## 6. If something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `cl` not found | Not in Developer Command Prompt | Open "Developer Command Prompt for VS 2022", `cd` into `HashChainParallelized\`, retry. |
| `Cannot open include file: 'CL/cl.h'` | OpenCL SDK not on include path | `set OPENCL_SDK=C:\Program Files (x86)\OCL_SDK_Light` (install OCL_SDK_Light first if missing), retry `rodar_testes.bat`. |
| `unresolved external symbol clGetPlatformIDs` | OpenCL.lib not linked | Same — `OPENCL_SDK` must point to a folder containing `lib\x64\OpenCL.lib`. |
| `Falha ao abrir arquivo: bd/dna.100MB` | Wrong CWD | Ensure `bd\dna.100MB` etc. exist under `HashChainParallelized\`. Run from that folder. |
| `Falha ao abrir kernel: kernel.cl` | Same | Same. |
| `Falha no build do kernel:` followed by OpenCL log | `kernel.cl` was modified or driver mismatch | `git status`, revert `kernel.cl`. If clean, paste the log into the report. |
| `Nenhuma plataforma OpenCL encontrada` | AMD driver not installed or ICD not registered | Install AMD Adrenalin Software. Reboot. |
| `same_indices: false` | Real regression | See section 5.4. |
| Numbers vary wildly between runs | Other GPU work / power state / first-run JIT | Run 3× in a row; report the median, note the variance. |

---

## 7. The exact command to start

Once VS or the dev prompt is open in `C:\path\to\TCC\HashChainParallelized\`:

```
rodar_testes.bat
```

…will compile and run. If it succeeds, the new CSVs are in `results-csv\windows\` (that's what the .bat hardcodes — note: **not** `results-csv\step1\`). To match the plan, after the .bat finishes, additionally run:

```
hc8_opencl_test.exe --run-examples --csv-dir results-csv\step1 > results-csv\step1\stdout.txt 2>&1
```

…to get a clean labeled set under `step1\`.

That's the whole job for this session. Report back with the table from section 5.5 and the verdict. Don't proceed to Step 2.