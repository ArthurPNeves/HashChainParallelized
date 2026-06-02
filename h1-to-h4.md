# RESOLVER H1–H4 — Lacunas metodológicas (foco no H2)

> **Para o Claude Code que abrir na máquina do amigo (GPU AMD / Windows):**
> Este documento te orienta a fechar quatro lacunas metodológicas da revisão do TCC — **H1 a H4, com
> prioridade no H2**. Diferente do `INSTRUCOES_TESTES_AMIGO.md` (que só *coleta* os benchmarks), aqui você
> **pode e deve editar** `main.cpp` e `analise_benchmarks.py`, porque é nesta máquina que dá para **compilar
> e testar com a GPU real**. Leia antes o `CLAUDE.md` (pasta acima) e o `INSTRUCOES_TESTES_AMIGO.md`.
>
> **Regras inegociáveis:** mantenha strings em PT-BR; não toque em `bd/` nem `src/`; não dessincronize
> `ALPHA/Q` entre `main.cpp` e `kernel.cl`; ao final, rode a regressão (`[REGRESSION-FAIL]`) e o
> `analise_benchmarks.py --strict`. As mudanças aqui **estendem** o que já existe (não refaça o que já está pronto).

---

## 0. Visão geral — o que cada H pede e o status atual

| H | Problema (revisão) | Status | O que fazer aqui |
|---|--------------------|--------|------------------|
| **H1** | §4.1 não descreve CPU, RAM, disco, driver, compilador, flags | **Falta coletar** | Imprimir info do *device* (OpenCL) + capturar ambiente do host; preencher a tabela de §4.1 |
| **H2** ⭐ | §4.3 promete 4 métricas; só *speedup* é reportado (faltam **vazão**, **taxa de colisões**, **energia**) | **Falta instrumentar** | Implementar **vazão (GB/s)** e **taxa de colisões/falsos positivos**; energia = opcional/futuro |
| **H3** | *cold/warm* confundido (amostra única, *m* diferente por base) | **Já resolvido pelo código preparado** | Apenas ajustar o **texto** de §5 (instruções na Seção 4) |
| **H4** | decomposição *cold* apresentada como medida sem instrumentação | **Já resolvido pelo código preparado** | Apenas ajustar o **texto** de §5 (instruções na Seção 5) |

> **Por que H3 e H4 já estão resolvidos:** o `main.cpp` preparado agora mede `file_read_ms`, `gpu_upload_*`,
> `kernel_*`, `d2h` **separadamente** (H4 deixa de ser estimativa) e grava a coluna `regime` com **31 repetições**,
> medindo *cold* **e** *warm* no mesmo `(base, m)` (H3 deixa de ser confundido). Falta só refletir isso no texto.

---

## 1. ⭐ H2 — Reportar as métricas prometidas

A §4.3 do `tcc.tex` promete **Speedup, Taxa de Colisões, Vazão (Throughput) e Consumo Energético/Ocupação**.
Hoje só o *speedup* aparece. Vamos fechar **vazão** e **taxa de colisões** (baratas e diretas) e tratar **energia**
como opcional/futuro.

### 1.1 Vazão (Throughput, GB/s)

É trivial: `vazão = bytes_do_texto / tempo`. Só falta **gravar `text_bytes`** no CSV e **computar** na análise.

**(a) `main.cpp` — gravar o tamanho do texto.**
- No `struct TimingsRow`, adicione: `uint64_t text_bytes = 0;`
- No laço de `run_benchmark_examples`, logo após obter `text_ref`, capture `text_ref.size()` e atribua a
  `warm.text_bytes` (e a `cold.text_bytes`, quando `first_load`).
- No `write_timings_csv`, acrescente `text_bytes` ao cabeçalho e ao corpo (pode ser a primeira coluna nova ao final).

**(b) `analise_benchmarks.py` — computar e reportar.**
- Vazão GPU (warm): `vazao_gpu = text_bytes / (gpu_query_wall_ms_median/1000) / 1e9` (GB/s).
- Vazão CPU (warm): `vazao_cpu = text_bytes / (cpu_search_ms_median/1000) / 1e9`.
- Acrescente uma coluna **"Vazão GPU (GB/s)"** (e opcionalmente CPU) na *Tabela — regime WARM* impressa por
  `imprimir_tabela`. Ordem de grandeza esperada: ~4 GB/s na GPU para m=8; CPU ~1 GB/s.

### 1.2 Taxa de Colisões (falsos positivos do filtro)

A §4.3 define: *"frequência de falsos positivos na etapa de filtragem… verificações desnecessárias"*. O filtro $F$
(estilo Bloom) gera candidatos que **não** são casamentos reais; essa é a taxa de colisão. É uma propriedade
**algorítmica** (igual na CPU e na GPU), então basta **contar na CPU** — fora do laço cronometrado, para não
perturbar o tempo.

**(a) `main.cpp` — função de contagem (cópia fiel do laço de `run_hc8_cpu`, sem timing e sem gravar índices).**
Coloque logo **depois** de `run_hc8_cpu` (as constantes `ASIZE/TABLE_MASK/Q/Q2/END_FIRST_QGRAM`, `preprocessing_hc8`,
`chain_hash8`, `link_hash` já estão no escopo):

```cpp
struct CollisionStats {
    uint64_t probes = 0;               // posições onde o filtro foi consultado
    uint64_t filter_candidates = 0;    // F[H&MASK] != 0 (passou o filtro barato)
    uint64_t chain_verifications = 0;  // cadeia completa validada (chegou ao memcmp)
    uint64_t memcmp_calls = 0;         // H == Hm -> memcmp de fato executado
    uint64_t matches = 0;              // casamentos confirmados
};

CollisionStats count_filter_collisions(const std::vector<unsigned char>& text,
                                       const std::vector<unsigned char>& pattern) {
    CollisionStats cs;
    const int n = static_cast<int>(text.size());
    const int m = static_cast<int>(pattern.size());
    const int MQ1 = m - Q + 1;
    std::vector<uint32_t> F(ASIZE, 0u);
    const uint32_t Hm = preprocessing_hc8(pattern, F);

    int pos = m - 1;
    while (pos < n) {
        ++cs.probes;
        uint32_t H = chain_hash8(text, pos);
        uint32_t V = F[H & TABLE_MASK];
        if (V) {
            ++cs.filter_candidates;
            const int end_second_qgram_pos = pos - m + Q2;
            while (pos >= end_second_qgram_pos) {
                pos -= Q;
                H = chain_hash8(text, pos);
                if (!(V & link_hash(H))) goto shift_cc;
                V = F[H & TABLE_MASK];
            }
            pos = end_second_qgram_pos - Q;
            const int match_start = pos - END_FIRST_QGRAM;
            ++cs.chain_verifications;
            if (H == Hm) {
                ++cs.memcmp_calls;
                if (std::memcmp(text.data() + match_start, pattern.data(), static_cast<size_t>(m)) == 0) {
                    ++cs.matches;
                }
            }
        }
        shift_cc:
        pos += MQ1;
    }
    return cs;
}
```

> ⚠️ Mantenha a estrutura **idêntica** à de `run_hc8_cpu` (o `goto shift_cc` salta para *fora* dos escopos, o que é
> válido — o original faz o mesmo). Se você alterar `run_hc8_cpu` algum dia, espelhe aqui.

**(b) `main.cpp` — chamar 1× por cenário (fora do laço de repetições).** No `run_benchmark_examples`, depois de
`const auto pattern = sample_pattern_from_text(...)` e **antes** do laço `for (int it = 0; ...)`:

```cpp
const CollisionStats cs = count_filter_collisions(text_ref, pattern);
```

Adicione ao `TimingsRow` os campos `uint64_t probes, filter_candidates, chain_verifications, memcmp_calls;` e
preencha em `warm` (e `cold`, se `first_load`) a partir de `cs`. Inclua-os no `write_timings_csv` (cabeçalho + corpo).

**(c) `analise_benchmarks.py` — reportar.** Calcule e imprima numa pequena tabela (uma linha por `(base, m)`, regime warm):
- **FP do filtro** = `(filter_candidates - matches) / max(filter_candidates, 1)`
- **FP do memcmp** = `(memcmp_calls - matches) / max(memcmp_calls, 1)`
- **candidatos por KiB** = `filter_candidates / (text_bytes/1024)` (densidade de candidatos)

Esses são exatamente os "verificações desnecessárias" citados na §4.3, e explicam a divergência de *wavefront*
mencionada na §3.3.

### 1.3 Consumo Energético / Ocupação (opcional — senão, futuro declarado)

Esta é a única métrica difícil de automatizar no Windows. Duas saídas:

- **Se houver ferramenta AMD:** use o **AMD uProf** (`AMDuProfCLI.exe timechart --event power ...`) ou o
  **Radeon Developer Tool Suite** para amostrar a **potência média da GPU (W)** enquanto roda o *sweep* warm.
  Energia ≈ potência média × tempo total; reporte **energia por consulta** (J/consulta) e a **ocupação** (se o
  profiler expuser). Rode em paralelo ao `rodar_tcc_final.bat`.
- **Se não houver:** **mantenha como trabalho futuro** (a Conclusão do `tcc.tex` já reconhece isso) e
  declare explicitamente na §4.3/§5 que energia/ocupação ficaram fora do escopo desta coleta. **Não invente números.**

### 1.4 Texto para o `tcc.tex` (H2)

**§4.3 — substituir as definições por métricas que realmente serão reportadas:**

```latex
\item \textbf{Vazão (Throughput):} dados varridos por unidade de tempo,
      $\mathrm{vazão} = N / T$, reportada em GB/s para CPU e GPU no regime \textit{warm}.
\item \textbf{Taxa de colisões do filtro:} fração de candidatos do filtro $F$ que não
      correspondem a ocorrências reais, $\mathrm{FP} = (\text{candidatos} - \text{ocorrências})/\text{candidatos}$,
      medida na varredura sequencial (propriedade algorítmica, idêntica em CPU e GPU).
```

E, se energia ficar fora, ajuste a Conclusão para deixar claro que ocupação/energia permanecem como trabalho futuro
(já está lá — só reforçar). **§5 — adicionar** um parágrafo e, se quiser, uma tabela com Vazão (GB/s) e FP% por cenário,
a partir das tabelas impressas por `analise_benchmarks.py`.

---

## 2. H1 — Descrever o ambiente de teste (§4.1)

Faltam: **modelo da CPU** (o *baseline*!), RAM, **tipo de disco** (a §5.4 atribui ~50–80 ms à leitura), versão do
Windows, **versão do driver AMD**, compilador e **flags**, e os parâmetros do *device* OpenCL. Vamos **coletar
automaticamente** e preencher a tabela.

### 2.1 `main.cpp` — imprimir info do *device* OpenCL

Adicione um helper (logo após `create_opencl`) e chame-o no início de `run_benchmark_examples` (após o `[SETUP]`):

```cpp
void print_device_info(const OclContext& ocl) {
    auto str_info = [&](cl_device_info p) {
        size_t sz = 0; clGetDeviceInfo(ocl.device, p, 0, nullptr, &sz);
        std::string s(sz, '\0'); clGetDeviceInfo(ocl.device, p, sz, s.data(), nullptr);
        if (!s.empty() && s.back() == '\0') s.pop_back();
        return s;
    };
    cl_uint cu = 0;    clGetDeviceInfo(ocl.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    cl_ulong gmem = 0; clGetDeviceInfo(ocl.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
    size_t wg = 0;     clGetDeviceInfo(ocl.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
    std::cout << "[DEVICE] name=" << str_info(CL_DEVICE_NAME)
              << " | vendor=" << str_info(CL_DEVICE_VENDOR)
              << " | driver=" << str_info(CL_DRIVER_VERSION)
              << " | cl_version=" << str_info(CL_DEVICE_VERSION)
              << " | compute_units=" << cu
              << " | global_mem_MiB=" << (gmem / (1024 * 1024))
              << " | max_workgroup=" << wg << "\n";
}
```

Isso joga a linha `[DEVICE] ...` no `results-csv\final\stdout.txt`.

### 2.2 `rodar_tcc_final.bat` — capturar o ambiente do host

Acrescente, **antes** do passo de build, a captura para um arquivo `ambiente.txt`:

```bat
echo [INFO] Capturando ambiente para %OUTDIR%\ambiente.txt
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
( wmic cpu get name,maxclockspeed,numberofcores,numberoflogicalprocessors /format:list
  wmic memorychip get capacity,speed /format:list
  wmic diskdrive get model,mediatype,size /format:list
  ver
  cl 2>&1 | findstr /i version
  g++ --version 2>nul ) > "%OUTDIR%\ambiente.txt" 2>&1
```

> Em Windows recentes, `wmic` pode estar ausente; nesse caso use PowerShell:
> `powershell -Command "Get-CimInstance Win32_Processor | Select Name,MaxClockSpeed,NumberOfCores"` etc.

### 2.3 Texto para o `tcc.tex` (H1) — esqueleto de tabela em §4.1

Preencha com o que saiu em `ambiente.txt` e na linha `[DEVICE]`:

```latex
\begin{table}[htbp]\centering
\caption{Ambiente de teste.}\label{tab:ambiente}
\begin{tabular}{|l|l|}\hline
\textbf{Componente} & \textbf{Especificação} \\ \hline
GPU            & AMD Radeon RX 9060 XT (RDNA 4), \_\_ CUs, \_\_ GiB VRAM \\ \hline
Driver / OpenCL & \_\_\_ / OpenCL \_\_\_ (CL1.2) \\ \hline
CPU            & \_\_\_ (\_\_ núcleos / \_\_ threads, \_\_ GHz) \\ \hline
RAM            & \_\_ GB \\ \hline
Armazenamento  & \_\_\_ (NVMe/SATA SSD/HDD) \\ \hline
SO             & Windows \_\_\_ \\ \hline
Compilador (host) & MSVC \_\_\_ / \texttt{cl /O2 /std:c++17} (ou g++ \_\_\_ \texttt{-O2}) \\ \hline
Compilador (device) & OpenCL C 1.2 (\texttt{-cl-std=CL1.2}) \\ \hline
PCIe           & Gen\_\_ x\_\_ \\ \hline
\end{tabular}
\end{table}
```

> **Importante (validade do *baseline*):** declare que o *baseline* sequencial foi compilado com **`-O2`** (não `-O0`).
> Confirme também a **geração do PCIe** (afeta a aritmética de ~4–5 ms da §5.1): no Windows, veja em
> GPU-Z ou `Get-PnpDeviceProperty`.

---

## 3. H3 — *cold/warm* (apenas texto; código já resolve)

O `main.cpp` preparado já: (i) grava a coluna **`regime`**; (ii) mede **31 repetições** com mediana+CV%;
(iii) mede **cada base em *cold* E *warm* no mesmo `(base, m)`**. Isso elimina o confundimento e a amostra única.
**Não precisa mexer no código.** Ajuste o texto:

- **§5.2:** explique que o regime é determinado pelo *cache* (1ª carga de cada texto = *cold*; consultas seguintes
  sobre o buffer em VRAM = *warm*), que há **3 cold + 15 warm** e que cada cold tem um *warm* correspondente no mesmo
  `(base, m)`, permitindo comparação direta.
- **Limitação residual a declarar (honestidade):** o *cold* depende de **uma única leitura de disco** por texto
  (o SO mantém o arquivo em cache depois) — logo o componente "leitura" do *cold* é uma amostra única; a dispersão
  reportada (CV%) cobre o **upload** e a **busca**, não a leitura de disco.

---

## 4. H4 — decomposição do *cold* (apenas texto; código já resolve)

O `main.cpp` preparado mede e grava **separadamente**: `file_read_ms`, `gpu_upload_ms_median`, `kernel_ms_median`,
`d2h_ms`. A **Figura 3** já empilha esses componentes no painel *cold* (host-side visível). A decomposição deixou de
ser estimativa. **Não precisa mexer no código.** Ajuste o texto:

- **§5.4 ponto 4:** troque a frase "decompõe-se em ~50–80 ms…/~20–30 ms…" (estimativa) por valores **medidos**
  (leitura de disco e upload vêm do `timings_*.csv`; *kernel*/D2H idem) e cite a Figura 3 atualizada.
- **Limitação residual a declarar:** o `gpu_upload` é medido como **um número agregado** (pin + `memcpy` para a
  região fixada + DMA). A separação fina entre *pinning* e `memcpy` **não** foi instrumentada — declare isso em vez de
  estimar as sub-parcelas.

---

## 5. Passo a passo nesta máquina

1. Leia `CLAUDE.md` e `INSTRUCOES_TESTES_AMIGO.md`.
2. Implemente **H2** (Seção 1: vazão + colisões) e **H1** (Seção 2: device info + captura de ambiente) no
   `main.cpp`, `analise_benchmarks.py` e `rodar_tcc_final.bat`.
3. Compile e rode `rodar_tcc_final.bat`. Exija:
   - `[OK] Regressao` (nenhum `[REGRESSION-FAIL]`);
   - `analise_benchmarks.py --strict` saindo com sucesso (`kernel+h2d+d2h ≤ wall`);
   - no `timings_*.csv`, as **novas colunas** preenchidas (`text_bytes`, `filter_candidates`, `chain_verifications`,
     `memcmp_calls`, `matches`/`found_total`);
   - a análise imprimindo **Vazão (GB/s)** e **FP%**;
   - a linha `[DEVICE] ...` e o `ambiente.txt` gerados.
4. (Opcional) Energia com AMD uProf, conforme 1.3 — ou deixe como futuro.
5. Atualize o texto do `tcc.tex` (§4.1, §4.3, §5.2, §5.4) com os snippets das Seções 1–4.

### Checklist de verificação
- [ ] `text_bytes` ≈ 100 MB em todas as linhas.
- [ ] `filter_candidates ≥ matches` (sempre) e `chain_verifications ≥ matches`.
- [ ] FP% plausível (filtros de Bloom: pode variar de ~0% a dezenas de % conforme `m` e alfabeto).
- [ ] Vazão GPU > Vazão CPU no warm (coerente com os *speedups*).
- [ ] `[DEVICE]` mostra a RX 9060 XT e o driver.

---

## 6. O que trazer de volta

- `results-csv\final\timings_*.csv` (com as novas colunas), `stdout.txt` e `ambiente.txt`.
- As tabelas impressas pela análise (WARM com Vazão, e a nova tabela de **colisões/FP%**).
- Os 3 PDFs de `figuras-benchmark\`.
- (Se mediu energia) o relatório do AMD uProf.
- Os **diffs** de `main.cpp` / `analise_benchmarks.py` / `rodar_tcc_final.bat` que você fez aqui, para revisarmos em casa.

Com isso, H1 e H2 ficam **reportados de fato** e H3/H4 ficam **refletidos no texto** — fechando as quatro lacunas.