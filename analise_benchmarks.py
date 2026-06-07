"""Processa o CSV de *timings* do Hash Chain (CPU vs GPU) e gera as figuras do TCC.

Diferente da versão anterior, este script NÃO faz rglob recursivo nem média entre
lotes (causa da contaminação que reportava tempos pela metade). Ele lê apenas o
arquivo ``timings_*.csv`` mais recente do diretório informado em ``--input`` — esse
arquivo já traz mediana e CV% calculados no host (uma linha por cenário/regime).

Uso:
    python analise_benchmarks.py --input results-csv/final --out-dir figuras-benchmark [--strict]
"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EPS = 0.05  # tolerância (ms) para o assert kernel+h2d+d2h <= wall
CV_WARN = 15.0  # avisa se algum coeficiente de variação ultrapassar este valor (%)
ORDEM_BASES = ["DNA", "English", "Proteins", "Outros"]


def vazao_gbs(text_bytes: float, ms: float) -> float:
    """Vazão efetiva em GB/s (GB = 1e9 bytes): N/T. bytes / ms / 1e6 == bytes/(s)/1e9."""
    if text_bytes and text_bytes > 0 and ms and ms > 0:
        return float(text_bytes) / float(ms) / 1e6
    return float("nan")


def achar_timings_csv(input_path: Path) -> Path:
    """Retorna o timings_*.csv mais recente (ou o próprio arquivo, se for um)."""
    if input_path.is_file():
        return input_path
    candidatos = sorted(input_path.glob("timings_*.csv"))
    if not candidatos:
        raise RuntimeError(
            f"Nenhum timings_*.csv encontrado em {input_path}. "
            "Rode o benchmark com --csv-dir apontando para um diretório limpo."
        )
    # O timestamp no nome (%Y%m%d_%H%M%S) ordena lexicograficamente = cronologicamente.
    return candidatos[-1]


def carregar(input_path: Path) -> pd.DataFrame:
    arq = achar_timings_csv(input_path)
    print(f"[INFO] Lendo: {arq}")
    df = pd.read_csv(arq)
    df.columns = [c.strip().lower() for c in df.columns]
    obrigatorias = {"base", "pattern_len", "regime", "speedup"}
    faltando = obrigatorias - set(df.columns)
    if faltando:
        raise RuntimeError(f"CSV de timings sem colunas obrigatórias: {sorted(faltando)}")
    df["pattern_len"] = pd.to_numeric(df["pattern_len"], errors="coerce").astype(int)
    return df


def verificar(df: pd.DataFrame, strict: bool) -> None:
    """Sanidade: kernel+h2d+d2h <= wall, correção preservada, CV% e cobertura de cenários."""
    problemas: List[str] = []
    avisos: List[str] = []

    for _, r in df.iterrows():
        soma = float(r["kernel_ms_median"]) + float(r["h2d_ms"]) + float(r["d2h_ms"])
        wall = float(r["gpu_query_wall_ms_median"])
        if soma > wall + EPS:
            problemas.append(
                f"{r['base']} m={r['pattern_len']} ({r['regime']}): "
                f"kernel+h2d+d2h={soma:.3f} > wall={wall:.3f} (impossível)"
            )
        if int(r.get("same_indices", 1)) != 1:
            problemas.append(f"{r['base']} m={r['pattern_len']} ({r['regime']}): same_indices=0 (regressão)")
        if int(r.get("overflow", 0)) != 0:
            avisos.append(f"{r['base']} m={r['pattern_len']} ({r['regime']}): overflow=1 (índices inválidos)")
        for col in ["cpu_search_ms_cv", "kernel_ms_cv", "gpu_query_wall_ms_cv", "gpu_upload_ms_cv"]:
            if col in r and pd.notna(r[col]) and float(r[col]) > CV_WARN:
                avisos.append(f"{r['base']} m={r['pattern_len']} ({r['regime']}): {col}={float(r[col]):.1f}% (>{CV_WARN}%)")

    n_warm = int((df["regime"] == "warm").sum())
    n_cold = int((df["regime"] == "cold").sum())
    if n_warm != 15:
        avisos.append(f"Esperava 15 linhas warm (3 bases x 5 m), encontrei {n_warm}.")
    if n_cold < 1:
        avisos.append(f"Esperava ao menos 1 linha cold, encontrei {n_cold}.")

    print(f"\n[VERIFICAÇÃO] {n_warm} warm + {n_cold} cold.")
    for a in avisos:
        print(f"  [AVISO] {a}")
    for p in problemas:
        print(f"  [ERRO]  {p}")

    if problemas:
        print(f"\n[VERIFICAÇÃO] FALHOU: {len(problemas)} erro(s).")
        if strict:
            sys.exit(1)
    else:
        print("[VERIFICAÇÃO] OK: nenhuma inconsistência física e correção preservada.")


def _bases_presentes(df: pd.DataFrame) -> List[str]:
    return [b for b in ORDEM_BASES if b in set(df["base"])]


def _speedup_err(sub: pd.DataFrame) -> np.ndarray:
    """Erro absoluto do speedup propagando CV% de CPU e GPU (raiz da soma dos quadrados)."""
    cv_cpu = pd.to_numeric(sub.get("cpu_search_ms_cv"), errors="coerce").fillna(0.0) / 100.0
    cv_gpu = pd.to_numeric(sub.get("gpu_query_wall_ms_cv"), errors="coerce").fillna(0.0) / 100.0
    return (sub["speedup"].to_numpy() * np.sqrt(cv_cpu.to_numpy() ** 2 + cv_gpu.to_numpy() ** 2))


def grafico_speedup(df: pd.DataFrame, out_dir: Path) -> None:
    """fig1: speedup por (base, m). Painéis SEPARADOS para warm e cold (M1)."""
    fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
    bases = _bases_presentes(df)
    cores = dict(zip(ORDEM_BASES, sns.color_palette("Set2", len(ORDEM_BASES))))

    # ---- Painel WARM: agrupado por m, barras por base ----
    warm = df[df["regime"] == "warm"].copy()
    ms = sorted(warm["pattern_len"].unique())
    x = np.arange(len(ms))
    largura = 0.8 / max(len(bases), 1)
    for i, base in enumerate(bases):
        sub = warm[warm["base"] == base].set_index("pattern_len").reindex(ms).reset_index()
        ax_w.bar(x + i * largura, sub["speedup"], largura, label=base, color=cores[base],
                 yerr=_speedup_err(sub), capsize=3, edgecolor="black", linewidth=0.4)
    ax_w.axhline(1.0, color="red", linestyle="--", linewidth=1.2)
    ax_w.set_xticks(x + largura * (len(bases) - 1) / 2)
    ax_w.set_xticklabels([str(m) for m in ms])
    ax_w.set_xlabel("Tamanho do Padrão ($m$)")
    ax_w.set_ylabel("Speedup ($T_{CPU}/T_{GPU}$)")
    ax_w.set_title("Regime warm (texto residente em VRAM)")
    ax_w.legend(title="Base")

    # ---- Painel COLD: 1 barra por base (regime cold), hachurado ----
    cold = df[df["regime"] == "cold"].copy()
    if not cold.empty:
        xc = np.arange(len(cold))
        ax_c.bar(xc, cold["speedup"], 0.6,
                 color=[cores.get(b, "gray") for b in cold["base"]],
                 hatch="//", edgecolor="black", linewidth=0.4)
        ax_c.set_xticks(xc)
        ax_c.set_xticklabels([f"{b}\n(m={m})" for b, m in zip(cold["base"], cold["pattern_len"])])
    ax_c.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Empate (CPU=GPU)")
    ax_c.set_title("Regime cold (1ª consulta)")
    ax_c.set_ylabel("Speedup")
    ax_c.legend(loc="upper right")

    fig.suptitle("Speedup do Hash Chain: CPU Sequencial vs GPU OpenCL", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_speedup.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def grafico_tempos_absolutos(df: pd.DataFrame, out_dir: Path) -> None:
    """fig2: tempos absolutos WARM (CPU search vs GPU wall por consulta), com barras de erro (M2)."""
    warm = df[df["regime"] == "warm"].copy()
    bases = _bases_presentes(warm)
    if not bases:
        return
    fig, axes = plt.subplots(1, len(bases), figsize=(5.5 * len(bases), 4.8), sharey=True)
    if len(bases) == 1:
        axes = [axes]

    for ax, base in zip(axes, bases):
        d = warm[warm["base"] == base].sort_values("pattern_len")
        cpu_err = d["cpu_search_ms_median"] * (pd.to_numeric(d["cpu_search_ms_cv"], errors="coerce").fillna(0) / 100.0)
        gpu_err = d["gpu_query_wall_ms_median"] * (pd.to_numeric(d["gpu_query_wall_ms_cv"], errors="coerce").fillna(0) / 100.0)
        ax.errorbar(d["pattern_len"], d["cpu_search_ms_median"], yerr=cpu_err,
                    marker="o", linewidth=1.8, capsize=3, label="CPU sequencial (search)")
        ax.errorbar(d["pattern_len"], d["gpu_query_wall_ms_median"], yerr=gpu_err,
                    marker="s", linewidth=1.8, capsize=3, label="GPU OpenCL (wall por consulta)")
        ax.set_yscale("log")
        ax.set_xlabel("Tamanho do Padrão ($m$)")
        ax.set_title(f"Base: {base}")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Tempo (ms) — mediana, escala log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Tempo Absoluto no regime warm (mediana de 30 repetições)", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_tempos_absolutos.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def grafico_decomposicao(df: pd.DataFrame, out_dir: Path) -> None:
    """fig3: decomposição. Cold inclui o host-side (leitura+upload), que domina (M3)."""
    fig, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 3]})

    cores = {
        "file_read_ms": "#386cb0",         # leitura de disco (host)
        "gpu_upload_ms_median": "#7fc97f",  # RAM->VRAM (pin+memcpy+DMA)
        "kernel_ms_median": "#beaed4",      # kernel
        "d2h_ms": "#fdc086",                # VRAM->RAM
    }
    nomes = {
        "file_read_ms": "Leitura de disco (host)",
        "gpu_upload_ms_median": "Upload RAM→VRAM (pin+memcpy+DMA)",
        "kernel_ms_median": "Kernel (cálculo)",
        "d2h_ms": "Transferência D2H",
    }

    # ---- COLD: file_read + upload + kernel + d2h ----
    cold = df[df["regime"] == "cold"].sort_values("base")
    if not cold.empty:
        rotulos = [f"{b}\n(m={m})" for b, m in zip(cold["base"], cold["pattern_len"])]
        bottom = np.zeros(len(cold))
        for c in ["file_read_ms", "gpu_upload_ms_median", "kernel_ms_median", "d2h_ms"]:
            vals = cold[c].to_numpy(dtype=float)
            ax_c.bar(rotulos, vals, bottom=bottom, label=nomes[c], color=cores[c],
                     edgecolor="black", linewidth=0.4)
            bottom += vals
        ax_c.set_title("Regime cold (host-side domina)")
        ax_c.set_ylabel("Tempo (ms)")
        ax_c.grid(True, axis="y", linestyle="--", alpha=0.35)

    # ---- WARM: kernel + d2h (texto já em VRAM) ----
    warm = df[df["regime"] == "warm"].copy()
    bases = _bases_presentes(warm)
    ms = sorted(warm["pattern_len"].unique())
    x = np.arange(len(ms))
    largura = 0.8 / max(len(bases), 1)
    cores_base = dict(zip(ORDEM_BASES, sns.color_palette("Set2", len(ORDEM_BASES))))
    for i, base in enumerate(bases):
        sub = warm[warm["base"] == base].set_index("pattern_len").reindex(ms).reset_index()
        kern = sub["kernel_ms_median"].to_numpy(dtype=float)
        d2h = sub["d2h_ms"].to_numpy(dtype=float)
        ax_w.bar(x + i * largura, kern, largura, color=cores_base[base],
                 edgecolor="black", linewidth=0.4, label=f"{base} (kernel)")
        ax_w.bar(x + i * largura, d2h, largura, bottom=kern, color=cores_base[base],
                 edgecolor="black", linewidth=0.4, hatch="..", alpha=0.6)
    ax_w.set_xticks(x + largura * (len(bases) - 1) / 2)
    ax_w.set_xticklabels([str(m) for m in ms])
    ax_w.set_xlabel("Tamanho do Padrão ($m$)")
    ax_w.set_title("Regime warm (kernel + D2H; H2D≈0)")
    ax_w.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_w.legend(fontsize=8, ncol=2)

    handles, labels = ax_c.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=9)
    fig.suptitle("Decomposição do Tempo na GPU por Componente", y=1.06)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_memoria_gpu.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def grafico_vazao(df: pd.DataFrame, out_dir: Path) -> None:
    """fig4 (H2 §4.3): vazão efetiva (N/T, GB/s) no regime warm, CPU vs GPU por base e m.

    É vazão *efetiva* (texto/tempo): como o Hash Chain desloca de forma sublinear e, no warm,
    o texto já reside em VRAM, a vazão da GPU em m grande ultrapassa a banda do PCIe — por isso
    o eixo y é logarítmico (faixa de ~0,5 a ~60 GB/s)."""
    if "text_bytes" not in df.columns:
        return
    warm = df[df["regime"] == "warm"].copy()
    bases = _bases_presentes(warm)
    if not bases:
        return
    fig, axes = plt.subplots(1, len(bases), figsize=(5.5 * len(bases), 4.8), sharey=True)
    if len(bases) == 1:
        axes = [axes]

    for ax, base in zip(axes, bases):
        d = warm[warm["base"] == base].sort_values("pattern_len")
        ms = d["pattern_len"].to_numpy()
        x = np.arange(len(ms))
        v_cpu = [vazao_gbs(tb, t) for tb, t in zip(d["text_bytes"], d["cpu_search_ms_median"])]
        v_gpu = [vazao_gbs(tb, t) for tb, t in zip(d["text_bytes"], d["gpu_query_wall_ms_median"])]
        ax.bar(x - 0.2, v_cpu, 0.4, label="CPU sequencial", color="#386cb0",
               edgecolor="black", linewidth=0.4)
        ax.bar(x + 0.2, v_gpu, 0.4, label="GPU OpenCL", color="#7fc97f",
               edgecolor="black", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(m)) for m in ms])
        ax.set_xlabel("Tamanho do Padrão ($m$)")
        ax.set_title(f"Base: {base}")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Vazão efetiva (GB/s, escala log)")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("Vazão efetiva no regime warm ($N/T$): CPU sequencial vs GPU OpenCL", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_vazao.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def grafico_colisoes(df: pd.DataFrame, out_dir: Path) -> None:
    """fig5 (H2 §4.3): taxa de colisões do filtro $F$ no regime warm.

    Painel esquerdo: FP do filtro (candidatos que não são ocorrências). Painel direito: FP do
    memcmp (candidatos que passam pelo hash final $H_m$ mas não casam). São "verificações
    desnecessárias" — ligadas à divergência de wavefront da §3.3."""
    if "filter_candidates" not in df.columns:
        return
    warm = df[df["regime"] == "warm"].copy()
    bases = _bases_presentes(warm)
    if not bases:
        return
    cores = dict(zip(ORDEM_BASES, sns.color_palette("Set2", len(ORDEM_BASES))))
    ms = sorted(warm["pattern_len"].unique())
    x = np.arange(len(ms))
    largura = 0.8 / max(len(bases), 1)

    def fp(col_num: str, sub: pd.DataFrame) -> np.ndarray:
        cand = sub[col_num].to_numpy(dtype=float)
        mtc = sub["matches"].to_numpy(dtype=float)
        return np.where(cand > 0, (cand - mtc) / np.maximum(cand, 1.0) * 100.0, 0.0)

    fig, (ax_f, ax_m) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, col, titulo in ((ax_f, "filter_candidates", "FP do filtro $F$ (%)"),
                            (ax_m, "memcmp_calls", "FP do memcmp (%)")):
        for i, base in enumerate(bases):
            sub = warm[warm["base"] == base].set_index("pattern_len").reindex(ms).reset_index()
            ax.bar(x + i * largura, fp(col, sub), largura, label=base, color=cores[base],
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(x + largura * (len(bases) - 1) / 2)
        ax.set_xticklabels([str(int(m)) for m in ms])
        ax.set_xlabel("Tamanho do Padrão ($m$)")
        ax.set_title(titulo)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_f.set_ylabel("Falsos positivos (%)")
    ax_f.legend(title="Base")
    fig.suptitle("Taxa de colisões do filtro de fatores fracos (regime warm)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_colisoes.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def imprimir_tabela(df: pd.DataFrame) -> None:
    def linha(vals):
        return "| " + " | ".join(str(v) for v in vals) + " |"

    # H2 (§4.3): vazão (Throughput) = bytes do texto / tempo (ver vazao_gbs no módulo).
    tem_vazao = "text_bytes" in df.columns

    print("\n## Tabela — regime WARM (mediana ± CV%)\n")
    hdr = ["Base", "m", "CPU search (ms)", "GPU wall (ms)", "Speedup"]
    if tem_vazao:
        hdr += ["Vazão CPU (GB/s)", "Vazão GPU (GB/s)"]
    print(linha(hdr))
    print(linha(["---"] * len(hdr)))
    warm = df[df["regime"] == "warm"].sort_values(["base", "pattern_len"])
    for _, r in warm.iterrows():
        cols = [
            r["base"], int(r["pattern_len"]),
            f"{r['cpu_search_ms_median']:.2f} ± {r['cpu_search_ms_cv']:.1f}%",
            f"{r['gpu_query_wall_ms_median']:.2f} ± {r['gpu_query_wall_ms_cv']:.1f}%",
            f"{r['speedup']:.2f}×",
        ]
        if tem_vazao:
            cols += [
                f"{vazao_gbs(r['text_bytes'], r['cpu_search_ms_median']):.2f}",
                f"{vazao_gbs(r['text_bytes'], r['gpu_query_wall_ms_median']):.2f}",
            ]
        print(linha(cols))

    print("\n## Tabela — regime COLD (leitura cobrada de ambos os lados)\n")
    hdr = ["Base", "m", "leitura (ms)", "upload (ms)", "CPU total (ms)", "GPU total (ms)", "Speedup"]
    print(linha(hdr))
    print(linha(["---"] * len(hdr)))
    cold = df[df["regime"] == "cold"].sort_values(["base", "pattern_len"])
    for _, r in cold.iterrows():
        print(linha([
            r["base"], int(r["pattern_len"]),
            f"{r['file_read_ms']:.1f}",
            f"{r['gpu_upload_ms_median']:.1f} ± {r['gpu_upload_ms_cv']:.1f}%",
            f"{r['cpu_total_ms']:.1f}", f"{r['gpu_total_ms']:.1f}",
            f"{r['speedup']:.2f}×",
        ]))

    # H2 (§4.3): taxa de colisões do filtro F (falsos positivos) — "verificações desnecessárias".
    if "filter_candidates" in df.columns:
        print("\n## Tabela — colisões do filtro $F$ (regime warm)\n")
        hdr = ["Base", "m", "candidatos", "ocorrências", "FP filtro (%)", "FP memcmp (%)", "candidatos/KiB"]
        print(linha(hdr))
        print(linha(["---"] * len(hdr)))
        for _, r in warm.iterrows():
            cand = float(r["filter_candidates"])
            mtc = float(r["matches"]) if "matches" in df.columns else float(r.get("found_total", 0))
            memc = float(r["memcmp_calls"]) if "memcmp_calls" in df.columns else float("nan")
            fp_filtro = (cand - mtc) / max(cand, 1.0) * 100.0
            fp_memcmp = (memc - mtc) / max(memc, 1.0) * 100.0 if np.isfinite(memc) else float("nan")
            tb = float(r["text_bytes"]) if "text_bytes" in df.columns else float("nan")
            cand_kib = cand / (tb / 1024.0) if np.isfinite(tb) and tb > 0 else float("nan")
            print(linha([
                r["base"], int(r["pattern_len"]),
                f"{int(cand)}", f"{int(mtc)}",
                f"{fp_filtro:.2f}", f"{fp_memcmp:.2f}", f"{cand_kib:.3f}",
            ]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera as figuras do TCC a partir do timings_*.csv do diretório do run final."
    )
    parser.add_argument("--input", default="results-csv/final",
                        help="Diretório com timings_*.csv (ou o próprio arquivo). Padrão: results-csv/final.")
    parser.add_argument("--out-dir", default="figuras-benchmark",
                        help="Pasta de saída para os PDFs. Padrão: figuras-benchmark.")
    parser.add_argument("--strict", action="store_true",
                        help="Sai com código !=0 se a verificação de sanidade falhar.")
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 140

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = carregar(Path(args.input))
    verificar(df, args.strict)

    grafico_speedup(df, out_dir)
    grafico_tempos_absolutos(df, out_dir)
    grafico_decomposicao(df, out_dir)
    grafico_vazao(df, out_dir)
    grafico_colisoes(df, out_dir)

    print("\nArquivos gerados:")
    print(f"- {out_dir / 'fig1_speedup.pdf'}")
    print(f"- {out_dir / 'fig2_tempos_absolutos.pdf'}")
    print(f"- {out_dir / 'fig3_memoria_gpu.pdf'}")
    if "text_bytes" in df.columns:
        print(f"- {out_dir / 'fig4_vazao.pdf'}")
    if "filter_candidates" in df.columns:
        print(f"- {out_dir / 'fig5_colisoes.pdf'}")

    imprimir_tabela(df)


if __name__ == "__main__":
    main()
