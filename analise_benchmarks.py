import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def inferir_base(text_file: str) -> str:
    s = str(text_file).lower()
    if "english" in s:
        return "English"
    if "dna" in s:
        return "DNA"
    if "protein" in s or "proteins" in s:
        return "Proteins"
    return "Outros"


def carregar_csvs(input_path: Path) -> pd.DataFrame:
    arquivos: List[Path]
    if input_path.is_file():
        arquivos = [input_path]
    else:
        arquivos = sorted(input_path.rglob("*.csv"))

    frames = []
    for arq in arquivos:
        try:
            df = pd.read_csv(arq)
        except Exception:
            continue

        cols = {c.lower() for c in df.columns}
        # Filtra somente CSVs de benchmark do seu pipeline.
        if not ({"text_file", "pattern_len"} <= cols):
            continue

        df.columns = [c.strip().lower() for c in df.columns]
        df["source_file"] = arq.name
        frames.append(df)

    if not frames:
        raise RuntimeError("Nenhum CSV compatível encontrado no caminho informado.")

    return pd.concat(frames, ignore_index=True)


def extrair_execucoes_unicas(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicação por índice de ocorrência (index_order/index_value),
    # mantendo uma linha por execução/cenário por arquivo.
    drop_cols = {"index_order", "index_value"}
    cols_base = [c for c in df.columns if c not in drop_cols]
    return df[cols_base].drop_duplicates().copy()


def preparar_cpu_gpu(df_runs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df_runs.copy()

    for col in [
        "pattern_len",
        "chunk_size",
        "max_results",
        "h2d_ms",
        "kernel_ms",
        "d2h_ms",
        "gpu_total_ms",
        "gpu_wall_ms",
        "cpu_total_ms",
        "preprocessing_ms",
        "search_ms",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cpu_total_ms" not in df.columns:
        df["cpu_total_ms"] = pd.NA

    # Se CPU vier separado como preprocessing + search.
    has_pre = "preprocessing_ms" in df.columns
    has_search = "search_ms" in df.columns
    if has_pre and has_search:
        mask_cpu_nan = df["cpu_total_ms"].isna()
        df.loc[mask_cpu_nan, "cpu_total_ms"] = (
            df.loc[mask_cpu_nan, "preprocessing_ms"].fillna(0)
            + df.loc[mask_cpu_nan, "search_ms"].fillna(0)
        )

    if "gpu_total_ms" not in df.columns:
        df["gpu_total_ms"] = pd.NA

    # Se GPU total não existir, calcula pela soma de componentes.
    gpu_parts = [c for c in ["h2d_ms", "kernel_ms", "d2h_ms"] if c in df.columns]
    if gpu_parts:
        mask_gpu_nan = df["gpu_total_ms"].isna()
        df.loc[mask_gpu_nan, "gpu_total_ms"] = df.loc[mask_gpu_nan, gpu_parts].fillna(0).sum(axis=1)

    # gpu_wall_ms é o tempo de parede honesto (introduzido na fase TCC II);
    # CSVs antigos não têm essa coluna, então caímos para gpu_total_ms.
    if "gpu_wall_ms" not in df.columns:
        df["gpu_wall_ms"] = pd.NA
    df["gpu_wall_ms"] = df["gpu_wall_ms"].fillna(df["gpu_total_ms"])

    run_tag = df["run_tag"].astype(str).str.lower() if "run_tag" in df.columns else pd.Series("", index=df.index)

    # Heurística robusta para separar CPU e GPU.
    cpu_mask = df["cpu_total_ms"].notna() | run_tag.str.contains("cpu")
    gpu_mask = df["gpu_total_ms"].notna() | run_tag.str.contains("gpu") | df["gpu_wall_ms"].notna()

    cpu_raw = df[cpu_mask].copy()
    gpu_raw = df[gpu_mask].copy()

    chaves = ["text_file", "pattern_len"]

    cpu_agg = (
        cpu_raw.groupby(chaves, as_index=False)
        .agg(cpu_total_ms=("cpu_total_ms", "mean"))
        .dropna(subset=["cpu_total_ms"])
    )

    agg_map = {"gpu_total_ms": "mean", "gpu_wall_ms": "mean"}
    for c in ["h2d_ms", "kernel_ms", "d2h_ms"]:
        if c in gpu_raw.columns:
            agg_map[c] = "mean"

    gpu_agg = (
        gpu_raw.groupby(chaves, as_index=False)
        .agg(agg_map)
        .dropna(subset=["gpu_wall_ms"])
    )

    merged = pd.merge(cpu_agg, gpu_agg, on=chaves, how="inner")
    merged["base"] = merged["text_file"].map(inferir_base)
    # Prefere o tempo de parede honesto. Se vier de CSV antigo (sem gpu_wall_ms),
    # ele já foi preenchido com gpu_total_ms acima.
    merged["speedup"] = merged["cpu_total_ms"] / merged["gpu_wall_ms"]

    merged = merged.sort_values(["base", "pattern_len"]).reset_index(drop=True)
    return cpu_agg, gpu_agg, merged


def grafico_speedup(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 5.2))
    d = df.copy()
    
    # 1. Extrai os valores únicos numéricos e os ordena (8, 16, 32, 64, 128)
    ordem_numerica = sorted(d["pattern_len"].unique())
    # 2. Converte a lista ordenada para string, que será usada como gabarito do eixo X
    ordem_str = [str(int(x)) for x in ordem_numerica]
    
    d["pattern_len_cat"] = d["pattern_len"].astype(int).astype(str)

    sns.barplot(
        data=d,
        x="pattern_len_cat",
        y="speedup",
        hue="base",
        palette="Set2",
        order=ordem_str,  # <--- Esta linha obriga o gráfico a respeitar a ordem numérica!
        errorbar=None,
    )
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Empate (CPU = GPU)")
    plt.xlabel("Tamanho do Padrão ($m$)")
    plt.ylabel("Speedup ($T_{CPU}/T_{GPU}$)")
    plt.title("Speedup do Hash Chain: CPU Sequencial vs GPU OpenCL")
    plt.legend(title="Base", loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_speedup.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def grafico_tempos_absolutos(df: pd.DataFrame, out_dir: Path) -> None:
    bases = [b for b in ["DNA", "English", "Proteins", "Outros"] if b in set(df["base"])]
    if not bases:
        return

    fig, axes = plt.subplots(1, len(bases), figsize=(5.5 * len(bases), 4.8), sharey=True)
    if len(bases) == 1:
        axes = [axes]

    for ax, base in zip(axes, bases):
        d = df[df["base"] == base].sort_values("pattern_len")
        if d.empty:
            continue

        ax.plot(d["pattern_len"], d["cpu_total_ms"], marker="o", linewidth=1.8, label="CPU sequencial")
        ax.plot(d["pattern_len"], d["gpu_total_ms"], marker="s", linewidth=1.8, label="GPU OpenCL")
        ax.set_yscale("log")
        ax.set_xlabel("Tamanho do Padrão ($m$)")
        ax.set_title(f"Base: {base}")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Tempo (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Tempo de Execução Absoluto (Escala Logarítmica)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_tempos_absolutos.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def grafico_memoria_gpu(df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["h2d_ms", "kernel_ms", "d2h_ms"]
    if not set(cols).issubset(df.columns):
        return

    bases = [b for b in ["DNA", "English", "Proteins", "Outros"] if b in set(df["base"])]
    if not bases:
        return

    fig, axes = plt.subplots(1, len(bases), figsize=(5.7 * len(bases), 4.8), sharey=True)
    if len(bases) == 1:
        axes = [axes]

    cores = {"h2d_ms": "#7fc97f", "kernel_ms": "#beaed4", "d2h_ms": "#fdc086"}
    nomes = {
        "h2d_ms": "Transferência H2D (RAM→VRAM)",
        "kernel_ms": "Kernel (cálculo)",
        "d2h_ms": "Transferência D2H (VRAM→RAM)",
    }

    for ax, base in zip(axes, bases):
        d = df[df["base"] == base].sort_values("pattern_len")
        if d.empty:
            continue

        x = d["pattern_len"].astype(int).astype(str)
        bottom = pd.Series(0.0, index=d.index)

        for c in cols:
            ax.bar(x, d[c], bottom=bottom, label=nomes[c], color=cores[c], edgecolor="black", linewidth=0.4)
            bottom = bottom + d[c]

        ax.set_xlabel("Tamanho do Padrão ($m$)")
        ax.set_title(f"Base: {base}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Tempo (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.suptitle("Perfil de Tempo na GPU por Componente", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_memoria_gpu.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def imprimir_tabela_markdown(df: pd.DataFrame) -> None:
    tabela = df.copy()
    colunas = [
        "base",
        "text_file",
        "pattern_len",
        "cpu_total_ms",
        "gpu_wall_ms",
        "gpu_total_ms",
        "speedup",
    ]
    for c in ["h2d_ms", "kernel_ms", "d2h_ms"]:
        if c in tabela.columns:
            colunas.append(c)

    tabela = tabela[colunas].sort_values(["base", "pattern_len"]).reset_index(drop=True)

    renomear = {
        "base": "Base",
        "text_file": "Arquivo",
        "pattern_len": "m",
        "cpu_total_ms": "CPU médio (ms)",
        "gpu_wall_ms": "GPU parede (ms)",
        "gpu_total_ms": "GPU eventos (ms)",
        "speedup": "Speedup",
        "h2d_ms": "H2D (ms)",
        "kernel_ms": "Kernel (ms)",
        "d2h_ms": "D2H (ms)",
    }
    tabela = tabela.rename(columns=renomear)

    for c in tabela.columns:
        if c not in ["Base", "Arquivo", "m"]:
            tabela[c] = tabela[c].map(lambda x: f"{x:.4f}")

    # Impressão em Markdown sem dependência externa (tabulate)
    headers = list(tabela.columns)
    rows = tabela.astype(str).values.tolist()

    def linha(vals):
        return "| " + " | ".join(vals) + " |"

    print("\n## Tabela Resumida (Médias por Cenário)\n")
    print(linha(headers))
    print(linha(["---"] * len(headers)))
    for r in rows:
        print(linha(r))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Processa benchmarks CPU vs GPU do Hash Chain e gera gráficos acadêmicos em PDF."
    )
    parser.add_argument(
        "--input",
        default="results-csv",
        help="Arquivo CSV consolidado ou pasta com CSVs (padrão: results-csv).",
    )
    parser.add_argument(
        "--out-dir",
        default="figuras-benchmark",
        help="Pasta de saída para gráficos PDF (padrão: figuras-benchmark).",
    )
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    plt.rcParams["figure.dpi"] = 140

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = carregar_csvs(input_path)
    df_runs = extrair_execucoes_unicas(df_raw)
    _, _, df_cenarios = preparar_cpu_gpu(df_runs)

    if df_cenarios.empty:
        raise RuntimeError(
            "Não foi possível montar cenários CPU vs GPU. "
            "Verifique se há colunas cpu_total_ms/gpu_total_ms ou arquivos separados CPU/GPU."
        )

    # 1) Speedup
    grafico_speedup(df_cenarios, out_dir)

    # 2) Tempos absolutos
    grafico_tempos_absolutos(df_cenarios, out_dir)

    # 3) Perfil de memória GPU
    grafico_memoria_gpu(df_cenarios, out_dir)

    print("\nArquivos gerados:")
    print(f"- {out_dir / 'fig1_speedup.pdf'}")
    print(f"- {out_dir / 'fig2_tempos_absolutos.pdf'}")
    print(f"- {out_dir / 'fig3_memoria_gpu.pdf'}")

    imprimir_tabela_markdown(df_cenarios)


if __name__ == "__main__":
    main()
