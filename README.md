[README.md](https://github.com/user-attachments/files/31108752/README.md)
# Fourier Neural Operators as Surrogates for the LLP Magnetic-Hysteresis Model

**Doctoral thesis (UFABC) — Wanys Arnaldo Antônio Rocha**

Fourier Neural Operators (FNO) as surrogates for the LLP hysteresis model, for the
**direct** ($H \mapsto B$) and **inverse** ($B \mapsto H$) problems, benchmarked against
DeepONet, EDLSTM and an Elman RNN, with an out-of-distribution (OOD) generalization study.

> 🇧🇷 Operadores Neurais de Fourier (FNO) como *surrogates* do modelo de histerese LLP,
> nos problemas **direto** ($H \mapsto B$) e **inverso** ($B \mapsto H$), comparados a
> DeepONet, EDLSTM e RNN de Elman, com estudo de generalização fora da distribuição (OOD).

---

## Repository structure / Estrutura do repositório

```
FNO-LLP-hysteresis/
├── thesis/                     # Final thesis PDF / PDF final da tese
│   ├── tese_UFABC_Wanys_Rocha.pdf
│   └── latex_source/           #   Full LaTeX source (compiles with pdflatex+bibtex)
├── direct_HtoB/                # Direct problem  H -> B  / Problema direto
│   ├── data_generation/        #   LLP data generators (semi-sine + OOD battery)
│   ├── FNO/                    #   FNO train/test/metrics + outputs
│   ├── DeepONet/              #   DeepONet baseline
│   ├── EDLSTM/                #   Encoder–decoder LSTM baseline (+ models/, utils/)
│   ├── RNN/                   #   Elman RNN baseline
│   └── results/              #   Consolidated figures & metric tables
├── inverse_BtoH/               # Inverse problem  B -> H  / Problema inverso
│   ├── data_generation/       #   Inverse LLP generator + OOD battery generator
│   ├── FNO/                   #   FNO train/test/OOD + trained model + logs
│   ├── DeepONet/              #   DeepONet baseline
│   ├── EDLSTM/                #   Encoder–decoder LSTM baseline (+ models/, utils/)
│   └── RNN/                   #   Elman RNN baseline
├── triangular_family/          # Triangular-family experiment / Experimento triangular
│   ├── training_direct/       #   Direct training scripts (4 architectures)
│   ├── training_inverse/      #   Inverse training scripts (4 architectures)
│   └── plots/                 #   Plots & metrics per architecture
│       ├── metrics_direct/    #     Raw metric CSVs, direct problem (4 architectures)
│       └── metrics_inverse/   #     Raw metric CSVs, inverse problem (FNO)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

Each architecture folder is **self-contained**: it ships the `models/` and `utils/`
modules it needs, its trained checkpoint (`*.pth`), and its plotting scripts, so the code
runs as originally executed.

> 🇧🇷 Cada pasta de arquitetura é **autossuficiente**: traz os módulos `models/` e `utils/`
> de que precisa, o modelo treinado (`*.pth`) e os scripts de plotagem, de modo que o código
> roda como foi executado originalmente.

---

## Requirements / Requisitos

Python 3.12. Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries: [`neuraloperator`](https://github.com/neuraloperator/neuraloperator) (FNO
— note the PyPI package is `neuraloperator` while the import name is `neuralop`),
`torch`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `openpyxl`.

---

## Reproducing the experiments / Reproduzindo os experimentos

The pipeline for each problem is **generate data → train → test/OOD → plot**.
Run each script from inside its own folder (paths are relative to the script).

> 🇧🇷 O fluxo de cada problema é **gerar dados → treinar → testar/OOD → plotar**.
> Rode cada script de dentro da sua própria pasta (os caminhos são relativos ao script).

**Direct problem ($H \mapsto B$):**

```bash
cd direct_HtoB/data_generation && python GeraBeH_LLP.py        # generate datasets (.npz)
cd ../FNO && python FNO_treino.py                              # train the FNO
python FNO_teste.py                                            # in-distribution test
python FNO_metrics.py                                          # metrics tables
```

**Inverse problem ($B \mapsto H$):**

```bash
cd inverse_BtoH/data_generation && python GeraFORCsInverseLLP.py   # base dataset
python OOD_battery_generator.py                                    # 5 OOD protocols
cd ../FNO && python FNO_inverso_train2.py                          # train the inverse FNO
python FNO_inverso_test_OOD.py                                     # OOD evaluation (5 protocols)
```

Baselines (DeepONet / EDLSTM / RNN) follow the same `*.train.py` → `*.test.py` pattern
inside their folders.

> 🇧🇷 Os *baselines* (DeepONet / EDLSTM / RNN) seguem o mesmo padrão `*.train.py` →
> `*.test.py` dentro de suas pastas.

---

## Trained models & datasets / Modelos treinados e datasets

Trained checkpoints (`*.pth`) and datasets (`*.npz`) are **included** in this repository
for full reproducibility (largest file ≈ 9 MB, well under GitHub's 100 MB limit). To keep a
fork lightweight, uncomment the `*.pth` / `*.npz` lines in `.gitignore` and regenerate them
with the scripts in each `data_generation/` folder.

> 🇧🇷 Os modelos treinados (`*.pth`) e os datasets (`*.npz`) estão **incluídos** para
> reprodutibilidade total (maior arquivo ≈ 9 MB). Para um fork leve, descomente as linhas
> `*.pth` / `*.npz` no `.gitignore` e regenere-os com os scripts em `data_generation/`.

---

## Key configurations / Configurações principais

| | Direct FNO ($H\to B$) | Inverse FNO ($B\to H$) |
|---|---|---|
| Fourier modes / channels | 16 / 64 | 16 / 64 |
| Factorization | Tucker (rank 0.42) | Tucker (rank 0.42) |
| Optimizer | AdamW (wd $10^{-4}$) | AdamW (wd $10^{-6}$) |
| LR schedule | cosine ($T_{max}=100$) | cosine ($T_{max}=500$) |
| Training loss | relative $L_2$ (LpLoss) | relative $L_2$ (LpLoss) |
| Epochs / patience | $\le 100$ / 30 | $\le 500$ / 100 |
| Batch / stopping | 32 / early stopping | 32 / early stopping |

Baselines are trained with the mean squared error (MSE). Full hyper-parameters for every
architecture are documented in the thesis (Chapters 5–7 and the triangular chapter).

**Methodological disclosure.** Early stopping monitors the *training* loss
(`best_train_loss`); no separate validation split was used, and every reported result
comes from a **single seed**. Both limitations are stated explicitly in the thesis. Read
the comparisons as orders of magnitude, not as last-digit rankings.

> 🇧🇷 **Ressalva metodológica.** A parada antecipada monitora a perda de *treino*; não
> houve conjunto de validação separado, e todos os resultados vêm de **uma única semente**.
> As duas limitações estão declaradas na tese. Leia as comparações em ordem de grandeza.

---

## Which file backs which table / Qual arquivo sustenta qual tabela

| Thesis table / Tabela | Source file / Arquivo-fonte |
|---|---|
| Direct OOD, per-scenario metrics | `direct_HtoB/results/tabela_metricas_experimentos.csv` and each architecture's `metricas_*.tex` |
| Triangular family, direct problem | `triangular_family/plots/metrics_direct/metricas_triangular_direto/*.csv` (FNO, DeepONet, EDLSTM, RNN) |
| Triangular family, resolution sweep ($N_t$) | same CSVs, rows `Nt_50` … `Nt_10000` (FNO and EDLSTM only) |
| Triangular family, inverse problem (FNO, physical units) | `triangular_family/plots/metrics_inverse/metricas_FNO_inverso_triangular.csv` |

The relative $L_2$ error $\mathcal{R}$ reported for the FNO in some tables is derived from
the exported RMSE as $\mathcal{R} = \mathrm{RMSE}/\mathrm{rms}(\text{target})$; the thesis
captions state this explicitly, including the caveat that a ratio of aggregates differs
slightly from a mean of per-sample ratios.

> 🇧🇷 O erro relativo $\mathcal{R}$ do FNO em algumas tabelas é derivado do RMSE exportado
> por $\mathcal{R} = \mathrm{RMSE}/\mathrm{rms}(\text{alvo})$; as legendas da tese declaram
> isso, inclusive a ressalva de que razão de agregados difere da média das razões.

---

## Citation / Citação

If you use this code, please cite the thesis:

```bibtex
@phdthesis{rocha_fno_llp,
  author = {Rocha, Wanys Arnaldo Ant\^onio},
  title  = {Fourier Neural Operators as Surrogates for the LLP Magnetic-Hysteresis Model},
  school = {Universidade Federal do ABC (UFABC)},
  year   = {2026}
}
```

## License / Licença

Source code: MIT (see `LICENSE`). The thesis PDF under `thesis/` is © the author / UFABC,
all rights reserved. / Código-fonte: MIT. O PDF da tese é © do autor / UFABC.
