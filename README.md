# AI adoption, digital-skill diffusion and achievement gaps on a homophilous school network

Simulation code for the manuscript *[TITLE]* (submitted to *Mathematical Modelling and Numerical Simulation with Applications*).

The model simulates AI-tool adoption as a complex contagion on an SES-homophilous student network, digital-skill diffusion between peers, and latent achievement accumulation. Experiments E0–E20 reproduce every table and figure in the paper.

## Contents

| File | Purpose |
|------|---------|
| `AI_education_inequality.ipynb` | Main notebook: writes the modules below, runs the validation suite, all experiments, figures, and exports |
| `model.py` | Core model: `Config`, network generation, policies, dynamics, metrics, paired contrasts, mean-field boundary |
| `manuscript.py` | Primary configuration (`PC`) with the common 50-period evaluation horizon |
| `runmode.py` | `FAST_MODE` switch and replication scaling |
| `requirements.txt` | Pinned package set |

`model.py`, `manuscript.py` and `runmode.py` are identical to the `%%writefile` cells in the notebook; they are included so the code can be read without opening the notebook.

## Reproducing the results

**Google Colab** — open the notebook, run all cells. The default `FAST_MODE = True` runs ~1/10 of the replications and only verifies that the pipeline executes.

**Manuscript-grade run** — set `FAST_MODE = False` in `runmode.py`, restart the runtime, run all cells. Expect several hours on a single Colab CPU; E8 (Sobol, 512 × 8 replications) is the most expensive block.

**Local**
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute AI_education_inequality.ipynb
```

All experiments are seeded (`BASE_SEED = 20260901` plus per-experiment offsets); a fixed seed gives bit-identical output within a package-version set. Package versions used are exported to `results_robustness/full_run_metadata.json`.

## Output directories

`results/`, `results_extended/`, `results_robustness/`, `round2_targeted_results/`, `final_theta_sweep_results/` (CSV tables); `figs/`, `figs_extended/` (PNG figures). Combined archives are written as `aiedu_outputs.zip`, `aiedu_figs.zip`, `extended_results.zip`, `extended_figures.zip`, `all_results.zip`.

## Citation

See `CITATION.cff`. A DOI is minted through Zenodo for each tagged release.

## License

MIT — see `LICENSE`.
