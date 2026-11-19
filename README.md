# Data Stream Mining final project · FP-T02
Open VSCode_files and install requirements.txt into the notebook kernel. Open src/Final_Project_Heat_Stress.ipynb and run all cells. The 20 cached ERA5 files support offline replay after installing the packages. Only ENF_UnimNeuron_PA.py and ENF_UnimNeuron_Safe.py are required beside the main notebook. The original exploratory notebook is in Old_Exploratory_Notebook_Not_Main.

Results include all 28 configurations. Section 8 of the notebook tunes the rule budget (`max_rules`, grid 1–100 in steps of 3) for UnimPA and UnimSafe on the 2016–2017 warm-up only, and reports a held-out check on 2018–2025. Its outputs are results/tables/fuzzy_max_rules_*.csv and results/figures/fuzzy_max_rules_tuning.png. The models in the main experiment keep the default five rules. results/tables/model_complexity.csv gives final sizes in model-specific units, rule_history.csv records checkpoint counts, and final_rules.csv gives feature centres, weights, support, width and consequent evidence. Report/main.tex and rolling.pdf form the editable Overleaf project. The paper's rolling plot uses representation B checkpoints for Online Bagging, UnimPA and UnimSafe. Package versions for the supplied run are in results/environment.json.

## Setup (virtual environment)

Python 3.12 is recommended (the supplied results were produced with 3.12).

```powershell
python -m venv venv            # or: uv venv venv --python 3.12
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name dsm-final --display-name "DSM final (venv 3.12)"
```

Then open `src/Final_Project_Heat_Stress.ipynb`, select the `venv` / "DSM final" kernel and Run All (~2–3 min).
Headless run: `python -m nbconvert --to notebook --execute src/Final_Project_Heat_Stress.ipynb --output-dir <dir>`.
Running the notebook overwrites files in `results/`; only the timing columns (`*_ms_per_event`) change between runs.

## Hyperparameter tuning

`src/tune_hyperparameters.py` runs a leakage-safe random search (selection on 2016–2017 only, then one replay on 2018–2025). Findings, selected parameters and baseline-vs-tuned scores: [results/tuning/FINDINGS.md](results/tuning/FINDINGS.md).
