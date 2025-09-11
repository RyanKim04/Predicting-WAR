Generate **current-year MLB WAR projections**  
The repo provides a reproducible notebook workflow that outputs a CSV you can share or analyze.

[Open the analysis notebook](files/predictions_analysis.ipynb) •
[Run the final notebook](files/predictions_final.ipynb) •
[Latest CSV output](files/war_predictions.csv)

---

## Quickstart (local)

```bash
# 1) create and activate a virtual env (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2) install dependencies
pip install -r requirements.txt

# 3) execute the final notebook headlessly to (re)generate the CSV
jupyter nbconvert --to notebook --inplace --execute files/predictions_final.ipynb

# Output written to:
# files/war_predictions.csv
