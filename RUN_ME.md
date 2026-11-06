# Run order (after application_train.csv is in data/raw/)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python src/features.py
PYTHONPATH=src python src/train.py
PYTHONPATH=src python src/cutoff.py
PYTHONPATH=src python src/explain.py
PYTHONPATH=src python src/make_readme.py
streamlit run app.py
```

Claude runs all of this once the CSV is staged; this file is the record of what was run.
