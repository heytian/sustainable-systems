# Final Project for Spring 2026 1.C51 Machine Learning for Sustainable Systems

This repository contains two python scripts to test three imputation methods (LSTM, Interpolation, KNN) across 2 ablations (with seasonal encoding and season-stratified masking, versus no seasonal encoding and random masking) on a dataset containing atmospheric CO2 concentration XCO2. Here are the steps for setup:

1. Create a virtual environment (recommended)

```
python -m venv .venv
source .venv/bin/activate
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Retrieve dataset "co2_sam.csv" from https://huggingface.co/datasets/heytian/oco-3 and place under a newly created folder in the root named 'datasource'

4. Run each script
```
python impute-season-encode.py
python impute-no-season.py
```

The resultant plots from both scripts are in this root folder as pngs, for reference.
