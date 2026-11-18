# ICU Pressure Injury Risk Calculator (Vercel)

Flask app serving the Gaussian Naïve Bayes model from `analysis.ipynb`
(`ulcer_prediction.pkl` = scikit-learn `CalibratedClassifierCV`, isotonic, no imblearn dependency).

```
ulcer_vercel/
├── api/index.py          # Flask app (serverless entry point)
├── templates/index.html  # English UI (6 inputs incl. ICU length of stay)
├── model/ulcer_prediction.pkl
├── requirements.txt
└── vercel.json
```

## Model inputs (order fixed by the notebook's `feature_cols`)
1. RASS mean  2. lower-limb muscle strength (0–10)  3. max temp  4. min temp
5. `log1p(total incontinence episodes / max(ICU days, 1))`

The app derives item 5 from the two form fields `incontinence_total` and `icu_days`.

## Run locally
```
pip install -r requirements.txt
python api/index.py          # http://localhost:5000
```

## Deploy
```
vercel login
vercel --prod
```
When the notebook is re-run, copy the new pickle:
`copy ..\ulcer_prediction.pkl model\` and redeploy.
