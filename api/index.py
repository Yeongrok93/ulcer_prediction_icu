"""Pressure Injury Risk Calculator — Flask app for Vercel (serverless).

Model: Gaussian Naive Bayes (SMOTE-trained) + isotonic recalibration, saved from
analysis.ipynb as a plain scikit-learn CalibratedClassifierCV (no imblearn dependency).

Feature order expected by the model (same as the notebook's feature_cols):
    0 RASS mean, 1 lower-limb muscle strength (0-10), 2 max temp, 3 min temp,
    4 log1p(daily incontinence episodes)
"""
import os
import math

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static"),
)

MODEL_PATH = os.path.join(ROOT_DIR, "model", "ulcer_prediction.pkl")
model = joblib.load(MODEL_PATH)

FEATURE_LABELS = [
    "Mean level of consciousness (RASS) during stay",
    "Mean lower-limb muscle strength during stay",
    "Highest body temperature during stay",
    "Lowest body temperature during stay",
    "Daily incontinence episodes (total ÷ ICU days)",
]


def _base_pipeline():
    """Return the underlying (scaler -> GaussianNB) pipeline inside CalibratedClassifierCV."""
    cc = model.calibrated_classifiers_[0]
    return getattr(cc, "estimator", None) or getattr(cc, "base_estimator")


def feature_contributions(x_row):
    """Exact per-feature log-likelihood-ratio decomposition of Gaussian Naive Bayes.

    log P(PI | x) / P(no PI | x) = log(prior ratio) + sum_i [ log N(x_i | mu_1i, var_1i)
                                                              - log N(x_i | mu_0i, var_0i) ]
    Positive value -> the feature pushes the prediction toward pressure injury.
    """
    pipe = _base_pipeline()
    scaler, gnb = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    z = scaler.transform(np.asarray(x_row, dtype=float).reshape(1, -1))[0]
    theta, var = gnb.theta_, gnb.var_
    contrib = []
    for i in range(len(z)):
        ll1 = -0.5 * math.log(2 * math.pi * var[1, i]) - (z[i] - theta[1, i]) ** 2 / (2 * var[1, i])
        ll0 = -0.5 * math.log(2 * math.pi * var[0, i]) - (z[i] - theta[0, i]) ** 2 / (2 * var[0, i])
        contrib.append(ll1 - ll0)
    return contrib


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f = request.form
        rass = float(f["rass"])
        temp_min = float(f["temp_min"])
        temp_max = float(f["temp_max"])
        strength = float(f["strength"])
        incont_total = float(f["incontinence_total"])
        icu_days = float(f["icu_days"])

        # ---- input validation (same plausible ranges used in model development)
        errors = []
        if not (-5 <= rass <= 4):
            errors.append("RASS must be between -5 and +4.")
        if not (0 <= strength <= 10):
            errors.append("Lower-limb muscle strength must be between 0 and 10.")
        if not (30 <= temp_min <= 43) or not (30 <= temp_max <= 43):
            errors.append("Body temperature must be between 30 and 43 °C.")
        if temp_min > temp_max:
            errors.append("Minimum temperature cannot exceed maximum temperature.")
        if incont_total < 0:
            errors.append("Incontinence episodes cannot be negative.")
        if icu_days < 0:
            errors.append("ICU length of stay cannot be negative.")
        if errors:
            return jsonify({"error": " ".join(errors)})

        # ---- derived feature: daily incontinence = total / days (window < 1 day treated as 1 day)
        obs_days = max(icu_days, 1.0)
        daily_incont = incont_total / obs_days
        x = [rass, strength, temp_max, temp_min, math.log1p(daily_incont)]

        proba = float(model.predict_proba(np.array([x]))[0, 1])

        contrib = feature_contributions(x)
        shown_vals = [rass, strength, temp_max, temp_min, round(daily_incont, 2)]
        items = [
            {"feature": FEATURE_LABELS[i], "value": shown_vals[i], "contribution": contrib[i]}
            for i in range(5)
        ]
        # keep only the more influential of the two temperature terms (as in the original app)
        if items[2]["contribution"] >= items[3]["contribution"]:
            items.pop(3)
        else:
            items.pop(2)
        positives = [it for it in items if it["contribution"] > 0]
        total_pos = sum(it["contribution"] for it in positives) or 1.0
        for it in positives:
            it["share"] = round(it["contribution"] / total_pos * 100, 1)
            it["contribution"] = round(it["contribution"], 3)
        positives.sort(key=lambda d: -d["share"])

        return jsonify({
            "probability": round(proba * 100, 1),
            "daily_incontinence": round(daily_incont, 2),
            "risk_factors": positives,
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}"})
    except ValueError:
        return jsonify({"error": "All fields must be numeric."})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
