import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from flask import Flask, request, jsonify

# Config

RUTA_MODELOS = os.getenv("RUTA_MODELOS", "modelos")


# Cargar información de artefactos

def cargar_artefactos(ruta_modelos: str):
    return {
        "escalador": joblib.load(os.path.join(ruta_modelos, "scaler.joblib")),
        "ohe": joblib.load(os.path.join(ruta_modelos, "ohe.joblib")),
        "numericas": joblib.load(os.path.join(ruta_modelos, "numericas.joblib")),
        "categoricas": joblib.load(os.path.join(ruta_modelos, "categoricas.joblib")),
        "feature_columns": joblib.load(os.path.join(ruta_modelos, "feature_columns.joblib")),
        "clf": joblib.load(os.path.join(ruta_modelos, "model_clf.joblib")),
    }

def preparar_features(df_raw: pd.DataFrame, artefactos: dict) -> pd.DataFrame:
    numericas = artefactos["numericas"]
    categoricas = artefactos["categoricas"]
    escalador = artefactos["escalador"]
    ohe = artefactos["ohe"]
    feature_columns = artefactos["feature_columns"]

    faltantes = [c for c in (numericas + categoricas) if c not in df_raw.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    X_num = df_raw[numericas].copy()
    X_cat = df_raw[categoricas].copy()

    for c in categoricas:
        X_cat[c] = X_cat[c].astype("object").fillna("")

    X_num_scaled = escalador.transform(X_num)
    X_cat_ohe = ohe.transform(X_cat)

    X_all = sparse.hstack([sparse.csr_matrix(X_num_scaled), X_cat_ohe], format="csr")
    X_final = pd.DataFrame.sparse.from_spmatrix(X_all, columns=feature_columns)
    return X_final


# App

app = Flask(__name__)
ARTEFACTOS = cargar_artefactos(RUTA_MODELOS)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "modelos_path": RUTA_MODELOS})

@app.post("/predict")
def predict():
    """
    Espera JSON:
    {
      "record": { ... columnas originales ... }
    }
    """
    try:
        payload = request.get_json(force=True, silent=False)
        if not payload or "record" not in payload:
            return jsonify({"error": "Falta 'record' en el JSON"}), 400

        df_raw = pd.DataFrame([payload["record"]])
        X_final = preparar_features(df_raw, ARTEFACTOS)

        clf = ARTEFACTOS["clf"]
        y_pred = clf.predict(X_final)[0]

        resp = {
            "pred_clase": str(y_pred),
            "n_features": int(X_final.shape[1]),
        }

        # Probabilidades (si el modelo las soporta)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_final)[0]
            classes = list(getattr(clf, "classes_", []))
            if classes and len(classes) == len(proba):
                resp["prob_max"] = float(np.max(proba))
                resp["probs"] = {str(c): float(p) for c, p in zip(classes, proba)}

        return jsonify(resp)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    # Para desarrollo local
    app.run(host="0.0.0.0", port=5000, debug=True)
