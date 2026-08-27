"""
Explainability Agent  (paper Section III-C / IV-C)

Produces:
  - global_importance(model, X, feature_names)      -> ranked feature list
      Uses shap.TreeExplainer when the `shap` package is installed;
      otherwise falls back to the model's own impurity-based
      feature_importances_ (tree models) or |coefficient| (linear models).

  - local_explanation(model, X, row_idx, feature_names, top_k)
      Uses lime.lime_tabular when `lime` is installed; otherwise falls back
      to a lightweight perturbation-based local attribution ("mini-LIME"):
      each feature is individually reset to the training-data median and the
      resulting drop in predicted probability is used as that feature's
      local contribution. This is cheap enough to run on every alert, which
      mirrors the paper's real-time-vs-retrospective trade-off (Section
      III-C / VII): fast local attributions live, full SHAP reserved for
      retrospective auditing.

Both paths return the SAME simple schema so the Coordination/Explainability
consolidation logic never needs to know which backend produced them:
    [{"feature": str, "contribution": float}, ...]   # sorted by |contribution|
"""
from __future__ import annotations
import numpy as np

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    _HAS_LIME = True
except ImportError:
    _HAS_LIME = False


def backends():
    return {"shap_available": _HAS_SHAP, "lime_available": _HAS_LIME}


# ---------------------------------------------------------------- global ----
def global_importance(model, X_background, feature_names, top_k=5):
    """Dataset-level feature importance for a trained agent model."""
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_background)
            sv = sv[1] if isinstance(sv, list) else sv
            mean_abs = np.abs(sv).mean(axis=0)
            order = np.argsort(-mean_abs)[:top_k]
            return [{"feature": feature_names[i], "contribution": float(mean_abs[i])}
                    for i in order]
        except Exception:
            pass  # fall through to the generic fallback below

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_))
    else:
        imp = np.ones(len(feature_names))
    order = np.argsort(-imp)[:top_k]
    return [{"feature": feature_names[i], "contribution": float(imp[i])} for i in order]


# ----------------------------------------------------------------- local ----
def _predict_proba_pos(model, X):
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()


def _mini_lime(model, X_background, x_row, feature_names, top_k):
    """Perturbation-based local attribution: cheap, dependency-free fallback."""
    baseline = _predict_proba_pos(model, x_row.reshape(1, -1))[0]
    medians = np.median(X_background, axis=0)
    contributions = np.zeros(len(feature_names))
    for j in range(len(feature_names)):
        perturbed = x_row.copy()
        perturbed[j] = medians[j]
        new_pred = _predict_proba_pos(model, perturbed.reshape(1, -1))[0]
        contributions[j] = baseline - new_pred  # drop in score when feature "normalized"
    order = np.argsort(-np.abs(contributions))[:top_k]
    return [{"feature": feature_names[i], "contribution": float(contributions[i])}
            for i in order]


def local_explanation(model, X_background, x_row, feature_names, top_k=5):
    """Per-instance explanation for a single alert. x_row is a 1D array."""
    if _HAS_LIME:
        try:
            explainer = LimeTabularExplainer(
                X_background, feature_names=feature_names, mode="classification",
                discretize_continuous=True, verbose=False,
            )
            exp = explainer.explain_instance(
                x_row, model.predict_proba, num_features=top_k
            )
            return [{"feature": f, "contribution": float(c)} for f, c in exp.as_list()]
        except Exception:
            pass  # fall through

    return _mini_lime(model, X_background, x_row, feature_names, top_k)
