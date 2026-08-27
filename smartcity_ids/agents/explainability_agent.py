"""
Lightweight Explainability Agent

Uses dependency-free feature importance and local perturbation
explanations to keep memory usage low for cloud deployment.
"""

import numpy as np


# Disable heavy SHAP and LIME libraries
_HAS_SHAP = False
_HAS_LIME = False


def backends():
    return {
        "shap_available": False,
        "lime_available": False,
        "backend": "Lightweight Mini-LIME"
    }


# ==================================================
# GLOBAL FEATURE IMPORTANCE
# ==================================================

def global_importance(model, X_background, feature_names, top_k=5):

    if hasattr(model, "feature_importances_"):

        imp = model.feature_importances_

    elif hasattr(model, "coef_"):

        imp = np.abs(
            np.ravel(model.coef_)
        )

    else:

        imp = np.ones(
            len(feature_names)
        )

    order = np.argsort(
        -imp
    )[:top_k]

    return [
        {
            "feature": feature_names[i],
            "contribution": float(imp[i])
        }

        for i in order
    ]


# ==================================================
# PREDICT POSITIVE PROBABILITY
# ==================================================

def _predict_proba_pos(model, X):

    proba = model.predict_proba(X)

    if (
        proba.ndim == 2
        and proba.shape[1] > 1
    ):

        return proba[:, 1]

    return proba.ravel()


# ==================================================
# LIGHTWEIGHT LOCAL EXPLANATION
# ==================================================

def _mini_lime(
    model,
    X_background,
    x_row,
    feature_names,
    top_k
):

    baseline = _predict_proba_pos(
        model,
        x_row.reshape(1, -1)
    )[0]

    medians = np.median(
        X_background,
        axis=0
    )

    contributions = np.zeros(
        len(feature_names)
    )

    for j in range(
        len(feature_names)
    ):

        perturbed = x_row.copy()

        perturbed[j] = medians[j]

        new_pred = _predict_proba_pos(
            model,
            perturbed.reshape(1, -1)
        )[0]

        contributions[j] = (
            baseline - new_pred
        )

    order = np.argsort(
        -np.abs(contributions)
    )[:top_k]

    return [

        {
            "feature": feature_names[i],
            "contribution":
                float(contributions[i])
        }

        for i in order
    ]


# ==================================================
# LOCAL EXPLANATION
# ==================================================

def local_explanation(
    model,
    X_background,
    x_row,
    feature_names,
    top_k=5
):

    return _mini_lime(
        model,
        X_background,
        x_row,
        feature_names,
        top_k
    )