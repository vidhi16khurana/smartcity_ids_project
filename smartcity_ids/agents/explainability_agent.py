"""
Lightweight Explainability Agent

Optimized for deployment on low-memory environments such as Render Free.
Uses model feature importance for global explanations and a lightweight
perturbation-based method for local explanations.
"""

import numpy as np


def backends():
    return {
        "shap_available": False,
        "lime_available": False,
        "backend": "Lightweight Explainability"
    }


# ------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# ------------------------------------------------

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


# ------------------------------------------------
# PREDICT POSITIVE PROBABILITY
# ------------------------------------------------

def _predict_proba_pos(model, X):

    proba = model.predict_proba(X)

    if (
        proba.ndim == 2
        and proba.shape[1] > 1
    ):

        return proba[:, 1]

    return proba.ravel()


# ------------------------------------------------
# LIGHTWEIGHT LOCAL EXPLANATION
# ------------------------------------------------

def local_explanation(
    model,
    X_background,
    x_row,
    feature_names,
    top_k=3
):

    x_row = np.asarray(
        x_row,
        dtype=float
    )

    baseline = _predict_proba_pos(
        model,
        x_row.reshape(1, -1)
    )[0]


    # Use a smaller background sample
    if len(X_background) > 500:

        sample = X_background[:500]

    else:

        sample = X_background


    medians = np.median(
        sample,
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
            "contribution": float(
                contributions[i]
            )
        }

        for i in order

    ]