# Explainable Multi-Agent AI Framework for Early Cyberattack Detection in Smart Cities

A runnable reference implementation of the architecture described in the
accompanying paper: specialized detection agents per infrastructure layer,
a coordination agent that correlates alerts across layers into multi-stage
attack campaigns, and an explainability agent that turns raw scores into
human-readable justifications for a security operator.

## What this is (and isn't)

This is a **design-blueprint-to-code translation**, not a reproduction of
the paper's reported numbers. Two honest limitations, upfront:

1. **Data.** The paper's methodology calls for CICIDS2017 and UNSW-NB15.
   Those require an internet download this environment doesn't have, so
   `data/generate_data.py` generates *structurally realistic synthetic*
   telemetry instead — normal traffic plus injected multi-stage campaigns
   (network recon → IoT/SCADA command tampering → application injection)
   with deliberate class overlap so metrics aren't a trivial 100%. Point
   the agents at real CICIDS2017/UNSW-NB15/IoT feature files instead by
   replacing `generate_dataset()`'s output with your own labeled
   DataFrames (same column names, or update `agents/*_agent.py`'s
   `FEATURES` lists to match).

2. **Explainability libraries.** `shap` and `lime` aren't installed in
   every environment. `agents/explainability_agent.py` uses them
   automatically if present, and otherwise falls back to two lightweight,
   dependency-free equivalents: impurity/coefficient-based global
   importance, and a perturbation-based local attribution ("mini-LIME").
   Install the real libraries (`pip install shap lime`) for production use
   — no other code changes needed.

Everything else — the three agents, the time-window cross-layer
correlation, confidence-weighted arbitration, and consolidated per-campaign
narrative — is a full, working implementation of Section III of the paper.

## Project layout

```
smartcity_ids/
├── data/
│   └── generate_data.py        # synthetic multi-layer telemetry + injected attacks
├── agents/
│   ├── network_agent.py        # gradient-boosted trees on flow features
│   ├── iot_agent.py            # random forest on IoT/SCADA telemetry
│   ├── application_agent.py    # gradient-boosted trees on app-layer features
│   ├── coordination_agent.py   # time-window correlation + confidence-weighted fusion
│   └── explainability_agent.py # SHAP/LIME with dependency-free fallback
├── dashboard/
│   ├── app.py                  # Flask app serving the operator view
│   └── templates/index.html
├── main.py                     # end-to-end pipeline
├── outputs/                    # alerts.json / campaigns_summary.csv (generated)
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python3 main.py                 # trains agents, runs detection + correlation,
                                 # writes outputs/alerts.json
python3 dashboard/app.py        # http://127.0.0.1:5000 — operator dashboard
```

`main.py` also prints, to the console:
- per-agent accuracy / precision / recall / F1 / AUC / false-positive rate
  (Section IV-D's evaluation metrics),
- global feature importance per agent (the retrospective SHAP-style audit
  described in Section III-C),
- the top consolidated, cross-layer campaigns with a human-readable
  narrative naming which features, from which layer, drove the alert.

## How the pieces map to the paper

| Paper section | Code |
|---|---|
| III-A, Overall Architecture | `agents/network_agent.py`, `iot_agent.py`, `application_agent.py` |
| III-B, Multi-Agent Coordination | `agents/coordination_agent.py` (`correlate`, confidence-weighted fusion) |
| III-C, Explainability Layer | `agents/explainability_agent.py` + `main.py::build_campaign_narrative` |
| III-D, Data Flow | `data/generate_data.py` (normalized, time-synchronized multi-layer stream) |
| IV-D, Evaluation Metrics | `main.py::evaluate` |
| "security dashboard ... municipal operators" | `dashboard/` |

## Extending this into something closer to a full deployment

- Swap `ApplicationAgent`'s GradientBoosting model for a real sequence
  model (small Transformer/LSTM) over raw request logs — the paper calls
  for a sequence-aware model; this reference keeps it lightweight and
  swaps in cleanly without touching the coordination or explainability code.
- Replace `generate_dataset()` with loaders for CICIDS2017 / UNSW-NB15 /
  your own IoT telemetry.
- `pip install shap lime` to get the real attribution methods instead of
  the built-in fallbacks — the interface (`global_importance`,
  `local_explanation`) doesn't change.
- Put each agent behind its own process/container communicating over a
  message bus, as Section V describes, instead of the in-process calls
  used here for simplicity.
