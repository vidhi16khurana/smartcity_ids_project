"""
Explainable Multi-Agent AI Framework for Early Cyberattack Detection
in Chandigarh Smart City.

This is a synthetic cyberattack detection simulation.

The system:
1. Generates smart city telemetry.
2. Trains Network, IoT and Application detection agents.
3. Detects suspicious activity independently.
4. Correlates alerts across multiple layers.
5. Identifies the likely attack type.
6. Generates a human-readable explanation.
7. Displays results for a Chandigarh Smart City dashboard.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flask import Flask, render_template, jsonify

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data.generate_data import generate_dataset
from agents.network_agent import NetworkAgent
from agents.iot_agent import IoTAgent
from agents.application_agent import ApplicationAgent
from agents.coordination_agent import CoordinationAgent, Alert
from agents.explainability_agent import (
    global_importance,
    local_explanation,
    backends
)


# ==================================================
# PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUT_DIR = BASE_DIR / "outputs"

OUT_DIR.mkdir(exist_ok=True)


# ==================================================
# FLASK APPLICATION
# ==================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)


# ==================================================
# CHANDIGARH SMART CITY CONFIGURATION
# ==================================================

CITY_NAME = "Chandigarh Smart City"

CHANDIGARH_LOCATIONS = [
    {
        "name": "Sector 17",
        "infrastructure": "Public Services and Citizen Network",
        "type": "City Center"
    },
    {
        "name": "IT Park",
        "infrastructure": "Smart City Network Infrastructure",
        "type": "Technology Zone"
    },
    {
        "name": "Manimajra",
        "infrastructure": "IoT and Smart Utility Infrastructure",
        "type": "Smart Infrastructure Zone"
    },
    {
        "name": "Industrial Area",
        "infrastructure": "Industrial IoT and SCADA Systems",
        "type": "Industrial Zone"
    },
    {
        "name": "Sector 43",
        "infrastructure": "Application and E-Governance Services",
        "type": "Digital Services Zone"
    }
]


# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate(model, X_test, y_test, name):

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "agent": name,

        "accuracy": round(
            accuracy_score(y_test, pred),
            4
        ),

        "precision": round(
            precision_score(
                y_test,
                pred,
                zero_division=0
            ),
            4
        ),

        "recall": round(
            recall_score(
                y_test,
                pred,
                zero_division=0
            ),
            4
        ),

        "f1": round(
            f1_score(
                y_test,
                pred,
                zero_division=0
            ),
            4
        ),

        "auc": round(
            roc_auc_score(y_test, proba),
            4
        ) if len(set(y_test)) > 1 else None,

        "false_positive_rate": round(
            (
                ((pred == 1) & (y_test == 0)).sum()
                / max((y_test == 0).sum(), 1)
            ),
            4
        )
    }

    return metrics


# ==================================================
# ATTACK ANALYSIS
# ==================================================

def classify_attack(campaign):

    layers = set(campaign.layers_involved)

    if (
        "network" in layers
        and "iot" in layers
        and "application" in layers
    ):
        return {
            "attack_type": "Coordinated Multi-Layer Cyberattack",
            "target": "Multiple Chandigarh Smart City Services",
            "description": (
                "Suspicious activity was detected simultaneously across "
                "network, IoT and application infrastructure."
            )
        }

    if "network" in layers and "iot" in layers:
        return {
            "attack_type": "Suspected IoT Botnet / Network Flooding Attack",
            "target": "Smart IoT and Network Infrastructure",
            "description": (
                "Correlated network and IoT anomalies suggest that "
                "compromised smart devices may be generating malicious traffic."
            )
        }

    if "network" in layers and "application" in layers:
        return {
            "attack_type": "Suspected Distributed Denial-of-Service (DDoS)",
            "target": "Network and Digital Application Services",
            "description": (
                "High network anomalies and correlated application activity "
                "indicate a possible coordinated traffic flooding attack."
            )
        }

    if "iot" in layers and "application" in layers:
        return {
            "attack_type": "Suspected Multi-Stage IoT Compromise",
            "target": "Smart Devices and Connected Services",
            "description": (
                "IoT anomalies followed by suspicious application activity "
                "suggest a possible compromise spreading across connected services."
            )
        }

    if "network" in layers:
        return {
            "attack_type": "Network Anomaly / Possible DDoS",
            "target": "Chandigarh Smart City Network",
            "description": (
                "Abnormal network behavior was detected by the Network Agent."
            )
        }

    if "iot" in layers:
        return {
            "attack_type": "IoT Device Anomaly",
            "target": "Smart City IoT Infrastructure",
            "description": (
                "Abnormal behavior was detected among monitored smart devices."
            )
        }

    return {
        "attack_type": "Application Layer Attack",
        "target": "Digital and E-Governance Services",
        "description": (
            "Suspicious behavior was detected in application-level activity."
        )
    }


# ==================================================
# LOCATION SELECTION
# ==================================================

def select_chandigarh_location(campaign, campaign_index):

    layers = set(campaign.layers_involved)

    if "network" in layers and "iot" in layers:
        location_index = 1

    elif "iot" in layers:
        location_index = 2

    elif "application" in layers:
        location_index = 4

    else:
        location_index = campaign_index % len(CHANDIGARH_LOCATIONS)

    return CHANDIGARH_LOCATIONS[location_index]


# ==================================================
# HUMAN-READABLE EXPLANATION
# ==================================================

def generate_campaign_explanation(
    campaign,
    attack_info,
    location
):

    layer_names = []

    for layer in campaign.layers_involved:

        if layer == "network":
            layer_names.append("Network Agent")

        elif layer == "iot":
            layer_names.append("IoT/SCADA Agent")

        elif layer == "application":
            layer_names.append("Application Agent")

    agents_text = ", ".join(layer_names)

    total_alerts = len(campaign.member_alerts)

    start_time = min(
        alert.timestamp
        for alert in campaign.member_alerts
    )

    end_time = max(
        alert.timestamp
        for alert in campaign.member_alerts
    )

    time_window = round(
        end_time - start_time,
        2
    )

    explanation = (
        f"The system detected a suspected "
        f"'{attack_info['attack_type']}' affecting "
        f"{location['name']} in Chandigarh. "
        f"{total_alerts} suspicious events were correlated by the "
        f"{agents_text}. "
        f"The alerts occurred within approximately "
        f"{time_window} seconds, which increased the likelihood that "
        f"they belong to the same coordinated attack campaign. "
        f"{attack_info['description']}"
    )

    return explanation


# ==================================================
# AI ASSESSMENT
# ==================================================

def generate_ai_assessment(
    campaign,
    attack_info,
    location,
    fused_score
):

    confidence = int(fused_score * 100)

    layers_text = ", ".join(
        campaign.layers_involved
    )

    return (
        f"AI Assessment: The system has {confidence}% confidence that "
        f"the correlated anomalies represent a {attack_info['attack_type']}. "
        f"The affected simulated location is {location['name']}, "
        f"and the primary target is {attack_info['target']}. "
        f"Detection was based on correlated evidence from: {layers_text}."
    )


# ==================================================
# BUILD TECHNICAL EXPLANATION
# ==================================================

def build_campaign_narrative(
    campaign,
    agents_by_name,
    top_k=3,
    max_per_layer=2
):

    lines = []

    seen_per_layer = {}

    for alert in sorted(
        campaign.member_alerts,
        key=lambda a: a.timestamp
    ):

        seen_per_layer.setdefault(
            alert.layer,
            0
        )

        if seen_per_layer[alert.layer] >= max_per_layer:
            continue

        seen_per_layer[alert.layer] += 1

        agent = agents_by_name[alert.agent]

        x_row = np.array([
            alert.row[feature]
            for feature in agent.features
        ])

        explanation = local_explanation(
            agent.model,
            agent.X_train,
            x_row,
            agent.features,
            top_k=top_k
        )

        important_features = []

        for item in explanation:

            feature_name = item["feature"]

            contribution = item["contribution"]

            feature_value = alert.row[feature_name]

            important_features.append(
                f"{feature_name}={feature_value:.2f} "
                f"(impact {contribution:+.3f})"
            )

        features_text = ", ".join(
            important_features
        )

        lines.append(
            f"{agent.name} detected suspicious behavior. "
            f"Key contributing factors: {features_text}"
        )

    return "\n".join(lines)


# ==================================================
# SEVERITY CLASSIFICATION
# ==================================================

def classify_severity(fused_score):

    if fused_score >= 0.90:
        return "CRITICAL"

    if fused_score >= 0.75:
        return "HIGH"

    if fused_score >= 0.50:
        return "MEDIUM"

    return "LOW"


# ==================================================
# RUN DETECTION PIPELINE
# ==================================================

def run_detection_pipeline():

    print("\n" + "=" * 78)
    print("CHANDIGARH SMART CITY CYBER THREAT DETECTION")
    print("Synthetic Cyberattack Simulation")
    print(f"Explainability backends: {backends()}")
    print("=" * 78)

    # ------------------------------------------------
    # STEP 1: GENERATE DATA
    # ------------------------------------------------

    print("\nGenerating Smart City telemetry...")

    net_df, iot_df, app_df = generate_dataset()

    # ------------------------------------------------
    # STEP 2: CREATE AGENTS
    # ------------------------------------------------

    agents = {
        "network": NetworkAgent(),
        "iot": IoTAgent(),
        "application": ApplicationAgent()
    }

    datasets = {
        "network": net_df,
        "iot": iot_df,
        "application": app_df
    }

    layer_of = {
        "network": "network",
        "iot": "iot",
        "application": "application"
    }

    # ------------------------------------------------
    # STEP 3: TRAIN AGENTS
    # ------------------------------------------------

    metrics = []

    print("\nTraining specialized AI detection agents...")

    for key, agent in agents.items():

        df = datasets[key]

        X_test, y_test = agent.fit(df)

        agent_metrics = evaluate(
            agent.model,
            X_test,
            y_test,
            agent.name
        )

        metrics.append(agent_metrics)

    print("\n--- Per-Agent Evaluation ---")

    print(
        pd.DataFrame(metrics).to_string(
            index=False
        )
    )

    # ------------------------------------------------
    # STEP 4: GLOBAL FEATURE IMPORTANCE
    # ------------------------------------------------

    print("\n--- Global Feature Importance ---")

    global_importance_results = {}

    for key, agent in agents.items():

        importance = global_importance(
            agent.model,
            agent.X_train,
            agent.features,
            top_k=3
        )

        global_importance_results[
            agent.name
        ] = importance

        ranked = ", ".join(
            f"{item['feature']} "
            f"({item['contribution']:.3f})"
            for item in importance
        )

        print(
            f"{agent.name}: {ranked}"
        )

    # ------------------------------------------------
    # STEP 5: GENERATE LOCAL ALERTS
    # ------------------------------------------------

    print(
        "\nMonitoring simulated Chandigarh Smart City infrastructure..."
    )

    all_alerts = []

    for key, agent in agents.items():

        df = datasets[key]

        for _, row in df.iterrows():

            result = agent.score_row(row)

            if result.label == 1:

                all_alerts.append(
                    Alert(
                        agent=agent.name,
                        layer=layer_of[key],
                        timestamp=float(
                            row["timestamp"]
                        ),
                        score=result.score,
                        row=result.row
                    )
                )

    print(
        f"Total Local Alerts Detected: "
        f"{len(all_alerts)}"
    )

    # ------------------------------------------------
    # STEP 6: COORDINATION AGENT
    # ------------------------------------------------

    print("\nCorrelating alerts across layers...")

    coordinator = CoordinationAgent(
        window_seconds=12.0,
        min_layers=2
    )

    campaigns = coordinator.correlate(
        all_alerts
    )

    print(
        f"Coordinated Attack Campaigns Identified: "
        f"{len(campaigns)}"
    )

    # ------------------------------------------------
    # STEP 7: BUILD EXPLAINABLE REPORT
    # ------------------------------------------------

    agents_by_name = {
        agent.name: agent
        for agent in agents.values()
    }

    report = []

    sorted_campaigns = sorted(
        campaigns,
        key=lambda campaign: -campaign.campaign_score
    )

    for index, campaign in enumerate(
        sorted_campaigns
    ):

        fused_score = round(
            campaign.campaign_score,
            3
        )

        severity = classify_severity(
            fused_score
        )

        attack_info = classify_attack(
            campaign
        )

        location = select_chandigarh_location(
            campaign,
            index
        )

        human_explanation = generate_campaign_explanation(
            campaign,
            attack_info,
            location
        )

        ai_assessment = generate_ai_assessment(
            campaign,
            attack_info,
            location,
            fused_score
        )

        technical_narrative = build_campaign_narrative(
            campaign,
            agents_by_name
        )

        report.append({

            "campaign_id":
                f"CAM-{index + 1:03d}",

            "city":
                CITY_NAME,

            "location":
                location["name"],

            "zone_type":
                location["type"],

            "attack_type":
                attack_info["attack_type"],

            "target":
                attack_info["target"],

            "fused_score":
                fused_score,

            "confidence":
                f"{int(fused_score * 100)}%",

            "severity":
                severity,

            "layers_involved":
                campaign.layers_involved,

            "n_alerts":
                len(campaign.member_alerts),

            "start_ts":
                round(
                    min(
                        alert.timestamp
                        for alert
                        in campaign.member_alerts
                    ),
                    2
                ),

            "end_ts":
                round(
                    max(
                        alert.timestamp
                        for alert
                        in campaign.member_alerts
                    ),
                    2
                ),

            "why_detected":
                human_explanation,

            "ai_assessment":
                ai_assessment,

            "technical_explanation":
                technical_narrative
        })

    # ------------------------------------------------
    # STEP 8: SAVE RESULTS
    # ------------------------------------------------

    all_campaigns_df = (
        coordinator.to_dataframe(campaigns)
    )

    all_campaigns_df.to_csv(
        OUT_DIR / "campaigns_summary.csv",
        index=False
    )

    results = {

        "generated_at":
            time.time(),

        "city":
            CITY_NAME,

        "simulation_type":
            "Synthetic Cyberattack Simulation",

        "explainability_backends":
            backends(),

        "per_agent_metrics":
            metrics,

        "global_feature_importance":
            global_importance_results,

        "n_total_local_alerts":
            len(all_alerts),

        "n_campaigns":
            len(campaigns),

        "top_campaigns":
            report[:5]
    }

    with open(
        OUT_DIR / "alerts.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print(
        f"\nResults saved to: "
        f"{OUT_DIR / 'alerts.json'}"
    )

    return results


# ==================================================
# DASHBOARD ROUTES
# ==================================================

@app.route("/")
def dashboard():

    return render_template(
        "index.html"
    )


@app.route(
    "/run-detection",
    methods=["POST"]
)
def run_detection():

    try:

        results = run_detection_pipeline()

        campaigns = results.get(
            "top_campaigns",
            []
        )

        critical_threats = sum(
            1
            for campaign in campaigns
            if campaign.get("severity") == "CRITICAL"
        )

        return jsonify({

            "success": True,

            "city":
                results.get(
                    "city",
                    CITY_NAME
                ),

            "simulation_type":
                results.get(
                    "simulation_type",
                    "Synthetic Cyberattack Simulation"
                ),

            "total_alerts":
                results.get(
                    "n_total_local_alerts",
                    0
                ),

            "total_campaigns":
                results.get(
                    "n_campaigns",
                    0
                ),

            "critical_threats":
                critical_threats,

            "campaigns":
                campaigns,

            "metrics":
                results.get(
                    "per_agent_metrics",
                    []
                )
        })

    except Exception as error:

        print("\nDETECTION ERROR:")
        print(str(error))

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ==================================================
# START APPLICATION
# ==================================================

if __name__ == "__main__":

    print("\nStarting Chandigarh Smart City IDS Dashboard...")

    print(f"Project directory: {BASE_DIR}")

    print(f"Template directory: {TEMPLATE_DIR}")

    print(f"Static directory: {STATIC_DIR}")

    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000
    )