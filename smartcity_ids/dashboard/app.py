"""
Minimal security-operator dashboard (paper Section III-D: "delivered to a
security dashboard intended for municipal operators").

Reads outputs/alerts.json (written by ../main.py) and renders it as a
human-readable page: per-agent evaluation metrics, and each consolidated
multi-stage campaign with its fused confidence score and plain-language
explanation -- the "concise, human-readable justification" the paper's
abstract promises operators, rather than a bare anomaly score.

Run:
    python3 main.py          # generates outputs/alerts.json
    python3 dashboard/app.py # serves http://127.0.0.1:5000
"""
import json
from pathlib import Path
from flask import Flask, render_template

BASE_DIR = Path(__file__).parent
ALERTS_PATH = BASE_DIR.parent / "outputs" / "alerts.json"

app = Flask(__name__)


@app.route("/")
def index():
    if not ALERTS_PATH.exists():
        return (
            "<h2>No results yet.</h2>"
            "<p>Run <code>python3 main.py</code> from the project root first "
            "to generate outputs/alerts.json, then refresh this page.</p>"
        )
    with open(ALERTS_PATH) as f:
        data = json.load(f)
    return render_template("index.html", data=data)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
