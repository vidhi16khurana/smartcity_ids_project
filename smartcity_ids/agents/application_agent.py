"""
Application Layer Agent  (paper Section III-A / IV-B)

"built around sequence-aware models trained on request and log patterns."

A full sequence model (e.g. an RNN/Transformer over raw request logs) is out
of scope for this lightweight reference implementation; a GradientBoosting
classifier over engineered, request-pattern-derived features (rate, error
rate, payload entropy, endpoint fan-out) approximates the same detection
target while keeping the project runnable without a GPU or large corpora.
Swap in a real sequence model here without touching the rest of the pipeline
-- that substitution is exactly the modularity the paper's architecture
(Section III-A) is designed to allow.
"""
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

FEATURES = [
    "request_rate", "error_rate", "payload_entropy",
    "auth_failures", "distinct_endpoints",
]


@dataclass
class AgentResult:
    name: str
    score: float
    label: int
    row: dict


class ApplicationAgent:
    name = "application_layer_agent"
    features = FEATURES

    def __init__(self, threshold=0.5):
        self.model = GradientBoostingClassifier(random_state=0)
        self.threshold = threshold
        self.X_train = None

    def fit(self, df):
        X = df[self.features].values
        y = df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )
        self.model.fit(X_train, y_train)
        self.X_train = X_train
        return X_test, y_test

    def score_row(self, row) -> AgentResult:
        x = row[self.features].values.reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        p_attack = proba[1] if len(proba) > 1 else proba[0]
        return AgentResult(
            name=self.name, score=float(p_attack),
            label=int(p_attack >= self.threshold),
            row={f: float(row[f]) for f in self.features},
        )
