"""
Network Traffic Agent  (paper Section III-A / IV-B)

"designed around gradient-boosted tree ensembles, which prior comparative
work has found to perform strongly on flow-based intrusion datasets"

A GradientBoostingClassifier stands in for the paper's boosted-tree +
lightweight-recurrent design; the recurrent short-term-pattern component is
approximated here with a rolling-window feature (avg_interarrival is already
a short-horizon signal) rather than a full sequence model, to keep the
reference implementation lightweight and dependency-free.
"""
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

FEATURES = [
    "duration", "packet_count", "byte_count",
    "unique_dst_ports", "syn_flag_ratio", "avg_interarrival",
]


@dataclass
class AgentResult:
    name: str
    score: float          # P(attack)
    label: int            # thresholded decision
    row: dict             # raw feature values, for explanation


class NetworkAgent:
    name = "network_traffic_agent"
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
