"""
IoT / SCADA Agent  (paper Section III-A / IV-B)

"uses a smaller, resource-conscious classifier suited to constrained edge
deployment, reflecting the practical limits of the devices it monitors."

RandomForest with a small number of shallow trees stands in for that
resource-conscious classifier -- cheap to run and to explain, appropriate
for an edge-deployed agent.
"""
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURES = [
    "sensor_delta", "command_rate", "auth_failures",
    "firmware_checksum_ok", "replay_gap",
]


@dataclass
class AgentResult:
    name: str
    score: float
    label: int
    row: dict


class IoTAgent:
    name = "iot_scada_agent"
    features = FEATURES

    def __init__(self, threshold=0.5):
        self.model = RandomForestClassifier(
            n_estimators=60, max_depth=6, random_state=0
        )
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
