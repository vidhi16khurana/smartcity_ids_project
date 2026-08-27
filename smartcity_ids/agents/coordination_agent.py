"""
Coordination Agent  (paper Section III-B)

Two responsibilities, matching the paper's description:

1. Temporal / cross-agent correlation:
     "checks whether alerts from different agents occur in a sequence or
     combination consistent with known multi-stage attack patterns, such as
     reconnaissance activity on the network layer followed shortly by
     anomalous device commands on the IoT layer."

   Implemented as a sliding time-window join: any set of per-agent alerts
   (from >=2 distinct layers) whose timestamps fall within `window_seconds`
   of each other is grouped into one multi-stage campaign, ordered exactly
   like the recon -> IoT -> app pattern the data generator injects.

2. Confidence-weighted arbitration:
     "applies a confidence-weighted arbitration step rather than a simple
     majority vote, so that an agent monitoring a data source with stronger
     evidence for a given event is weighted more heavily in the final
     decision."

   Implemented as a score fusion weighted by each agent's own confidence
   (distance from its decision threshold) rather than a flat average.
"""
from dataclasses import dataclass, field
from typing import List
import pandas as pd


@dataclass
class Alert:
    agent: str
    layer: str
    timestamp: float
    score: float
    row: dict


@dataclass
class CampaignAlert:
    campaign_score: float
    layers_involved: List[str]
    member_alerts: List[Alert] = field(default_factory=list)


class CoordinationAgent:
    def __init__(self, window_seconds: float = 12.0, min_layers: int = 2):
        self.window_seconds = window_seconds
        self.min_layers = min_layers

    @staticmethod
    def _confidence_weight(score, threshold=0.5):
        """Distance from the decision boundary -> how confident this agent is."""
        return abs(score - threshold) + 0.05  # small floor so nothing gets zero weight

    def correlate(self, alerts: List[Alert]) -> List[CampaignAlert]:
        """Group alerts across layers that fall inside the same time window."""
        if not alerts:
            return []
        alerts_sorted = sorted(alerts, key=lambda a: a.timestamp)
        campaigns = []
        used = [False] * len(alerts_sorted)

        for i, seed in enumerate(alerts_sorted):
            if used[i]:
                continue
            group = [seed]
            used[i] = True
            for j in range(i + 1, len(alerts_sorted)):
                if used[j]:
                    continue
                if alerts_sorted[j].timestamp - group[-1].timestamp <= self.window_seconds:
                    group.append(alerts_sorted[j])
                    used[j] = True
                else:
                    break

            layers = sorted(set(a.layer for a in group))
            if len(layers) >= self.min_layers:
                weights = [self._confidence_weight(a.score) for a in group]
                fused = sum(w * a.score for w, a in zip(weights, group)) / sum(weights)
                campaigns.append(CampaignAlert(
                    campaign_score=fused, layers_involved=layers, member_alerts=group
                ))
        return campaigns

    @staticmethod
    def to_dataframe(campaigns: List[CampaignAlert]) -> pd.DataFrame:
        rows = []
        for c in campaigns:
            rows.append(dict(
                campaign_score=round(c.campaign_score, 3),
                layers_involved=",".join(c.layers_involved),
                n_alerts=len(c.member_alerts),
                start_ts=round(min(a.timestamp for a in c.member_alerts), 2),
                end_ts=round(max(a.timestamp for a in c.member_alerts), 2),
            ))
        return pd.DataFrame(rows)
