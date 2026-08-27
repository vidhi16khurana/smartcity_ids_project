"""
Synthetic multi-layer smart-city telemetry generator.

Since public benchmarks like CICIDS2017 / UNSW-NB15 (used in the paper's
methodology) require an internet download that isn't available in every
environment, this module generates *structurally realistic* stand-ins:

  - Network-layer flow features   (packet/byte counts, flags, duration ...)
  - IoT/SCADA telemetry features  (sensor delta, command rate, auth failures ...)
  - Application-layer features    (request rate, error rate, payload entropy ...)

Normal traffic is drawn from layer-appropriate distributions. A configurable
number of *multi-stage campaigns* are injected: reconnaissance on the network
layer, followed shortly after by anomalous commands on the IoT layer, followed
by an injection attempt on the application layer -- the exact cross-layer
pattern the paper's Coordination Agent (Section III-B) is designed to catch.
Each campaign shares a `campaign_id` and tight timestamps so downstream code
can verify correlation.

Run directly to write CSVs to this directory; or `import` and call
`generate_dataset()` from other code.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
OUT_DIR = Path(__file__).parent


def _timestamps(n, start=0.0, rate=1.0, rng=None):
    """Poisson-ish arrival times, strictly increasing."""
    gaps = rng.exponential(scale=1.0 / rate, size=n)
    return start + np.cumsum(gaps)


def generate_dataset(n_normal=1500, n_campaigns=20, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    net_rows, iot_rows, app_rows = [], [], []
    t_cursor = 0.0

    # ---------------- normal background traffic ----------------
    net_ts = _timestamps(n_normal, start=0.0, rate=2.0, rng=rng)
    for ts in net_ts:
        net_rows.append(dict(
            timestamp=ts, campaign_id=-1, label=0,
            duration=rng.exponential(0.5),
            packet_count=rng.poisson(30),
            byte_count=rng.normal(4000, 800),
            unique_dst_ports=rng.integers(1, 3),
            syn_flag_ratio=np.clip(rng.normal(0.1, 0.03), 0, 1),
            avg_interarrival=rng.exponential(0.2),
        ))

    iot_ts = _timestamps(n_normal, start=0.0, rate=1.5, rng=rng)
    for ts in iot_ts:
        iot_rows.append(dict(
            timestamp=ts, campaign_id=-1, label=0,
            sensor_delta=rng.normal(0, 1),
            command_rate=rng.poisson(2),
            auth_failures=rng.poisson(0.05),
            firmware_checksum_ok=1,
            replay_gap=rng.exponential(5.0),
        ))

    app_ts = _timestamps(n_normal, start=0.0, rate=2.5, rng=rng)
    for ts in app_ts:
        app_rows.append(dict(
            timestamp=ts, campaign_id=-1, label=0,
            request_rate=rng.poisson(10),
            error_rate=np.clip(rng.normal(0.02, 0.01), 0, 1),
            payload_entropy=rng.normal(3.5, 0.4),
            auth_failures=rng.poisson(0.05),
            distinct_endpoints=rng.integers(1, 5),
        ))

    # ---------------- injected multi-stage campaigns ----------------
    # A fraction of attack events are deliberately "quiet" (closer to normal
    # distributions) so agents face genuine class overlap rather than a
    # trivially separable toy problem -- real detection metrics are rarely
    # a flat 1.00 across every score, and neither should this demo be.
    span = max(net_ts.max(), iot_ts.max(), app_ts.max())
    campaign_starts = rng.uniform(50, span - 50, size=n_campaigns)

    for cid, t0 in enumerate(campaign_starts):
        quiet = rng.random() < 0.25  # a stealthier variant of the campaign

        # Stage 1: network reconnaissance / port scanning
        n_recon = rng.integers(15, 40)
        recon_ts = t0 + np.sort(rng.uniform(0, 3, size=n_recon))
        for ts in recon_ts:
            scan_width = rng.integers(6, 25) if quiet else rng.integers(20, 200)
            syn_ratio = rng.normal(0.55, 0.1) if quiet else rng.normal(0.9, 0.05)
            net_rows.append(dict(
                timestamp=ts, campaign_id=cid, label=1,
                duration=rng.exponential(0.08 if quiet else 0.05),
                packet_count=rng.poisson(10 if quiet else 3),
                byte_count=rng.normal(1200, 400) if quiet else rng.normal(200, 50),
                unique_dst_ports=scan_width,
                syn_flag_ratio=np.clip(syn_ratio, 0, 1),
                avg_interarrival=rng.exponential(0.05 if quiet else 0.01),
            ))

        # Stage 2 (shortly after): anomalous IoT/SCADA commands
        t1 = t0 + rng.uniform(3, 8)
        n_iot = rng.integers(5, 15)
        iot_stage_ts = t1 + np.sort(rng.uniform(0, 2, size=n_iot))
        for ts in iot_stage_ts:
            iot_rows.append(dict(
                timestamp=ts, campaign_id=cid, label=1,
                sensor_delta=rng.normal(2.5 if quiet else 6, 1.5),
                command_rate=rng.poisson(6 if quiet else 15),
                auth_failures=rng.poisson(0.8 if quiet else 3),
                firmware_checksum_ok=rng.choice([0, 1], p=[0.3, 0.7] if quiet else [0.6, 0.4]),
                replay_gap=rng.exponential(1.5 if quiet else 0.2),
            ))

        # Stage 3 (shortly after): application-layer injection attempt
        t2 = t1 + rng.uniform(2, 6)
        n_app = rng.integers(5, 12)
        app_stage_ts = t2 + np.sort(rng.uniform(0, 2, size=n_app))
        for ts in app_stage_ts:
            app_rows.append(dict(
                timestamp=ts, campaign_id=cid, label=1,
                request_rate=rng.poisson(18 if quiet else 40),
                error_rate=np.clip(rng.normal(0.15 if quiet else 0.4, 0.08), 0, 1),
                payload_entropy=rng.normal(4.4 if quiet else 6.0, 0.5),
                auth_failures=rng.poisson(1.2 if quiet else 4),
                distinct_endpoints=rng.integers(4, 9) if quiet else rng.integers(8, 20),
            ))

    # A small amount of noisy-but-benign traffic (occasional bursts, odd
    # ports) further softens the class boundary so false positives aren't
    # trivially zero.
    n_noise = max(1, n_normal // 25)
    noise_ts = rng.uniform(0, span, size=n_noise)
    for ts in noise_ts:
        net_rows.append(dict(
            timestamp=ts, campaign_id=-1, label=0,
            duration=rng.exponential(0.3),
            packet_count=rng.poisson(12),
            byte_count=rng.normal(1500, 500),
            unique_dst_ports=rng.integers(4, 15),
            syn_flag_ratio=np.clip(rng.normal(0.35, 0.1), 0, 1),
            avg_interarrival=rng.exponential(0.05),
        ))
    for ts in rng.uniform(0, span, size=n_noise):
        iot_rows.append(dict(
            timestamp=ts, campaign_id=-1, label=0,
            sensor_delta=rng.normal(1.2, 0.8),
            command_rate=rng.poisson(5),
            auth_failures=rng.poisson(0.5),
            firmware_checksum_ok=1,
            replay_gap=rng.exponential(2.0),
        ))
    for ts in rng.uniform(0, span, size=n_noise):
        app_rows.append(dict(
            timestamp=ts, campaign_id=-1, label=0,
            request_rate=rng.poisson(16),
            error_rate=np.clip(rng.normal(0.08, 0.03), 0, 1),
            payload_entropy=rng.normal(4.0, 0.4),
            auth_failures=rng.poisson(0.3),
            distinct_endpoints=rng.integers(3, 7),
        ))

    net_df = pd.DataFrame(net_rows).sort_values("timestamp").reset_index(drop=True)
    iot_df = pd.DataFrame(iot_rows).sort_values("timestamp").reset_index(drop=True)
    app_df = pd.DataFrame(app_rows).sort_values("timestamp").reset_index(drop=True)
    return net_df, iot_df, app_df


if __name__ == "__main__":
    net_df, iot_df, app_df = generate_dataset()
    net_df.to_csv(OUT_DIR / "network_layer.csv", index=False)
    iot_df.to_csv(OUT_DIR / "iot_layer.csv", index=False)
    app_df.to_csv(OUT_DIR / "application_layer.csv", index=False)
    print(f"network_layer.csv      : {len(net_df):5d} rows  ({net_df.label.sum()} attack)")
    print(f"iot_layer.csv           : {len(iot_df):5d} rows  ({iot_df.label.sum()} attack)")
    print(f"application_layer.csv   : {len(app_df):5d} rows  ({app_df.label.sum()} attack)")
