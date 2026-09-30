"""End-to-end demo: the pump P-104 story.

Simulates 30 days of telemetry with a bearing wear fault starting day 14,
learns normal from the first 10 days, then streams the rest day by day
through detection, health scoring, and diagnosis, printing the story as it
unfolds.
"""
from sentinel.detectors import IsolationForestDetector, ZScoreDetector
from sentinel.diagnostics import classify, diagnose
from sentinel.explainer import TemplateExplainer
from sentinel.features import compute_features
from sentinel.health import HealthTracker
from sentinel.ingestion import validate
from sentinel.simulator import PumpSimulator

WINDOW = 60  # one hour of minute-resolution readings
FAULT_START = 14


def main() -> None:
    df = validate(
        PumpSimulator(seed=42).run(days=30, fault="bearing_wear", fault_start_day=FAULT_START)
    )

    baseline_raw = df.iloc[: 10 * 1440]
    zdet = ZScoreDetector().fit(baseline_raw)
    feats = compute_features(baseline_raw, window=WINDOW)
    iforest = IsolationForestDetector().fit(feats)
    train_scores = iforest.score(feats)

    tracker = HealthTracker(alpha=0.05, baseline=float(train_scores.mean()))
    explainer = TemplateExplainer()

    stream = df.iloc[10 * 1440 :].reset_index(drop=True)
    explained = False
    for day in range(10, 30):
        chunk = stream.iloc[(day - 10) * 1440 : (day - 9) * 1440]
        chunk_feats = compute_features(chunk, window=WINDOW)
        for v in iforest.score(chunk_feats):
            status = tracker.update(v)
        print(f"Day {day}: health={status['health']} trend={status['trend']} state={status['state']}")

        if status["state"] in ("advisory", "alert") and not explained:
            explained = True
            findings, stable = diagnose(feats, chunk_feats)
            hint, confidence = classify(findings)
            print()
            print(
                explainer.explain(
                    machine_id="P-104",
                    findings=findings,
                    stable=stable,
                    health=status["health"],
                    trend=status["trend"],
                    fault_hint=hint,
                    confidence=confidence,
                    window_days=max(day - FAULT_START, 1),
                    action="schedule bearing inspection within 7 days",
                )
            )
            print()

    if not explained:
        print("No advisory fired: check thresholds.")


if __name__ == "__main__":
    main()
