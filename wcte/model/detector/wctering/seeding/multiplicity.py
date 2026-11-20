"""Optimized optical evidence for standard counting and high-purity H1 selection."""

from functools import lru_cache
from importlib.resources import files
import json
from wctering.seeding.decision_trees import score_features, decide
from wctering.seeding.detector_features import detector_evidence
from wctering.seeding.optical_evidence import evidence_from_bank, SCHEMA


@lru_cache(maxsize=1)
def load_model():
    model = json.loads(files("wctering.seeding").joinpath("data/unified_counter_v226.json").read_text())
    if model.get("physics_evidence_schema") != SCHEMA:
        raise RuntimeError("Unsupported v2.26 optical evidence schema")
    return model


def classify_bank(
    payload, pmt_ids, positions, charges, times, config, bank, summaries, *, detector_features=None
):
    from wctering.seeding.partition_support import (
        _final_partition_background_assignment,
    )
    from wctering.serialization import seed_json as _json_safe

    d = dict(payload)
    model = load_model()
    # Recreate the feature names and definitions expected by the frozen trees.
    # The saved model is read once; this call never trains or updates its weights.
    features = (
        detector_evidence(pmt_ids, positions, charges, times, d)
        if detector_features is None
        else dict(detector_features)
    )
    optical = evidence_from_bank(pmt_ids, charges, times, bank, summaries)
    features.update(optical)
    probabilities, high = score_features(features, model)
    # Availability masks prevent unsupported hypotheses from winning. The H1
    # purity selector is a separate score; class 3 represents three or more.
    predicted, selected, allowed = decide(probabilities, high, d, model["high_purity_threshold"])
    d["predicted_ring_class"] = predicted
    d["prediction_meaning"] = (
        "0=no reconstructible observable ring; 1=one ring; 2=two rings; 3=three or more rings"
    )
    d["configuration"] = config.as_dict()
    d["decision_rule"] = (
        "v2.26: available multiplicity scored using jointly optimized optical hypotheses; calibrated independent high-purity H1 selection"
    )
    # Return the matching count-specific seed labels. Later optical fits may
    # replace their partition, while these remain the explicit fallback.
    d["final_partition_background_assignment"] = _final_partition_background_assignment(d, predicted)
    d["unified_counter"] = dict(
        model_id=model["model_id"],
        class_order=[0, 1, 2, 3],
        class_scores=probabilities.tolist(),
        available_classes=allowed.tolist(),
        high_purity_h1_score=high,
        high_purity_threshold=model["high_purity_threshold"],
        score_semantics="truth-trained detector-evidence scores; purity depends on event population",
        evidence_features=features,
        hypothesis_path="direct forced hypotheses plus independently optimized optical likelihoods",
        optical_evidence_schema=SCHEMA,
    )
    is_high = config.mode == "high_purity_one_ring"
    d["one_ring_selected"] = selected if is_high else predicted == 1
    d["one_ring_selection"] = dict(
        mode=config.mode,
        one_ring_selected=d["one_ring_selected"],
        score=high,
        threshold=model["high_purity_threshold"] if is_high else None,
        model_id=model["model_id"],
        criterion=("calibrated H1 score and observable support" if is_high else "standard count equals one"),
    )
    d = _json_safe(d)
    return d, summaries.get(predicted, {}).get("_candidates")
