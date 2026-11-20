"""Test a second residual lobe against one known-species finite primary."""

from light_competition import physical_template
from retained_history import compare_on_history


def compare(choice, particle, trials, original):
    from coupled_model import Model

    former = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        candidates = []
        _, _, _, known = choice.finite_primary(particle)
        if known.get("count") == 1 and known.get("parameters"):
            candidates.append(known)
        selected = original.get("selected") or {}
        species = selected.get("particles") or ()
        if selected.get("count") == 1 and species and species[0] == particle:
            candidates.append(selected)
        templates = []
        for record in candidates:
            model = physical_template(choice.event, choice.model, record)
            if model is None:
                continue
            profile = choice.fit(model, fixed_ratios=True)
            if profile is not None:
                templates.append(
                    (0.65 * profile["charge"] + 0.35 * profile["joint"], model, record["family"])
                )
        if not templates:
            return []
        _, primary, family = max(templates, key=lambda row: row[0])
    finally:
        Model.delta_scale = former
    result = compare_on_history(choice, trials, primary)
    for option in result:
        option["label"] = "primary_residual_" + option["label"]
        option["primary_family"] = family
    return result
