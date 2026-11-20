"""Fit a finite beam prefix against an existing calibrated intensity."""

import numpy as np


def fit_prefix(event, particle, initial_means):
    """Fit a finite prefix with the incumbent present as a profiled light field."""
    import coupled_model
    from balanced_search import feasible
    from track_coordinates import fit as optimize
    from value_only_receiver import seed_log_likelihood

    from staged_physics.search import seeds
    from terminated_primary_chart import TerminatedPrimaryChart

    incoming_basis = initial_means / initial_means.sum()

    class Incoming(coupled_model.Model):
        delta_scale = 0.0

        def __init__(self, order):
            super().__init__(event.pmt_ids, event.charges_pe, incoming_basis, (particle,), 4, order)
            self.range_bounds = tuple(self.bounds[9])
            self.width_bounds = tuple(self.bounds[11])
            self.response_bounds = self.bounds[-2:]

        @property
        def dimension(self):
            return super().dimension - 1

    coarse = Incoming(4)
    pool = []
    for _, point in seeds(coarse):
        point[-1] = np.clip(np.log(initial_means.sum()), *coarse.bounds[-1])
        chart = TerminatedPrimaryChart(coarse, point)
        for fraction in (0.04, 0.10, 0.25, 0.50, 1.0):
            trial = chart.initial.copy()
            trial[-1] = fraction
            full = chart.full(trial)[0]
            pool.append((seed_log_likelihood(coarse, full), full))
    x = max(pool, key=lambda row: row[0])[1]
    history = []
    for order in (8, 16, 32):
        model = Incoming(order)
        result, x = optimize(
            model, x, 40, 72, full_likelihood=True, chart=TerminatedPrimaryChart(model, x), check_initial=True
        )
        fine = Incoming(2 * order)
        state = fine(x, True)
        error = abs(state["ll"] - model(x, True)["ll"])
        valid = bool(feasible(fine, x) and fine.hardware_margin(x) > 1e-5)
        history.append(dict(order=order, error=error, calls=int(result.nfev), feasible=valid))
        if error < 0.01 and valid:
            return (fine, state), history
    return None, history
