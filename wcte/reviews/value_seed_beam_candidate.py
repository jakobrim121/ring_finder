"""One finite incoming-track candidate with the authorized beam species.

Uses the original charged-light source, charge likelihood and calibration. The
new research fitter supplies its surveyed entrance chart and finite endpoint
coordinate. Direction is initialized along the beam and remains fitted.
"""

import time
import numpy as np


def fit(event, background, particle, *, budget=72):
    import coupled_model
    from chain_comparison import counted
    from balanced_search import feasible
    from track_coordinates import fit as optimize

    from staged_physics.search import seeds
    from terminated_primary_chart import TerminatedPrimaryChart

    class Incoming(coupled_model.Model):
        def __init__(self, order):
            super().__init__(event.pmt_ids, event.charges_pe, background, (particle,), 4, order)
            self.particle = particle
            self.range_bounds = tuple(self.bounds[9])
            self.width_bounds = tuple(self.bounds[11])
            self.response_bounds = self.bounds[-2:]

        @property
        def dimension(self):
            # One surveyed coordinate (entry z) is fixed; xy and direction
            # are fitted. Keep the original amplitude-counting convention.
            return super().dimension - 1

    tick = time.process_time()
    coarse = Incoming(4)
    starts = []
    for _, complete in seeds(coarse):
        chart = TerminatedPrimaryChart(coarse, complete)
        for fraction in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
            point = chart.initial.copy()
            point[-1] = fraction
            x = chart.full(point)[0]
            from value_only_receiver import seed_log_likelihood

            starts.append((seed_log_likelihood(coarse, x), x.copy()))
    x = max(starts, key=lambda r: r[0])[1]
    seed_cpu = time.process_time() - tick
    evaluations = 0
    history = []
    for order in (8, 16, 32):
        model = Incoming(order)
        result, z = optimize(
            model,
            x,
            40,
            budget,
            full_likelihood=True,
            chart=TerminatedPrimaryChart(model, x),
            check_initial=True,
        )
        evaluations += int(result.nfev)
        fine = Incoming(2 * order)
        state = fine(z, True)
        error = abs(state["ll"] - model(z, True)["ll"])
        feasible_state = bool(feasible(fine, z) and fine.hardware_margin(z) > 1e-5)
        history.append(
            dict(
                order=order,
                evaluations=int(result.nfev),
                ll=state["ll"],
                error=error,
                feasible=feasible_state,
            )
        )
        x = z
        if error < 0.01 and feasible_state:
            fractions = counted(fine, state)
            return (
                fine,
                state,
                fractions,
                dict(
                    family="known_beam_finite_primary",
                    particles=(particle,),
                    order=order,
                    quadrature_ll_error=float(error),
                    parameters=x.tolist(),
                    count=fractions.shape[1] - 1,
                    supported=bool(fractions.shape[1] == 2 and np.any(fractions.argmax(1) == 1)),
                    evaluations=evaluations,
                    refinement_history=history,
                    cpu_seconds=time.process_time() - tick,
                    seed_cpu_seconds=seed_cpu,
                    dimension=fine.dimension,
                    incoming_species_supplied=True,
                    generated_energy_used=False,
                    beam_direction_is_initializer=True,
                    entry_z_fixed=True,
                ),
            )
    return (
        None,
        None,
        None,
        dict(
            family="known_beam_finite_primary",
            supported=False,
            reason="No feasible quadrature-qualified beam candidate",
            refinement_history=history,
            cpu_seconds=time.process_time() - tick,
            seed_cpu_seconds=seed_cpu,
        ),
    )
