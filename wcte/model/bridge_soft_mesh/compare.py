"""Compare qualified physical fits under one declared complexity penalty.

This is approximate BIC model selection, not a calibrated likelihood-ratio
test or an asserted marginal likelihood. Every realized direction counts
as a fitted parameter; internal bends receive no discounted dimension.
"""

MARGIN = 6.6349
