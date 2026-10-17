# Methodology

- Unit orders, price-time priority and resting-order execution prices.
- Unmatched orders are cleared each tick; each agent may trade again next tick.
- Efficiency is realised surplus divided by optimal one-tick surplus × elapsed ticks, clipped to 0–100%. Negative ZI-U surplus remains visible in profit outputs.
- Seeds reproduce sequential runs. The engine uses Python and NumPy global RNGs; concurrent markets in one process are not supported.
- Head-to-head uncertainty uses run-level means and paired tests. Holm corrections are monotone. Surplus shares sum agents within each type.
- Synthetic markets only; results do not establish live-market profitability.

## Corrections from the notebook

Fixed the shared efficiency denominator, mixed-market agent labels, pooled-agent confidence intervals, group surplus shares and Holm adjustments. Persistent-order mode is rejected because the notebook's accounting assumes per-tick clearing. Original notebook files are unchanged.

## References

Gode & Sunder (1993), Gjerstad & Dickhaut (1998), and Roth & Erev (1995).

GD uses the notebook’s belief rule without added exploration. Homogeneous GD populations can stall with no trades; this is retained as a model limitation. Its vectorized calculation is checked against the original summation.
