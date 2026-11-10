"""Rydberg receiver physics model, pure NumPy/SciPy. (FR-RX-6, FR-LIN-1..7, TSD §14)

Core units inside this package (TSD §14.2): time in us, angular frequencies in rad/us
(``2 pi x MHz``), wavevectors in rad/m, velocities in m/s. SI conversions happen only in
``model.py``; ``readout.py`` holds the SI readout formulas.
"""
