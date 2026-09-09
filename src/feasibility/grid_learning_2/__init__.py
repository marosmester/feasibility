"""Divergence-field learning with a PER-CELL command field and an interpolation-free readout --
see design.md for the full argument, and grid_learning/design.md for the parts of the
problem statement (locality, relief normalisation, masked loss, mirror symmetry) that v2 inherits
unchanged.

Two things differ from feasibility.grid_learning: the commanded yaw rate is a [G, G] FIELD rather
than one scalar per map (so the trunk is control-free and every op consuming the command is 1x1),
and the heightmap grid is odd and origin-centred at 0.125 m so the label lattice is an exact
integer crop of the feature map -- no F.grid_sample, no readout offset.

Deliberately independent of feasibility.grid_learning as well as feasibility.learning (design.md
section 11): the grid convention genuinely differs, so a shared utils would have to serve both,
which is how two experiments start silently constraining each other. Constants are restated, not
imported; feasibility.heightmap and feasibility.comparator are shared infrastructure and ARE
imported.
"""
