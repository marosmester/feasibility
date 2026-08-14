"""Root-level glue code shared between the ostrich and helhest_stack demos --
not itself a robot model or simulator, just small utilities neither submodule
owns exclusively.

Layout:
  heightmap/   Simulator-agnostic heightmap asset, adapted to each submodule's
               native terrain representation (HeightMapReader).
"""
