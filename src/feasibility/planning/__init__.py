"""Planning + path following, shared by benchmarks/ and demos/.

    gated_lattice  helhest_stack's CostToGo with a per-arc error gate; tracing its policy
    arc_network    per-arc predicted error fields from a lattice_learning ArcDivergenceNet
    planners       PLANNERS, their thresholds/checkpoints (PlannerConfig), PlanContext, plan_path
    pure_pursuit   the pure-pursuit Warp kernel inside ostrich's captured step (PurePursuitSimulator)
    evaluation     judge: the verdict on one ostrich world that followed a path
"""
