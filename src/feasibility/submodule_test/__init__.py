"""Paired ostrich-vs-helhest_stack regression harness for submodule bumps.

Replays a fixed set of scenarios in both simulators, records a committed baseline of how they
behaved, and after a `git submodule update` reports what moved and flags runs that blew up. See
run_check.py's module docstring for the CLI and the baseline-update policy.
"""
