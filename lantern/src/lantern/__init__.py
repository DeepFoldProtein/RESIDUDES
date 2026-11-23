"""lantern: a structure-prediction campaign agent built on lantern-engine (inputs, predictions) and lantern-qa (pool QA).

Layers:
  lantern.config     lantern.toml (where things live, budgets, approval policy)
  lantern.resources  what this machine and cluster have right now
  lantern.catalog    databases, weights and tools the campaign needs, with freshness checks
  lantern.setup      first-run onboarding (`lantern setup`) and `lantern doctor`
  lantern.dossier    one JSON dossier per target: stage, evidence, decisions, runs, slate
  lantern.guard      guardrails applied to every shell command the agent runs
  lantern.agent      one agent step on one target, through the Claude Agent SDK
"""
__version__ = "0.1.0"
