"""residudes: a structure-prediction campaign agent built on residudes-engine (inputs, predictions) and residudes-qa (pool QA).

Layers:
  residudes.config     residudes.toml (where things live, budgets, approval policy)
  residudes.resources  what this machine and cluster have right now
  residudes.catalog    databases, weights and tools the campaign needs, with freshness checks
  residudes.setup      first-run onboarding (`residudes setup`) and `residudes doctor`
  residudes.dossier    one JSON dossier per target: stage, evidence, decisions, runs, slate
  residudes.guard      guardrails applied to every shell command the agent runs
  residudes.agent      one agent step on one target, through the Claude Agent SDK
"""
__version__ = "0.1.0"
