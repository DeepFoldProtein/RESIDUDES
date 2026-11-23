"""lantern.toml: where things live, budgets and the approval policy.

Looked up in this order: --config PATH, $LANTERN_CONFIG, <lantern>/lantern.toml. `lantern setup` writes it.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

LANTERN_DIR = Path(__file__).resolve().parents[2]          # .../RESIDUDES/lantern
REPO_DIR = LANTERN_DIR.parent                               # .../RESIDUDES
DEFAULT_PATH = LANTERN_DIR / "lantern.toml"


@dataclass
class Config:
    work_root: str = str(LANTERN_DIR / "work")              # dossiers: <work_root>/targets/<T>/dossier.json
    targets_root: str = ""                                  # existing predictions tree (lantern-engine layout), e.g. casp17/targets
    lantern_engine: str = str(REPO_DIR / "lantern-engine")
    lantern_qa: str = str(REPO_DIR / "lantern-qa")
    gpu_hours_per_target: float = 400.0
    agent_usd_per_step: float = 2.0
    agent_max_turns: int = 60
    model: str = "claude-opus-5-5"
    outward_actions: str = "human"                          # CASP submission etc. always need `lantern approve`
    extra: dict = field(default_factory=dict)
    path: str = ""

    @property
    def targets_dir(self) -> Path:
        return Path(self.work_root) / "targets"


def config_path(explicit: str | None = None) -> Path:
    return Path(explicit or os.environ.get("LANTERN_CONFIG") or DEFAULT_PATH)


def load(explicit: str | None = None) -> Config:
    p = config_path(explicit)
    cfg = Config(path=str(p))
    if not p.exists():
        return cfg
    with open(p, "rb") as fh:
        d = tomllib.load(fh)
    paths, budget, agent, approval = d.get("paths", {}), d.get("budget", {}), d.get("agent", {}), d.get("approval", {})
    if "foldmux" in paths and "lantern_engine" not in paths:   # the key before lantern-engine was renamed from foldmux
        paths["lantern_engine"] = paths["foldmux"]
    for k in ("work_root", "targets_root", "lantern_engine", "lantern_qa"):
        if k in paths:
            setattr(cfg, k, str(paths[k]))
    cfg.gpu_hours_per_target = float(budget.get("gpu_hours_per_target", cfg.gpu_hours_per_target))
    cfg.agent_usd_per_step = float(budget.get("agent_usd_per_step", cfg.agent_usd_per_step))
    cfg.agent_max_turns = int(agent.get("max_turns", cfg.agent_max_turns))
    cfg.model = str(agent.get("model", cfg.model))
    cfg.outward_actions = str(approval.get("outward_actions", cfg.outward_actions))
    cfg.extra = {k: v for k, v in d.items() if k not in ("paths", "budget", "agent", "approval")}
    return cfg


def _q(v) -> str:
    return f'"{v}"' if isinstance(v, str) else str(v)


def save(cfg: Config, path: Path) -> None:
    lines = [
        "# lantern.toml - written by `lantern setup`; edit freely.",
        "[paths]",
        f"work_root = {_q(cfg.work_root)}        # dossiers live in <work_root>/targets/<T>/",
        f"targets_root = {_q(cfg.targets_root)}  # existing predictions (lantern-engine layout), may be empty",
        f"lantern_engine = {_q(cfg.lantern_engine)}",
        f"lantern_qa = {_q(cfg.lantern_qa)}",
        "",
        "[budget]",
        f"gpu_hours_per_target = {cfg.gpu_hours_per_target}",
        f"agent_usd_per_step = {cfg.agent_usd_per_step}",
        "",
        "[agent]",
        f"model = {_q(cfg.model)}",
        f"max_turns = {cfg.agent_max_turns}",
        "",
        "[approval]",
        f"outward_actions = {_q(cfg.outward_actions)}   # submissions always need `lantern approve`",
        "",
    ]
    path.write_text("\n".join(lines))
