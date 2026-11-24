"""residudes.toml: where things live, budgets and the approval policy.

Looked up in this order: --config PATH, $RESIDUDES_CONFIG, <residudes>/residudes.toml, and finally a
lantern.toml left there before the rename.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

RESIDUDES_DIR = Path(__file__).resolve().parents[2]          # .../RESIDUDES/residudes
REPO_DIR = RESIDUDES_DIR.parent                               # .../RESIDUDES
DEFAULT_PATH = RESIDUDES_DIR / "residudes.toml"


@dataclass
class Config:
    work_root: str = str(RESIDUDES_DIR / "work")              # dossiers: <work_root>/targets/<T>/dossier.json
    targets_root: str = ""                                  # existing predictions tree (residudes-engine layout), e.g. casp17/targets
    residudes_engine: str = str(REPO_DIR / "residudes-engine")
    residudes_qa: str = str(REPO_DIR / "residudes-qa")
    gpu_hours_per_target: float = 400.0
    agent_usd_per_step: float = 2.0
    agent_max_turns: int = 60
    model: str = "claude-opus-5-5"
    outward_actions: str = "human"                          # CASP submission etc. always need `residudes approve`
    extra: dict = field(default_factory=dict)
    path: str = ""

    @property
    def targets_dir(self) -> Path:
        return Path(self.work_root) / "targets"


LEGACY_PATH = RESIDUDES_DIR / "lantern.toml"       # what the file was called before the rename


def config_path(explicit: str | None = None) -> Path:
    named = explicit or os.environ.get("RESIDUDES_CONFIG")
    if named:
        return Path(named)
    if not DEFAULT_PATH.exists() and LEGACY_PATH.exists():
        return LEGACY_PATH
    return DEFAULT_PATH


def load(explicit: str | None = None) -> Config:
    p = config_path(explicit)
    cfg = Config(path=str(p))
    if not p.exists():
        return cfg
    with open(p, "rb") as fh:
        d = tomllib.load(fh)
    paths, budget, agent, approval = d.get("paths", {}), d.get("budget", {}), d.get("agent", {}), d.get("approval", {})
    for old, new in (("foldmux", "residudes_engine"), ("lantern_engine", "residudes_engine"),
                     ("lantern_qa", "residudes_qa")):      # what these keys were called before the renames
        if old in paths and new not in paths:
            paths[new] = paths[old]
    for k in ("work_root", "targets_root", "residudes_engine", "residudes_qa"):
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
        "# residudes.toml - written by `residudes setup`; edit freely.",
        "[paths]",
        f"work_root = {_q(cfg.work_root)}        # dossiers live in <work_root>/targets/<T>/",
        f"targets_root = {_q(cfg.targets_root)}  # existing predictions (residudes-engine layout), may be empty",
        f"residudes_engine = {_q(cfg.residudes_engine)}",
        f"residudes_qa = {_q(cfg.residudes_qa)}",
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
        f"outward_actions = {_q(cfg.outward_actions)}   # submissions always need `residudes approve`",
        "",
    ]
    path.write_text("\n".join(lines))
