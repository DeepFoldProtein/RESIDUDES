"""Check that a seed directory follows the unified sample layout
(see casp17 inferences/output.md) and summarise its confidence numbers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SUMMARY_REQUIRED = ("model_name", "seed", "sample_idx", "n_tokens", "n_atoms", "plddt", "ptm", "ranking_score")
# Each entry lists acceptable forms. pae/pde are int8 logits (af3/pt2/of3/bt2)
# or reduced Å arrays (bt2_jax / ef2 writers) — see casp17 inferences/output.md.
ARRAYS_EXPECTED = (
    ("plddt_atom.npy",),
    ("plddt_token.npy",),
    ("pae_logits.npz", "pae.npy"),
    ("pde_logits.npz", "pde.npy"),
    ("contact_probs.npy",),
)
SEED_FILES = ("timing.json", "distogram_logits.npz")


@dataclass
class SampleReport:
    name: str
    atoms: int = 0
    summary: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    arrays: dict[str, str] = field(default_factory=dict)   # canonical name → file actually found

    @property
    def pae_form(self) -> str:
        f = self.arrays.get("pae_logits.npz", "")
        return "logits" if f.endswith(".npz") else ("Å" if f else "-")


@dataclass
class Report:
    seed_dir: Path
    samples: list[SampleReport] = field(default_factory=list)
    seed_missing: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and all(not s.missing for s in self.samples)

    def top(self) -> SampleReport | None:
        ranked = [s for s in self.samples if isinstance(s.summary.get("ranking_score"), (int, float))]
        if not ranked:
            return self.samples[0] if self.samples else None
        return max(ranked, key=lambda s: s.summary["ranking_score"])


def _count_atoms(cif: Path) -> int:
    n = 0
    with cif.open() as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
    return n


def check_seed_dir(seed_dir: Path) -> Report:
    seed_dir = Path(seed_dir)
    rep = Report(seed_dir=seed_dir)
    if not seed_dir.is_dir():
        rep.problems.append(f"not a directory: {seed_dir}")
        return rep
    for f in SEED_FILES:
        if not (seed_dir / f).is_file():
            rep.seed_missing.append(f)
    samples = sorted(p for p in seed_dir.glob("sample_*") if p.is_dir())
    if not samples:
        rep.problems.append("no sample_<i>/ directories")
        return rep
    for s in samples:
        sr = SampleReport(name=s.name)
        cif = s / "model.cif"
        if cif.is_file() and cif.stat().st_size > 0:
            sr.atoms = _count_atoms(cif)
            if sr.atoms == 0:
                sr.missing.append("model.cif has no atoms")
        else:
            sr.missing.append("model.cif")
        summ = s / "summary.json"
        if summ.is_file():
            try:
                sr.summary = json.loads(summ.read_text())
            except ValueError:
                sr.missing.append("summary.json unreadable")
            for k in SUMMARY_REQUIRED:
                if k not in sr.summary:
                    sr.missing.append(f"summary.{k}")
        else:
            sr.missing.append("summary.json")
        if not (s / "meta.json").is_file():
            sr.missing.append("meta.json")
        for alts in ARRAYS_EXPECTED:
            found = next((a for a in alts if (s / "arrays" / a).is_file()), None)
            if found is None:
                sr.missing.append("arrays/" + "|".join(alts))
            else:
                sr.arrays[alts[0]] = found
        rep.samples.append(sr)
    return rep


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def format_report(rep: Report, ref: Report | None = None) -> str:
    lines = [f"seed dir : {rep.seed_dir}"]
    if rep.problems:
        lines += [f"  ✗ {p}" for p in rep.problems]
        return "\n".join(lines)
    status = "OK" if rep.ok else "INCOMPLETE"
    lines.append(f"layout   : {status}  ({len(rep.samples)} samples"
                 + (f"; seed-level missing: {', '.join(rep.seed_missing)}" if rep.seed_missing else "") + ")")
    lines.append(f"  {'sample':9} {'atoms':>6} {'plddt':>7} {'ptm':>6} {'iptm':>6} {'gpde':>6} {'rank':>6}  missing")
    for s in rep.samples:
        d = s.summary
        lines.append(f"  {s.name:9} {s.atoms:6d} {_fmt(d.get('plddt'),1):>7} {_fmt(d.get('ptm')):>6} "
                     f"{_fmt(d.get('iptm')):>6} {_fmt(d.get('gpde')):>6} {_fmt(d.get('ranking_score')):>6}  "
                     f"{', '.join(s.missing) if s.missing else '-'}")
    top = rep.top()
    if top:
        d = top.summary
        lines.append(f"top      : {top.name}  model={d.get('model_name')} seed={d.get('seed')} "
                     f"n_tokens={d.get('n_tokens')} plddt={_fmt(d.get('plddt'),1)} ptm={_fmt(d.get('ptm'))}"
                     f"  pae/pde as {top.pae_form}")
        if ref is not None and ref.top():
            r = ref.top().summary
            dp = (d.get("plddt") or 0) - (r.get("plddt") or 0)
            lines.append(f"reference: {ref.seed_dir}")
            lines.append(f"           top plddt {_fmt(r.get('plddt'),1)} ptm {_fmt(r.get('ptm'))} "
                         f"→ Δplddt {dp:+.1f}  n_atoms ref {r.get('n_atoms')} vs {d.get('n_atoms')}")
    return "\n".join(lines)
