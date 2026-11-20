"""Lightweight, dependency-free parsers for the VASP files committed in ``data/``.

The parsers read plain text only (no ASE/pymatgen) so they are fast on the
multi-MB OUTCARs and can run in CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CONVERGED_MARKER = "reached required accuracy"

_NELECT_RE = re.compile(r"^\s*NELECT\s*=\s*([-+0-9.]+)", re.MULTILINE)
_ZVAL_RE = re.compile(r"^[ \t]*ZVAL[ \t]*=[ \t]*([-+0-9. \t]+)$", re.MULTILINE)
_IONS_PER_TYPE_RE = re.compile(r"^[ \t]*ions per type[ \t]*=[ \t]*([0-9 \t]+)$", re.MULTILINE)
_NSW_RE = re.compile(r"^\s*NSW\s*=\s*(\d+)", re.MULTILINE)
_IVDW_RE = re.compile(r"^\s*IVDW\s*=\s*(\d+)", re.MULTILINE)
_ISPIN_RE = re.compile(r"^\s*ISPIN\s*=\s*(\d+)", re.MULTILINE)
_ENCUT_RE = re.compile(r"^\s*ENCUT\s*=\s*([0-9.]+)", re.MULTILINE)
_GGA_RE = re.compile(r"^\s*GGA\s*=\s*(\S+)", re.MULTILINE)
_LHFCALC_RE = re.compile(r"^\s*LHFCALC\s*=\s*\.?(T|F|TRUE|FALSE)\b", re.MULTILINE | re.IGNORECASE)
_HFSCREEN_RE = re.compile(r"^\s*HFSCREEN\s*=\s*([0-9.]+)", re.MULTILINE)
_KGRID_RE = re.compile(r"generate k-points for:\s+(\d+)\s+(\d+)\s+(\d+)")
_MAGNETIZATION_RE = re.compile(r"number of electron\s+([-+0-9.]+)\s+magnetization\s+([-+0-9.]+)")
# The double/triple spacing is the per-ionic-step summary line (electronic
# iterations print "free energy    TOTEN" with different spacing).
_TOTEN_RE = re.compile(r"free  energy   TOTEN\s*=\s*([-+0-9.]+)")
_SIGMA0_RE = re.compile(r"energy\(sigma->0\)\s*=\s*([-+0-9.]+)")
_VERSION_RE = re.compile(r"^\s*(vasp\.\S+)", re.MULTILINE)
_SITE_RE = re.compile(r"\[\s*([-+0-9.eE]+)[\s,]+([-+0-9.eE]+)[\s,]+([-+0-9.eE]+)\s*\]")

# Charge-state directory labels used in the public dataset.
CHARGE_LABELS = {"mi3": -3, "mi2": -2, "mi1": -1, "neu": 0, "pl1": 1, "pl2": 2}


@dataclass(frozen=True)
class OutcarSummary:
    """Key facts extracted from one OUTCAR."""

    path: str
    vasp_version: str | None
    nelect: float | None
    neutral_valence: float | None
    n_ions: int | None
    nsw: int | None
    n_ionic_steps: int
    is_relaxation: bool
    reached_required_accuracy: bool
    is_converged: bool
    final_energy_eV: float | None
    final_free_energy_eV: float | None
    final_magnetization: float | None
    ispin: int | None
    encut_eV: float | None
    gga: str | None
    lhfcalc: bool | None
    hfscreen: float | None
    ivdw: int | None
    kpoint_grid: str | None

    @property
    def charge_from_nelect(self) -> float | None:
        """Defect charge q = (neutral valence electrons) - NELECT."""
        if self.nelect is None or self.neutral_valence is None:
            return None
        return round(self.neutral_valence - self.nelect, 6)

    @property
    def functional(self) -> str:
        """Human-readable functional label inferred from GGA/LHFCALC/HFSCREEN."""
        if self.lhfcalc:
            if self.hfscreen is not None and abs(self.hfscreen - 0.2) < 1e-6:
                return "HSE06"
            return f"hybrid(HFSCREEN={self.hfscreen})"
        if self.gga in ("PE", "PBE"):
            return "PBE"
        return self.gga or "unknown"

    @property
    def dispersion(self) -> str:
        """Dispersion label for the IVDW tag (11 = D3 zero damping, 12 = D3BJ)."""
        return {None: "none", 0: "none", 11: "D3(zero)", 12: "D3(BJ)"}.get(
            self.ivdw, f"IVDW={self.ivdw}"
        )

    @property
    def unconverged_reason(self) -> str | None:
        """Why this calculation should not be used as a converged reference."""
        if self.final_energy_eV is None:
            return "no completed ionic step (no energy found)"
        if self.is_relaxation and not self.reached_required_accuracy:
            return (
                f"ionic relaxation did not reach required accuracy "
                f"({self.n_ionic_steps} ionic steps, NSW={self.nsw})"
            )
        return None


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _last_float(pattern: re.Pattern[str], text: str, group: int = 1) -> float | None:
    last = None
    for match in pattern.finditer(text):
        last = match
    return float(last.group(group)) if last else None


def _neutral_valence(text: str) -> tuple[float | None, int | None]:
    zval_text = _first(_ZVAL_RE, text)
    ions_text = _first(_IONS_PER_TYPE_RE, text)
    if zval_text is None or ions_text is None:
        return None, None
    zvals = [float(v) for v in zval_text.split()]
    counts = [int(v) for v in ions_text.split()]
    if len(zvals) != len(counts):
        return None, sum(counts)
    return sum(z * n for z, n in zip(zvals, counts, strict=True)), sum(counts)


def parse_outcar_text(text: str, path: str = "<memory>") -> OutcarSummary:
    """Parse the text of an OUTCAR into an :class:`OutcarSummary`."""
    nelect_text = _first(_NELECT_RE, text)
    nsw_text = _first(_NSW_RE, text)
    nsw = int(nsw_text) if nsw_text is not None else None
    neutral_valence, n_ions = _neutral_valence(text)
    reached = CONVERGED_MARKER in text
    is_relaxation = bool(nsw and nsw > 0)
    final_energy = _last_float(_SIGMA0_RE, text)
    lhf_text = _first(_LHFCALC_RE, text)
    kgrid = _KGRID_RE.search(text)
    ivdw_text = _first(_IVDW_RE, text)
    ispin_text = _first(_ISPIN_RE, text)
    encut_text = _first(_ENCUT_RE, text)
    hfscreen_text = _first(_HFSCREEN_RE, text)
    gga_text = _first(_GGA_RE, text)
    return OutcarSummary(
        path=path,
        vasp_version=_first(_VERSION_RE, text),
        nelect=float(nelect_text) if nelect_text is not None else None,
        neutral_valence=neutral_valence,
        n_ions=n_ions,
        nsw=nsw,
        n_ionic_steps=len(_TOTEN_RE.findall(text)),
        is_relaxation=is_relaxation,
        reached_required_accuracy=reached,
        is_converged=final_energy is not None and (reached or not is_relaxation),
        final_energy_eV=final_energy,
        final_free_energy_eV=_last_float(_TOTEN_RE, text),
        final_magnetization=_last_float(_MAGNETIZATION_RE, text, group=2),
        ispin=int(ispin_text) if ispin_text is not None else None,
        encut_eV=float(encut_text) if encut_text is not None else None,
        gga=None if gga_text in (None, "--") else gga_text,
        lhfcalc=None if lhf_text is None else lhf_text.upper().startswith("T"),
        hfscreen=float(hfscreen_text) if hfscreen_text is not None else None,
        ivdw=int(ivdw_text) if ivdw_text is not None else None,
        kpoint_grid="x".join(kgrid.groups()) if kgrid else None,
    )


def parse_outcar(path: str | Path) -> OutcarSummary:
    """Read and parse an OUTCAR file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_outcar_text(text, path=str(path))


def parse_defect_site_comment(comment: str) -> tuple[float, float, float] | None:
    """Extract the fractional defect site written by ShakeNBreak into a POSCAR comment.

    Two formats occur in the dataset:

    * ``0.0% N(Distort)=3 ~[0.4,0.5,0.4]`` (approximate, rounded to 0.1)
    * ``-0.0%_BDM__vac_3_Se[0.58333333 0.37211005 0.44742291]_-dNELECT=0...``
    """
    match = _SITE_RE.search(comment)
    if match is None:
        return None
    return tuple(float(v) for v in match.groups())  # type: ignore[return-value]


def read_poscar_comment(path: str | Path) -> str:
    """Return the first (comment) line of a POSCAR/CONTCAR."""
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip()


def find_structure(directory: Path, prefix: str) -> Path | None:
    """Return the first file in ``directory`` whose name starts with ``prefix``.

    Handles both ``POSCAR`` and the suffixed ``POSCAR_-30`` naming used in stage 1.
    """
    matches = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name.startswith(prefix)
    )
    return matches[0] if matches else None
