"""
tier3_picmcc/lxcat_parser.py
=============================

Minimal LXCat cross-section file parser.

LXCat cross-section files contain blocks separated by '-----' delimiter
lines, with a block-type keyword (ELASTIC, EFFECTIVE, EXCITATION,
IONIZATION, ATTACHMENT) followed by metadata and a two-column
(energy_eV, cross_section_m2) data table. This module parses one such
file into a list of ``CrossSection`` dataclasses, each exposing a
callable ``sigma(epsilon)`` that linearly interpolates the tabulated
cross section in energy space.

This parser covers only what the Tier 3 MCC module needs and is not
a general-purpose LXCat reader. Specifically it extracts:
- process type (elastic / inelastic / ionisation / attachment)
- threshold energy (for inelastic channels)
- the raw (E, sigma) pairs

References
----------
LXCat file format documentation at https://www.lxcat.net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class CrossSection:
    process: str         # "elastic" | "inelastic" | "ionization" | "attachment"
    name: str            # free-form description, e.g. "SF6 -> SF5+"
    threshold_eV: float  # 0.0 for elastic/attachment
    energy_eV: np.ndarray
    sigma_m2: np.ndarray
    comment: str = ""

    def sigma(self, epsilon: np.ndarray) -> np.ndarray:
        """Linearly interpolate the cross section at given energies.

        Below the tabulated range the cross section is clipped to zero.
        Above the tabulated range the last tabulated value is held
        constant (BOLSIG+ convention).
        """
        eps = np.atleast_1d(np.asarray(epsilon, dtype=np.float64))
        return np.interp(eps, self.energy_eV, self.sigma_m2,
                         left=0.0, right=self.sigma_m2[-1])


def parse_lxcat(path: Path, species: str = "SF6") -> List[CrossSection]:
    """Parse an LXCat file into a list of ``CrossSection`` objects.

    Parameters
    ----------
    path : Path
        Path to the LXCat plaintext file.
    species : str
        Species name filter (keeps only blocks whose target species
        matches). Case-sensitive substring match.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # LXCat uses lines of hyphens as block delimiters. Block structure:
    #   KEYWORD
    #   target name
    #   third-line metadata (threshold / mass ratio)
    #   optional parameter lines (e.g. SPECIES, PROCESS, PARAM)
    #   comment lines
    #   ------ (data start)
    #    E  sigma
    #    E  sigma
    #    ...
    #   ------ (data end)
    sections: List[CrossSection] = []

    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    i = 0
    keywords = {"ELASTIC", "EFFECTIVE", "EXCITATION",
                "IONIZATION", "ATTACHMENT"}

    while i < len(lines):
        line = lines[i].strip()
        if line in keywords:
            kw = line
            i += 1
            # second line: target species name (maybe with arrow)
            name_line = lines[i].strip() if i < len(lines) else ""
            i += 1
            # third line (except for ATTACHMENT): threshold or mass ratio
            threshold = 0.0
            if kw != "ATTACHMENT" and i < len(lines):
                third = lines[i].strip()
                try:
                    # First float on the line
                    threshold = float(third.split()[0])
                except (IndexError, ValueError):
                    threshold = 0.0
                i += 1
            # Skip until the first delimiter line of dashes
            while i < len(lines) and not _is_delim(lines[i]):
                i += 1
            if i >= len(lines):
                break
            i += 1  # past opening delim
            # Collect data rows until next delimiter
            energies: List[float] = []
            sigmas: List[float] = []
            while i < len(lines) and not _is_delim(lines[i]):
                parts = lines[i].split()
                if len(parts) >= 2:
                    try:
                        energies.append(float(parts[0]))
                        sigmas.append(float(parts[1]))
                    except ValueError:
                        pass
                i += 1
            if energies:
                proc = _classify(kw)
                if species in name_line or species == "":
                    sections.append(CrossSection(
                        process=proc,
                        name=name_line,
                        threshold_eV=float(threshold),
                        energy_eV=np.asarray(energies),
                        sigma_m2=np.asarray(sigmas),
                    ))
            # move past closing delim
            if i < len(lines) and _is_delim(lines[i]):
                i += 1
        else:
            i += 1

    return sections


def _is_delim(s: str) -> bool:
    s = s.strip()
    return len(s) >= 5 and set(s) <= {"-"}


def _classify(keyword: str) -> str:
    return {
        "ELASTIC": "elastic",
        "EFFECTIVE": "elastic",   # treat as elastic momentum transfer
        "EXCITATION": "inelastic",
        "IONIZATION": "ionization",
        "ATTACHMENT": "attachment",
    }[keyword]
