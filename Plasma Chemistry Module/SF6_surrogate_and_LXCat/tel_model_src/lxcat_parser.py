"""
LXCat cross-section parser.

Parses the standard LXCat TXT download format and returns structured
cross-section data grouped by reaction type.

Usage:
    from lxcat_parser import parse_lxcat, summarize
    cs = parse_lxcat('data/lxcat/SF6_Biagi_full.txt')
    summarize(cs)
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class CrossSection:
    """One collision process from an LXCat file."""
    reaction_type: str          # ELASTIC, EXCITATION, ATTACHMENT, IONIZATION
    target: str                 # e.g. "SF6"
    product: str                # e.g. "SF5-", "SF5 +", "SF6(v4)"
    threshold_eV: float         # energy loss / threshold (0 for attachment)
    mass_ratio: float           # m/M for elastic, 0 otherwise
    energy_eV: np.ndarray       # energy grid
    sigma_m2: np.ndarray        # cross section in m^2
    comment: str = ""


@dataclass
class LXCatData:
    """All cross sections from one LXCat file."""
    species: str
    processes: List[CrossSection] = field(default_factory=list)

    def by_type(self, rtype: str) -> List[CrossSection]:
        return [p for p in self.processes if p.reaction_type == rtype]

    @property
    def elastic(self) -> List[CrossSection]:
        return self.by_type('ELASTIC')

    @property
    def excitation(self) -> List[CrossSection]:
        return self.by_type('EXCITATION')

    @property
    def attachment(self) -> List[CrossSection]:
        return self.by_type('ATTACHMENT')

    @property
    def ionization(self) -> List[CrossSection]:
        return self.by_type('IONIZATION')

    def total_cross_section(self, rtype: str, energy_eV: np.ndarray) -> np.ndarray:
        """Sum all processes of a given type, interpolated onto a common grid."""
        total = np.zeros_like(energy_eV)
        for p in self.by_type(rtype):
            total += np.interp(energy_eV, p.energy_eV, p.sigma_m2,
                               left=0.0, right=0.0)
        return total


def parse_lxcat(filepath: str) -> LXCatData:
    """Parse a standard LXCat download file.

    Returns an LXCatData object with all collision processes.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Determine species from filename
    basename = os.path.basename(filepath)
    if 'SF6' in basename:
        species = 'SF6'
    elif 'Ar' in basename:
        species = 'Ar'
    else:
        species = 'unknown'

    data = LXCatData(species=species)
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # Look for reaction type keywords
        if line in ('ELASTIC', 'EFFECTIVE', 'EXCITATION', 'IONIZATION', 'ATTACHMENT'):
            rtype = line
            i += 1

            # Line 2: target (and product for excitation/ionization)
            target_line = lines[i].strip() if i < n else ''
            i += 1

            target = species
            product = target_line
            if '->' in target_line:
                parts = target_line.split('->')
                target = parts[0].strip()
                product = parts[1].strip() if len(parts) > 1 else ''
            elif '<->' in target_line:
                parts = target_line.split('<->')
                target = parts[0].strip()
                product = parts[1].strip() if len(parts) > 1 else ''

            # Line 3: threshold or mass ratio (missing for attachment)
            threshold = 0.0
            mass_ratio = 0.0
            if rtype == 'ATTACHMENT':
                # No third line for attachment
                pass
            else:
                if i < n:
                    param_line = lines[i].strip()
                    try:
                        vals = param_line.split()
                        if rtype == 'ELASTIC' or rtype == 'EFFECTIVE':
                            mass_ratio = float(vals[0])
                        else:
                            threshold = float(vals[0])
                    except (ValueError, IndexError):
                        pass
                    i += 1

            # Skip comment lines until we find the data table start (-----)
            comment_lines = []
            while i < n and not lines[i].strip().startswith('-----'):
                comment_lines.append(lines[i].strip())
                i += 1
            comment = ' '.join(c for c in comment_lines if c and not c.startswith('SPECIES') and not c.startswith('PROCESS') and not c.startswith('PARAM') and not c.startswith('COLUMNS') and not c.startswith('UPDATED'))

            # Skip the dashes line
            if i < n and lines[i].strip().startswith('-----'):
                i += 1

            # Read energy/sigma pairs until closing dashes
            energies, sigmas = [], []
            while i < n and not lines[i].strip().startswith('-----'):
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    try:
                        energies.append(float(parts[0]))
                        sigmas.append(float(parts[1]))
                    except ValueError:
                        pass
                i += 1

            # Skip closing dashes
            if i < n and lines[i].strip().startswith('-----'):
                i += 1

            if energies:
                cs = CrossSection(
                    reaction_type=rtype,
                    target=target,
                    product=product,
                    threshold_eV=threshold,
                    mass_ratio=mass_ratio,
                    energy_eV=np.array(energies),
                    sigma_m2=np.array(sigmas),
                    comment=comment[:200],
                )
                data.processes.append(cs)
        else:
            i += 1

    return data


def summarize(data: LXCatData) -> Dict:
    """Print and return a summary of parsed cross-section data."""
    summary = {
        'species': data.species,
        'n_processes': len(data.processes),
        'n_elastic': len(data.elastic),
        'n_excitation': len(data.excitation),
        'n_attachment': len(data.attachment),
        'n_ionization': len(data.ionization),
    }

    print(f"LXCat data: {data.species}")
    print(f"  Total processes: {len(data.processes)}")
    for rtype in ['ELASTIC', 'EXCITATION', 'ATTACHMENT', 'IONIZATION']:
        procs = data.by_type(rtype)
        print(f"  {rtype}: {len(procs)} processes")
        for p in procs:
            emin, emax = p.energy_eV.min(), p.energy_eV.max()
            smax = p.sigma_m2.max()
            print(f"    {p.product[:40]:40s} E=[{emin:.3e}, {emax:.1f}] eV  "
                  f"σ_max={smax:.2e} m²  thr={p.threshold_eV:.2f} eV")

    return summary


if __name__ == '__main__':
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..'))

    for fname in ['SF6_Biagi_full.txt', 'Ar_Biagi_full.txt']:
        path = os.path.join(repo, 'data', 'lxcat', fname)
        if os.path.exists(path):
            print(f"\n{'='*60}")
            data = parse_lxcat(path)
            summarize(data)
