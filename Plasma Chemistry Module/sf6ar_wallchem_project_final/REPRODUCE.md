# Reproduction Instructions

## Prerequisites
```bash
pip install numpy scipy matplotlib
```

## Generate all figures
```bash
python wallchem_benchmark.py
```
This produces 4 figures in the current directory and prints the quantitative comparison table.

## Compile the report
```bash
cp figures/*.png .
pdflatex wallchem_report.tex
pdflatex wallchem_report.tex
```

## Using wall chemistry in your own code
```python
from sf6_wallchem_model import solve_model

# Without wall chemistry (original model)
r0 = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0.3, eta=0.12, wall_chem=False)

# With wall chemistry
wc = {'s_F': 0.00105, 's_SFx': 0.00056, 'p_fluor': 0.025,
      'p_wallrec': 0.007, 'p_FF': 0.0035, 'p_F_SF3': 0.5, 'p_F_SF4': 0.2}
rw = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0.3, eta=0.16, wall_chem=wc)

print(f"α without wall chem: {r0['alpha']:.1f}")
print(f"α with wall chem:    {rw['alpha']:.1f}")
```
