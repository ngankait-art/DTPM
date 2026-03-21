# WebPlotDigitizer Guide for SF₆/Ar Paper Figures

## Setup

1. Open **https://automeris.io/WebPlotDigitizer/** in your browser
2. Download the cropped figure images from the outputs folder

You will digitize 6 subplots. Each one takes about 3–5 minutes.

---

## General Workflow (same for every subplot)

### Step 1: Load image
- Click **"Load Image"** → select the cropped PNG file

### Step 2: Choose plot type
- Select **"2D (X-Y) Plot"** → click **"Align Axes"**

### Step 3: Calibrate axes
- You will be asked to click **4 points**: two on the X-axis and two on the Y-axis
- Click on a known tick mark on the X-axis (e.g., the leftmost labeled tick)
- Click on another known tick mark on the X-axis (e.g., the rightmost labeled tick)
- Click on a known tick mark on the Y-axis (e.g., the bottom labeled tick)
- Click on another known tick mark on the Y-axis (e.g., the top labeled tick)
- Enter the numeric values for each point when prompted
- For **log-scale** axes: check the "Log Scale" checkbox for that axis

### Step 4: Add datasets
- By default there is one dataset. Rename it (e.g., "calculated")
- Click **"Add Dataset"** to add a second one (e.g., "experimental")

### Step 5: Digitize points
- Select a dataset in the left panel
- Click on each data point in the image
- The (x, y) coordinates appear in the data table on the right
- Switch to the next dataset and repeat

### Step 6: Export
- Click **"View Data"** → **"Download .CSV"**
- Save as e.g., `fig5a_data.csv`

---

## Per-Figure Calibration Instructions

### fig5a_ne_vs_power.png — nₑ vs Power
- **X-axis:** Linear. Click on `0` and `2000` tick marks → enter `0` and `2000`
- **Y-axis:** Linear. Click on `0` and `10` tick marks → enter `0` and `10`
- **Units:** X = Watts, Y = 10⁹ cm⁻³
- **Datasets:**
  - "calc" = filled (black) squares with solid line (5 points)
  - "exp" = open (white) squares with dashed line (5 points)

### fig5b_Te_vs_power.png — Tₑ vs Power
- **X-axis:** Linear. Click on `800` and `1800` (or closest ticks) → enter values
- **Y-axis:** Linear. Click on `0` and `5` → enter `0` and `5`
- **Units:** X = Watts, Y = eV
- **Datasets:**
  - "calc" = filled squares with solid line
  - "exp" = open squares with dashed line

### fig5c_ne_vs_Ar.png — nₑ vs Ar fraction (LOG SCALE)
- **X-axis:** Linear. Click on `0.0` and `1.0` → enter `0.0` and `1.0`
- **Y-axis:** ⚠️ **LOG SCALE** — check the "Log Scale" box! Click on `10⁹` and `10¹¹` → enter `1e9` and `1e11`
- **Units:** X = fraction (0–1), Y = cm⁻³
- **Datasets:**
  - "calc" = filled squares with solid line
  - "exp" = open squares with dashed line

### fig5d_Te_vs_Ar.png — Tₑ vs Ar fraction
- **X-axis:** Linear. Click on `0` and `100` → enter `0` and `100`
- **Y-axis:** Linear. Click on `0` and `5` → enter `0` and `5`
- **Units:** X = percent, Y = eV
- **Datasets:**
  - "calc" = filled squares with solid line
  - "exp" = open squares with dashed line

### fig7_alpha_vs_Ar.png — α vs Ar fraction (3 curves)
- **X-axis:** Linear. Click on `0.0` and `0.8` → enter `0.0` and `0.8`
- **Y-axis:** Linear. Click on `0` and `100` → enter `0` and `100`
- **Units:** X = fraction, Y = dimensionless
- **Datasets** (create 3):
  - "5mTorr" = triangles (▲) — bottom curve (4 points)
  - "10mTorr" = circles (●) — middle curve (4 points)
  - "20mTorr" = squares (■) — top curve (4 points)

### fig8_F_ne_vs_power.png — [F] and nₑ vs Power (dual Y-axis)
- ⚠️ This figure has **two Y-axes**. Digitize them separately.
- **First pass** (left Y-axis, [F]):
  - X-axis: Click on `0` and `2000` → enter `0` and `2000`
  - Y-axis: Click on `0` and `12` → enter `0` and `12` (units: 10¹³ cm⁻³)
  - Datasets: "F_calc" (filled squares), "F_exp" (open circles)
- **Second pass** (right Y-axis, nₑ):
  - Re-calibrate axes! X same, but Y-axis: click on `0` and `10` on the RIGHT axis → enter `0` and `10` (units: 10⁹ cm⁻³)
  - Dataset: "ne_calc" (filled squares connected by solid line)

---

## After Digitizing: Updating the Overlay Script

Once you have the CSV files, update `generate_overlays.py` like this:

```python
import numpy as np

# Replace the manually-digitized arrays with WebPlotDigitizer output:
data = np.genfromtxt('fig5a_data.csv', delimiter=',', skip_header=1)
fig5a_P_calc  = data[:, 0].tolist()    # first column = X
fig5a_ne_calc = data[:, 1].tolist()    # second column = Y
```

Or simply replace the arrays at the top of the script with your new values.

Then re-run:
```bash
python generate_overlays.py
```

---

## Tips

- **Zoom in** before clicking on points — WebPlotDigitizer has a zoom tool
- Click on the **center** of each marker (square/circle/triangle)
- For connected lines without distinct markers, use the **"Automatic Extraction"** tool: draw a box around the curve, select the color, and it traces the line automatically
- If you make a mistake, select the point and press **Delete**
- The "Adjust Points" tool lets you drag points to fine-tune position
- Export each dataset separately for cleaner organization
