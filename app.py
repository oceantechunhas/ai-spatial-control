import mikeio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from scipy.interpolate import griddata

print("1. Reading MIKE files...")
ds_before = mikeio.read("before.dfsu")
ds_after = mikeio.read("after.dfsu")

print("2. Extracting Significant Wave Height (Hm0) data...")
wave_before = ds_before["Sign. Wave Height"][-1]
wave_after = ds_after["Sign. Wave Height"][-1]

print("3. Extracting (X, Y) coordinates...")
xy_target = wave_before.geometry.element_coordinates[:, :2] 
xy_source = wave_after.geometry.element_coordinates[:, :2]

x_ori = xy_target[:, 0]
y_ori = xy_target[:, 1]

val_before = wave_before.to_numpy().flatten()
val_after = wave_after.to_numpy().flatten()

print("4. Performing pure mathematical interpolation...")
val_after_interp = griddata(xy_source, val_after, xy_target, method='linear')

if np.isnan(val_after_interp).any():
    val_after_nearest = griddata(xy_source, val_after, xy_target, method='nearest')
    mask = np.isnan(val_after_interp)
    val_after_interp[mask] = val_after_nearest[mask]

print("5. Calculating reclamation impact...")
diff_val = val_after_interp - val_before
wave_impact = wave_before.copy()
wave_impact.values = diff_val

print("6. Generating realistic spatial impact map...")
fig, ax = plt.subplots(figsize=(12, 9))

wave_impact.plot(ax=ax, cmap="coolwarm")

# ========================================================
# MATPLOTLIB TRICK (Double Coastline)
# ========================================================
num_lines_initial = len(ax.lines)
num_cols_initial = len(ax.collections)

try:
    wave_before.geometry.plot.outline(ax=ax)
except AttributeError:
    wave_before.geometry.plot(ax=ax, plot_type='outline')

for line in ax.lines[num_lines_initial:]:
    line.set_color('black')
    line.set_linewidth(1.5)
for col in ax.collections[num_cols_initial:]:
    col.set_color('black')
    col.set_linewidth(1.5)

num_lines_mid = len(ax.lines)
num_cols_mid = len(ax.collections)

try:
    wave_after.geometry.plot.outline(ax=ax)
except AttributeError:
    wave_after.geometry.plot(ax=ax, plot_type='outline')

for line in ax.lines[num_lines_mid:]:
    line.set_color('red')
    line.set_linewidth(2.0)
    line.set_linestyle('--')
for col in ax.collections[num_cols_mid:]:
    col.set_color('red')
    col.set_linewidth(2.0)
    col.set_linestyle('--')

# ========================================================
# RESTORING TITLE & ADDING LEGEND
# ========================================================
ax.set_title("Reclamation Impact on Significant Wave Height (Hm0)\nBlue = Diffraction, Red = Reflection", fontsize=14, pad=15)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', lw=1.5, label='Natural Coastline'),
    Line2D([0], [0], color='red', lw=2.0, linestyle='--', label='Reclamation Boundary')
]
ax.legend(handles=legend_elements, loc='lower right', facecolor='white', framealpha=1)

plt.savefig("reclamation_impact_map.png", bbox_inches='tight', dpi=300)
print("   -> Map image successfully saved as 'reclamation_impact_map.png'")

print("7. Preparing tabular data for Machine Learning (AI)...")
df_ai = pd.DataFrame({
    "X_Coordinate": x_ori,
    "Y_Coordinate": y_ori,
    "Wave_Change_m": diff_val
})

df_ai_clean = df_ai.dropna()
df_ai_clean.to_csv("ai_dataset.csv", index=False)
print("   -> AI dataset successfully saved as 'ai_dataset.csv'")

print("\nPROCESS COMPLETED! Please check the Explorer panel on the left.")