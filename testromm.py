import sys
import pandas as pd
from motion_analysis import compute_elbow_angles

# Usage: python testromm.py <filename>
# Example: python testromm.py dataset/raw/sess5_unknown_20260528_231437_rep007.csv

if len(sys.argv) < 2:
    print("Usage: python testromm.py <path_to_csv>")
    sys.exit(1)

path = sys.argv[1]
df = pd.read_csv(path)
angles = compute_elbow_angles(df)
print(f"Min: {angles.min():.1f}, Max: {angles.max():.1f}, ROM: {angles.max()-angles.min():.1f}")
