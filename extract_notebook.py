import json
import sys

notebook_path = r'd:\turbofan_RUL\Turbofan-Engine-Remaining_Useful_Life\04_predict_rul_with_ml.ipynb'

with open(notebook_path) as f:
    nb = json.load(f)

# Extract code cells
for i, cell in enumerate(nb['cells'][:30]):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'feature' in source.lower() or 'pca' in source.lower() or 'extract' in source.lower():
            print(f"\n{'='*80}")
            print(f"Code Cell {i}:")
            print(f"{'='*80}")
            print(source[:1000])

