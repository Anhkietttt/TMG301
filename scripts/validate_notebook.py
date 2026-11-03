"""Validate the notebook and its embedded Python modules without running GPU jobs."""
import ast
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / 'notebooks/DE200278_Domain_Adaptation_AR_EN_Kaggle.ipynb', as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == 'code' and not cell.source.startswith('%pip'):
        ast.parse(cell.source)
print(f'Validated {len(notebook.cells)} cells; no GPU execution claimed.')
