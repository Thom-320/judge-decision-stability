# Working on this repository

From the repository root, install the package and test dependencies, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -q
bash scripts/reproduce_offline.sh
```

Reproduction uses local inputs only. The checker exits 1 if pilot figures
lack the required inputs and 2 if a calculated value fails its comparison.
Report unavailable figures or discrepancies with the command and its output.

For reproduction work, leave analysis code, data, results and README figures
untouched. Skip steps that require model APIs or credentials. Do not merge,
deploy or access the VPS.
