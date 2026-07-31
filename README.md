# QA Explorer

QA Explorer combines a Streamlit interface with a Python browser-exploration
backend.

## Repository layout

```text
.
|-- frontend/                 # Streamlit presentation layer
|-- backend/
|   |-- app/                  # Exploration and browser logic
|   |-- agents/               # Agent implementations
|   |-- graphs/               # LangGraph state and workflows
|   |-- tests/                # Automated tests
|   |-- scripts/              # Developer utilities
|   `-- examples/             # Checked-in sample outputs and assets
|-- pyproject.toml            # Python tooling configuration
`-- README.md
```

The backend creates `generated/`, `logs/`, `runtime/`, `screenshots/`, and
`uploads/` as needed. These directories contain local runtime data and are not
version-controlled.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt -r backend\requirements-dev.txt
playwright install
```

Add the required API keys to a local `.env` file, then start the UI:

```powershell
streamlit run frontend\app.py
```

Run the test suite from the repository root:

```powershell
pytest
```
