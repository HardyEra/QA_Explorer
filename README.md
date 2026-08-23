# QA Explorer

QA Explorer combines a Streamlit interface with a Python browser-exploration
backend.

## PRD-driven workflow

Each QA run requires a detailed English product requirements document in
`.txt`, `.pdf`, or `.docx` form. Provide the target URL and, when needed, the
same login credentials in the UI. The pipeline extracts requirements, explores
the live product, saves an immutable screenshot for every observed page,
records successful action events and workflows, then designs and verifies test
cases only against that collected evidence. Screenshots and workflow matches
are available in each test's detail view and are retained under
`backend/generated/evidence/<run-id>/`.

When `GEMINI_API_KEY` is configured, the screenshot observer also supplies a
strict visual summary to test design; DOM/accessibility evidence remains the
fallback when vision is unavailable.

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
