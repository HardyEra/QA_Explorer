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

## Browser reliability

Browser task mode and the QA executor share the same control operations:

- Fields are read back after filling; a controlled input that changes or rejects
  the value fails the action. Telephone comparisons ignore formatting, but do
  not ignore extra country-code digits. One repair is attempted on the same field.
- Duplicate labels require a unique id. The runner does not choose the first
  matching contact field or attach a document to every upload input.
- Checkbox steps set a desired state and support hidden inputs with visible labels.
- Dropdown selections and file attachments are checked after the action.
- Browser task mode observes current field values, errors, checked states, and
  stored upload assets. It must verify an outcome after the last change before
  reporting completion. Password and OTP field values are redacted in these snapshots.
- QA assertions are retained even when absent from the exploration map. Unknown
  assertions are flagged for review, and an empty assertion list cannot pass.
- Healing may repair an observed target, but cannot change test data or assertions.

Add checkpoints between wizard pages and before submitting. For example:

```json
{
  "type": "assert",
  "expectation": {
    "type": "field_value",
    "target": "financialContacts-0-email",
    "value": "demo@example.com"
  }
}
```

The additional `check` step accepts a unique checkbox id as `target` and
`"true"` or `"false"` as `value`. Final expectations and `assert` checkpoints
support `field_value` and `checked` with a `target`, alongside `text_visible`,
`element_visible`, and `url_contains`. Upload values name assets saved in the
Test assets panel. Use distinct assets for MSA, FEIN, and incorporation documents.

For creation workflows, include the required prerequisite setup and verify the
uniquely named record in its resulting list or detail page. A form reset does
not establish that a record was saved. Supply valid authorized test identities
and documents when backend validation requires them.

## QA model configuration

When `AZURE_OPENAI_API_KEY` is configured, the document analyst, test designer,
coverage critic, and healer default to Azure, like the browser task planner.
Set these in `backend/.env` (deployment names must exist in your Azure resource):

```dotenv
QA_MODEL_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_QA_DEPLOYMENT=gpt-5.6-sol
AZURE_OPENAI_TASK_DEPLOYMENT=gpt-5.6-sol
```

Set `QA_MODEL_PROVIDER=groq` to retain Groq for the QA agents. The exploration
planner and vision analysis retain their existing providers. Model choice and
these checks improve reliability, but do not guarantee completion on every site.

`backend/tests/test_browser_reliability.py` runs local Chromium fixtures covering
wizard completion, delayed rendering, duplicate labels, phone repair, checkbox
copying, uploads, and completion evidence. It makes no external model calls or
live-site submissions. Install Chromium with `playwright install chromium` first.
