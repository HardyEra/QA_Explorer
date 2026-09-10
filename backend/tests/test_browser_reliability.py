"""Local Chromium regressions: no model requests, accounts, or external writes."""

from types import SimpleNamespace

import pytest
from browser_controls import BrowserControls, ControlError, check_expectation, page_evidence
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="module")
def chromium():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(chromium):
    page = chromium.new_page()
    yield page
    page.close()


def test_duplicate_contact_labels_require_unique_ids(page):
    page.set_content(
        '<input id="company-email" placeholder="Email"><input id="finance-email" placeholder="Email">'
    )
    controls = BrowserControls(page, 700)
    with pytest.raises(ControlError, match="Ambiguous"):
        controls.fill("Email", "alex@example.com")
    assert page.locator("input").evaluate_all("els => els.map(e=>e.value)") == ["", ""]
    controls.fill("finance-email", "alex@example.com")
    assert page.locator("#company-email").input_value() == ""
    assert page.locator("#finance-email").input_value() == "alex@example.com"


def test_fill_detects_controlled_input_reverting_value(page):
    page.set_content("""<input id="name" oninput="setTimeout(()=>this.value='rejected', 40)">""")
    with pytest.raises(ControlError, match="Value mismatch"):
        BrowserControls(page, 500).fill("name", "Alex")


def test_phone_prefix_repair_and_readback(page):
    page.set_content("""<div><input id="phone" type="tel" value="+1">
        <span title="United States: + 1"></span></div>
        <script>let first=true; document.querySelector('input').oninput=function(){
          if(first){first=false;this.value='+1'+this.value.replace(/\\D/g,'');}
        };</script>""")
    BrowserControls(page, 800).fill("phone", "+12125550123")
    assert page.locator("#phone").input_value() == "+12125550123"


def test_hidden_checkbox_label_autofills_without_toggling_twice(page):
    page.set_content("""<input id="source" value="Alex"><input id="copy" style="display:none" type="checkbox"
        onchange="document.querySelector('#dest').value=this.checked?document.querySelector('#source').value:''">
        <label for="copy">Same as Company Contact Person</label><input id="dest">""")
    controls = BrowserControls(page)
    controls.check("copy")
    controls.check("copy")
    assert check_expectation(page, {"type": "checked", "target": "copy", "value": "true"})
    assert check_expectation(page, {"type": "field_value", "target": "dest", "value": "Alex"})


def test_targeted_upload_leaves_other_documents_untouched(page, tmp_path):
    asset = tmp_path / "msa.pdf"
    asset.write_bytes(b"demo")
    page.set_content('<input type="file" id="fein" hidden><input type="file" id="msa" hidden>')
    controls = BrowserControls(page)
    with pytest.raises(ControlError, match="specify"):
        controls.upload("", asset)
    controls.upload("msa", asset)
    assert page.locator("#msa").evaluate("e=>e.files[0].name") == "msa.pdf"
    assert page.locator("#fein").evaluate("e=>e.files.length") == 0


def test_selects_native_and_portal_dropdown(page):
    page.set_content("""<label for="country">Country</label><select id="country"><option>India</option><option>United States</option></select>
        <button id="terms" aria-label="Payment terms" onclick="document.querySelector('#menu').hidden=false">Select terms</button>
        <div id="menu" hidden><div role="option" onclick="document.querySelector('#terms').textContent='NET15';this.parentElement.hidden=true">NET15</div></div>""")
    controls = BrowserControls(page)
    controls.select("Country", "United States")
    controls.select("Payment terms", "NET15")
    assert page.locator("#country").input_value() == "United States"
    assert page.locator("#terms").inner_text() == "NET15"


def test_waits_for_async_destination_control_and_does_not_repeat_submit(page):
    page.set_content(
        """<button onclick="this.remove();setTimeout(()=>document.body.insertAdjacentHTML('beforeend','<button id=next>Continue</button>'),350)">Open</button>"""
    )
    controls = BrowserControls(page, 1200)
    controls.click("Open")
    controls.click("Continue")
    page.set_content("""<button onclick="window.submits=(window.submits||0)+1">Submit</button>""")
    controls.click("Submit")
    assert page.evaluate("window.submits") == 1


def test_exact_click_fallback_handles_icon_accessible_name(page):
    page.set_content('<button onclick="window.clicked=true"><img alt="arrow-right">Back</button>')
    BrowserControls(page).click("Back")
    assert page.evaluate("window.clicked") is True


def test_iframe_fields_and_hidden_text_assertions(page):
    page.set_content(
        '<div hidden>Client created successfully</div><iframe srcdoc="<input id=company>"></iframe>'
    )
    BrowserControls(page).fill("company", "Demo client")
    assert check_expectation(
        page, {"type": "field_value", "target": "company", "value": "Demo client"}
    )
    assert not check_expectation(
        page, {"type": "text_visible", "value": "Client created successfully"}
    )


def test_snapshot_redacts_passwords_and_exposes_validation(page):
    page.set_content(
        '<input id="password" type="password" value="sensitive"><input id="otp" value="123456"><input id="tax" required><input type="hidden" value="token">'
    )
    evidence = page_evidence(page)
    assert "sensitive" not in str(evidence) and "123456" not in str(evidence)
    tax = next(f for f in evidence[0]["fields"] if f["id"] == "tax")
    assert tax["valid"] is False and tax["validation"]


def test_completion_requires_fresh_business_evidence(page):
    from task_agent import TaskAgent

    agent = TaskAgent.__new__(TaskAgent)
    agent.browser = SimpleNamespace(page=page)
    page.set_content("<p>QA Demo Client 123</p>")
    verified = {
        "action_type": "verify",
        "success": True,
        "verification": {"type": "text_visible", "value": "QA Demo Client 123"},
    }
    assert not agent._completion_verified({"history": []})
    assert agent._completion_verified({"history": [verified]})
    assert not agent._completion_verified(
        {"history": [verified, {"action_type": "verify", "success": False}]}
    )
    assert not agent._completion_verified(
        {"history": [verified, {"action_type": "click", "success": True}]}
    )
    page.set_content("<p>Loading...</p>")
    assert not agent._completion_verified({"history": [verified]})


def test_empty_assertions_cannot_pass():
    from agents.test_executor.executor_agent import ExecutorAgent

    executor = ExecutorAgent.__new__(ExecutorAgent)
    result = {}
    executor._validate(None, {"expected": []}, result)
    assert result["status"] == "failed"


def test_healer_cannot_change_data_or_use_unobserved_target():
    from agents.healer.healer import Healer

    client = SimpleNamespace(
        complete_json=lambda *args: {
            "decision": "retry",
            "revised_step": {
                "type": "fill",
                "target": "Company email",
                "value": "different@example.com",
            },
        }
    )
    healer = Healer(model_client=client)
    step = {"type": "fill", "target": "Email", "value": "original@example.com"}
    result = healer.heal("Contacts", step, "Missing target", {"fields": ["Company email"]})
    assert result["revised_step"]["value"] == step["value"]
    assert healer.heal("Contacts", step, "Missing target", {"fields": []})["decision"] == "bug"


def test_checkpoint_contract_survives_normalisation():
    from agents.test_designer.models import normalise_case

    check = {"type": "field_value", "target": "finance-email", "value": "demo@example.com"}
    result = normalise_case(
        {
            "title": "Verify contacts",
            "steps": [
                {"type": "check", "target": "copy", "value": "true"},
                {"type": "assert", "expectation": check},
            ],
            "expected": [check],
        },
        "contact",
    )
    assert result["steps"][1]["expectation"] == check
    assert result["expected"] == [check]


def test_qa_executor_replays_wizard_and_verifies_created_record(page, tmp_path, monkeypatch):
    from agents.test_executor.executor_agent import ExecutorAgent
    from asset_manager import AssetManager
    from browser import BrowserController

    asset = tmp_path / "msa.pdf"
    asset.write_bytes(b"demo MSA")
    monkeypatch.setattr(AssetManager, "list_assets", lambda self: {"msa": asset})
    page.set_content("""<section id="company"><input id="name"><button onclick="company.hidden=true;poc.hidden=false">Continue</button></section>
      <section id="poc" hidden><input id="email"><input type="checkbox" id="same" style="display:none" onchange="finance.value=email.value">
      <label for="same">Same as company</label><input id="finance"><button onclick="poc.hidden=true;contract.hidden=false">Continue</button></section>
      <section id="contract" hidden><select id="terms"><option>NET30</option><option>NET15</option></select><input id="msa" type="file">
      <button onclick="if(terms.value==='NET15' && msa.files.length && finance.value===email.value){window.submits=(window.submits||0)+1;contract.hidden=true;document.querySelector('#records').textContent=document.querySelector('#name').value}">Submit</button></section><div id="records"></div>""")
    browser = BrowserController()
    browser.page = page
    executor = ExecutorAgent.__new__(ExecutorAgent)
    steps = [
        {"type": "fill", "target": "name", "value": "QA Demo Client 123"},
        {"type": "click", "target": "Continue"},
        {"type": "fill", "target": "email", "value": "demo@example.com"},
        {"type": "check", "target": "same", "value": "true"},
        {
            "type": "assert",
            "expectation": {
                "type": "field_value",
                "target": "finance",
                "value": "demo@example.com",
            },
        },
        {"type": "click", "target": "Continue"},
        {"type": "select", "target": "terms", "value": "NET15"},
        {
            "type": "assert",
            "expectation": {"type": "field_value", "target": "terms", "value": "NET15"},
        },
        {"type": "upload", "target": "msa", "value": "msa"},
        {"type": "click", "target": "Submit"},
    ]
    for step in steps:
        assert executor._execute_step(browser, step, "about:blank") is None, step
    result = {}
    executor._validate(
        browser, {"expected": [{"type": "text_visible", "value": "QA Demo Client 123"}]}, result
    )
    assert result["status"] == "passed"
    assert page.evaluate("window.submits") == 1


def test_failed_checkpoint_stops_before_submit(page, monkeypatch):
    import agents.test_executor.executor_agent as module
    from browser import BrowserController

    page.set_content(
        '<input id="finance" value="wrong"><button onclick="window.submitted=true">Submit</button>'
    )
    browser = BrowserController()
    browser.page = page
    executor = module.ExecutorAgent.__new__(module.ExecutorAgent)
    executor.on_event = lambda event: None
    executor._emit_live_frame = lambda *args: None
    monkeypatch.setattr(module, "VALIDATION_WINDOW_MS", 100)
    result = {"test_id": "wrong-contact", "failure_reason": None}
    executor._run_steps(
        browser,
        {
            "steps": [
                {
                    "type": "assert",
                    "expectation": {"type": "field_value", "target": "finance", "value": "correct"},
                },
                {"type": "click", "target": "Submit"},
            ]
        },
        "about:blank",
        "",
        "",
        result,
    )
    assert result["status"] == "failed"
    assert page.evaluate("window.submitted || false") is False


def test_azure_qa_provider_uses_configured_deployment(monkeypatch):
    import openai
    from agents.llm_client import JsonModelClient

    calls = []
    response = SimpleNamespace(
        output_text='{"ok":true}', usage=SimpleNamespace(input_tokens=4, output_tokens=2)
    )
    fake = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: (calls.append(kwargs), response)[1])
    )
    monkeypatch.setenv("QA_MODEL_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_QA_DEPLOYMENT", "qa-deployment")
    monkeypatch.setattr(openai, "AzureOpenAI", lambda **kwargs: fake)
    client = JsonModelClient()
    assert client.complete_json("test", "Return JSON") == {"ok": True}
    assert calls[0]["model"] == "qa-deployment"


def test_app_map_preserves_duplicate_field_identity():
    from app_map_builder import AppMapBuilder

    fields = [
        {"id": field_id, "placeholder": "Email", "type": "email"}
        for field_id in ("companyEmail", "financialEmail")
    ]
    app_map = AppMapBuilder().build(
        "https://demo.test",
        [
            {
                "action_type": "navigation",
                "url": "https://demo.test",
                "inputs": fields,
            }
        ],
    )
    assert len(app_map["pages"][0]["fields"]) == 2
    prompt = AppMapBuilder.compact_text(app_map)
    assert "companyEmail" in prompt and "financialEmail" in prompt


def test_uncertain_submit_is_not_healed_or_replayed(page):
    from agents.test_executor.executor_agent import ExecutorAgent

    executor = ExecutorAgent.__new__(ExecutorAgent)
    executor.on_event = lambda event: None
    executor._emit_live_frame = lambda *args: None
    attempts = []
    executor._execute_step = lambda *args: (attempts.append(args), "TimeoutError")[1]
    browser = SimpleNamespace(page=page, current_url=lambda: "https://demo.test")
    result = {"test_id": "submit", "failure_reason": None}
    executor._run_steps(
        browser,
        {"steps": [{"type": "click", "target": "Submit"}]},
        "https://demo.test",
        "",
        "",
        result,
    )
    assert len(attempts) == 1 and result["status"] == "failed"
