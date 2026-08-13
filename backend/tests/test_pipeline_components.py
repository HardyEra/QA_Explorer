"""Deterministic-path tests for the multi-agent pipeline components.

No browser and no model key are required: every agent must degrade to its
documented fallback, and the orchestrator graph must compile and route.
"""

import json

import pytest

from agents.coverage_critic import CoverageCritic
from agents.doc_analyst import DocAnalyst
from agents.reporter import Reporter
from agents.test_designer import TestDesigner, normalise_case
from app_map_builder import AppMapBuilder


class UnavailableModelClient:
    """Stand-in for JsonModelClient when no provider is configured."""

    available = False

    def complete_json(self, name, prompt):
        return None


@pytest.fixture
def offline():
    return UnavailableModelClient()


def test_app_map_builder_folds_history_into_pages():
    history = [
        {"action_type": "navigation", "success": True, "page_title": "Login",
         "url": "https://shop.test/login?next=/"},
        {"action_type": "fill", "target": "Username", "value": "u", "success": True,
         "page_title": "Login", "url": "https://shop.test/login"},
        {"action_type": "click", "target": "Login", "success": True,
         "page_title": "Login", "url": "https://shop.test/login"},
        {"action_type": "navigation", "success": True, "page_title": "Inventory",
         "url": "https://shop.test/inventory"},
    ]

    app_map = AppMapBuilder().build("https://shop.test", history)

    urls = [page["url"] for page in app_map["pages"]]
    assert urls == ["https://shop.test/login", "https://shop.test/inventory"]
    login_page = app_map["pages"][0]
    assert {"label": "Login", "succeeded": True} in login_page["actions"]
    assert {"field": "Username"} in login_page["fills"]
    assert {"from": "https://shop.test/login", "to": "https://shop.test/inventory"} in app_map["transitions"]
    assert "Clickable: Login" in AppMapBuilder.compact_text(app_map)


def test_normalise_case_rejects_invalid_and_keeps_valid_steps():
    assert normalise_case({"title": "no steps", "expected": []}, "x") is None
    case = normalise_case(
        {
            "id": "Auth Login!",
            "title": "Valid login",
            "priority": "CRITICAL",
            "steps": [
                {"type": "navigate", "target": "/login"},
                {"type": "teleport", "target": "nowhere"},
                {"type": "fill", "target": "Username", "value": "{username}"},
                {"type": "click", "target": ""},
                {"type": "click", "target": "Login"},
            ],
            "expected": [
                {"type": "url_contains", "value": "inventory"},
                {"type": "mind_reading", "value": "nope"},
            ],
        },
        "fallback-1",
    )
    assert case["id"] == "auth-login"
    assert case["priority"] == "critical"
    assert [step["type"] for step in case["steps"]] == ["navigate", "fill", "click"]
    assert case["expected"] == [{"type": "url_contains", "value": "inventory"}]


def test_designer_fallback_yields_one_smoke_case_per_requirement(offline):
    designer = TestDesigner(model_client=offline)
    requirements = [
        {"id": "prd-R1", "feature": "auth", "title": "User can log in", "priority": "critical"},
        {"id": "prd-R2", "feature": "auth", "title": "User can log out"},
    ]
    cases, blocked = designer.design("auth", requirements, {"pages": []}, "https://shop.test")
    assert blocked == []
    assert [case["requirement_id"] for case in cases] == ["prd-R1", "prd-R2"]
    assert all(case["steps"][0]["type"] == "navigate" for case in cases)
    assert all(case["expected"] for case in cases)
    # Every case must be understandable without reading its JSON.
    assert all("This test will open" in case["description"] for case in cases)


def test_fallback_smokes_assert_observed_controls_not_bare_urls(offline):
    """Functionality over URLs: a smoke test must prove the screen rendered."""
    designer = TestDesigner(model_client=offline)
    app_map = {"pages": [{
        "url": "https://vms.test/candidates", "title": "Candidates",
        "actions": [], "fills": [], "fields": [], "roles": {},
        "controls": ["Add Candidate", "Filters", "View details"],
    }]}
    requirements = [{"id": "app-R1", "feature": "candidates", "title": "Candidates screen opens",
                     "source_url": "https://vms.test/candidates"}]

    cases, _ = designer.design("candidates", requirements, app_map, "https://vms.test")

    expected = cases[0]["expected"]
    assert all(item["type"] == "element_visible" for item in expected)
    assert [item["value"] for item in expected] == ["Add Candidate", "Filters", "View details"]


def test_grounding_drops_meaningless_bare_domain_checks():
    class Canned:
        available = True

        def complete_json(self, name, prompt):
            return {"test_cases": [{
                "id": "c1", "requirement_id": "r1", "title": "Open candidates",
                "steps": [{"type": "click", "target": "Candidates"}],
                "expected": [
                    {"type": "element_visible", "value": "Add Candidate"},
                    {"type": "url_contains", "value": "vms.test"},      # meaningless
                    {"type": "url_contains", "value": "candidates"},    # specific, observed
                ],
            }], "blocked": []}

    app_map = {"pages": [{"url": "https://vms.test/candidates", "title": "Candidates",
                          "actions": [], "fills": [], "fields": [], "roles": {},
                          "controls": ["Candidates", "Add Candidate"]}]}
    cases, _ = TestDesigner(model_client=Canned()).design(
        "candidates", [{"id": "r1", "title": "Candidates"}], app_map, "https://vms.test")

    values = [item["value"] for item in cases[0]["expected"]]
    assert "Add Candidate" in values and "candidates" in values
    assert "vms.test" not in values


def test_designer_grounds_expectations_and_keeps_valid_blocked_entries():
    class CannedModelClient:
        available = True

        def complete_json(self, name, prompt):
            return {
                "test_cases": [
                    {
                        "id": "auth-login",
                        "requirement_id": "prd-R1",
                        "title": "Valid login",
                        "steps": [{"type": "navigate", "target": "/login"}],
                        "expected": [
                            {"type": "text_visible", "value": "Sign In"},
                            {"type": "text_visible", "value": "Notification sent"},
                            {"type": "url_contains", "value": "shop.test/login"},
                            {"type": "url_contains", "value": "slide-1"},
                        ],
                    }
                ],
                "blocked": [
                    {"requirement_id": "prd-R2", "reason": "requires the Admin role"},
                    {"requirement_id": "prd-R99", "reason": "not a real requirement"},
                ],
            }

    designer = TestDesigner(model_client=CannedModelClient())
    requirements = [
        {"id": "prd-R1", "feature": "auth", "title": "User can log in"},
        {"id": "prd-R2", "feature": "auth", "title": "Admin dashboard shows metrics"},
    ]
    app_map = {"pages": [{"title": "Login", "url": "https://shop.test/login",
                          "actions": [], "fills": [], "controls": ["Sign In"], "fields": []}]}

    cases, blocked = designer.design("auth", requirements, app_map, "https://shop.test")

    # "Sign In" is in the app map; "Notification sent" was invented and dropped;
    # the observed URL survives while the invented "slide-1" URL is dropped.
    assert [e["value"] for e in cases[0]["expected"]] == ["Sign In", "shop.test/login"]
    assert [entry["requirement_id"] for entry in blocked] == ["prd-R2"]


def test_critic_treats_blocked_requirements_as_not_gaps(offline):
    critic = CoverageCritic(model_client=offline)
    requirements = [
        {"id": "prd-R1", "feature": "auth", "title": "Login works"},
        {"id": "prd-R2", "feature": "admin", "title": "Admin metrics"},
    ]
    cases = [{"id": "t1", "requirement_id": "prd-R1", "title": "login test"}]
    blocked = [{"requirement_id": "prd-R2", "reason": "requires Admin role"}]

    verdict = critic.review(requirements, cases, blocked)
    assert verdict["approved"] is True


def test_executor_login_boundary_detection():
    from agents.test_executor.executor_agent import ExecutorAgent

    steps = [
        {"type": "navigate", "target": "/login"},
        {"type": "fill", "target": "email", "value": "{username}"},
        {"type": "fill", "target": "password", "value": "{password}"},
        {"type": "click", "target": "Sign In"},
        {"type": "click", "target": "Claim"},
    ]
    assert ExecutorAgent._login_boundary(steps) == 3
    # Negative-credential tests use literal values, so they are not serialized.
    negative = [
        {"type": "fill", "target": "password", "value": "wrong-password"},
        {"type": "click", "target": "Sign In"},
    ]
    assert ExecutorAgent._login_boundary(negative) is None
    assert ExecutorAgent._login_boundary([]) is None


def test_session_reuse_skips_only_real_credential_logins():
    from agents.test_executor.executor_agent import ExecutorAgent

    login_steps = [
        {"type": "navigate", "target": "/login"},
        {"type": "fill", "target": "email", "value": "{username}"},
        {"type": "fill", "target": "password", "value": "{password}"},
        {"type": "click", "target": "Log In"},
        {"type": "click", "target": "Jobs"},
    ]
    # Real-credential test: boundary found → login prefix is skippable.
    assert ExecutorAgent._login_boundary(login_steps) == 3
    assert login_steps[ExecutorAgent._login_boundary(login_steps) + 1:] == [
        {"type": "click", "target": "Jobs"}
    ]
    # Negative-login test: literal wrong password → never uses the session.
    negative = [
        {"type": "fill", "target": "password", "value": "wrong"},
        {"type": "click", "target": "Log In"},
    ]
    assert ExecutorAgent._login_boundary(negative) is None
    # Login-free test: nothing to skip, runs fresh as before.
    assert ExecutorAgent._login_boundary([{"type": "navigate", "target": "/"}]) is None

    # Session eligibility requires BOTH placeholders: an invalid-username test
    # (literal username + real password) must run fresh and logged out.
    from agents.test_designer.models import USERNAME_PLACEHOLDER

    invalid_username = [
        {"type": "fill", "target": "Username", "value": "invalid-username"},
        {"type": "fill", "target": "Password", "value": "{password}"},
        {"type": "click", "target": "Sign In"},
    ]
    boundary = ExecutorAgent._login_boundary(invalid_username)
    uses_real_credentials = boundary is not None and any(
        USERNAME_PLACEHOLDER in str(step.get("value", "")) for step in invalid_username
    )
    assert boundary is not None and not uses_real_credentials


def test_verifier_corrects_near_misses_and_flags_inventions():
    from agents.test_verifier import TestVerifier

    app_map = {"pages": [{
        "url": "https://vms.test/login", "title": "Login", "actions": [], "fills": [],
        "controls": ["Sign in", "toggle password", "Onboarded Vendors 35"],
        "fields": [{"type": "email", "name": "email", "placeholder": "Enter your email"}],
        "roles": {},
    }]}
    cases = [{
        "id": "auth", "title": "Login", "steps": [
            {"type": "navigate", "target": "https://vms.test/login"},
            {"type": "fill", "target": "email", "value": "{username}"},
            {"type": "click", "target": "Sign In"},                     # case-only diff → exact
            {"type": "click", "target": "View Onboarded Vendors details"},  # substring → corrected
            {"type": "click", "target": "Teleport to Mars"},            # invented → flagged
        ],
        "expected": [
            {"type": "url_contains", "value": "dashboard"},
            {"type": "element_visible", "value": "Nonexistent Widget"},  # invented → dropped
        ],
    }]

    verified, problems = TestVerifier().verify(cases, app_map)
    steps = verified[0]["steps"]
    assert steps[2]["target"] == "Sign in"
    assert steps[3]["target"] == "Onboarded Vendors 35"
    assert steps[4]["target"] == "Teleport to Mars"          # kept, but flagged
    assert [e["value"] for e in verified[0]["expected"]] == ["dashboard"]
    assert verified[0]["unverified"]
    assert {p["target"] for p in problems} == {"Teleport to Mars", "Nonexistent Widget"}

    # A report must call this a test-design problem, not a product defect.
    from agents.reporter import Reporter

    explanation = Reporter._explain_result({
        "status": "failed", "unverified": verified[0]["unverified"],
        "failed_step": {"index": 5, "type": "click", "target": "Teleport to Mars"},
    })
    assert "test-design problem" in explanation


def test_guidance_channel_send_drain_ask():
    import threading
    import time as _time

    from guidance import GuidanceChannel

    channel = GuidanceChannel()
    # Proactive guidance: tester sends, agent drains before planning.
    channel.send("Client Name is a dropdown — select, don't type")
    channel.send("")  # blank messages are ignored
    assert channel.drain() == ["Client Name is a dropdown — select, don't type"]
    assert channel.drain() == []
    assert channel.transcript == [("tester", "Client Name is a dropdown — select, don't type")]

    # Ask-and-wait: a late reply satisfies the blocked ask.
    def reply_soon():
        _time.sleep(0.2)
        channel.send("Click the Purchase Order tab first")

    threading.Thread(target=reply_soon, daemon=True).start()
    answer = channel.ask("I'm stuck — what should I do?", timeout_s=3)
    assert answer == "Click the Purchase Order tab first"
    assert channel.pending_question is None
    assert channel.transcript[1] == ("agent", "I'm stuck — what should I do?")

    # Timeout: no reply → None, agent continues autonomously.
    assert channel.ask("still stuck?", timeout_s=1) is None


def test_total_ai_blackout_skips_test_execution():
    from agents import llm_client
    from graphs.orchestrator_graph import QAOrchestrator

    orchestrator = QAOrchestrator()
    cases = {"test_cases": [{"id": "t1", "title": "smoke", "steps": [], "expected": []}]}

    llm_client.reset_model_health()
    llm_client._HEALTH.update(calls=4, failures=4)
    try:
        assert orchestrator._collect_tests(cases)["test_plan"] == []
        # Partial availability still executes.
        llm_client._HEALTH.update(calls=4, failures=2)
        assert len(orchestrator._collect_tests(cases)["test_plan"]) == 1
    finally:
        llm_client.reset_model_health()


def test_result_explanations_cover_each_failure_shape():
    from agents.reporter import Reporter

    passed = {
        "status": "passed",
        "expectations": [{"type": "url_contains", "value": "inventory", "passed": True}],
    }
    step_failure = {
        "status": "failed",
        "failed_step": {"index": 3, "type": "click", "target": "Shopping Cart"},
        "failure_reason": "no clickable control matched 'Shopping Cart'",
    }
    check_failure = {
        "status": "failed",
        "expectations": [{"type": "text_visible", "value": "Offline", "passed": False}],
        "evidence": {"final_url": "https://vuc.test/vuc"},
    }
    error = {"status": "error", "failure_reason": "Page.goto: Timeout 30000ms exceeded."}

    assert "every check held" in Reporter._explain_result(passed)
    assert "Stopped at step 3" in Reporter._explain_result(step_failure)
    explanation = Reporter._explain_result(check_failure)
    assert "the flow itself worked" in explanation and "https://vuc.test/vuc" in explanation
    assert "could not run" in Reporter._explain_result(error)

    # Credential placeholders are described, never expanded.
    from agents.test_designer.models import describe_step

    assert describe_step({"type": "fill", "target": "password", "value": "{password}"}) \
        == "enter the configured password into 'password'"


def test_app_map_store_merges_across_runs(tmp_path):
    from app_map_store import AppMapStore

    store = AppMapStore(root=tmp_path)
    first = {
        "start_url": "https://vuc.test/login",
        "pages": [{"url": "https://vuc.test/login", "title": "Login",
                   "actions": [], "fills": [], "controls": ["Sign In"], "fields": []}],
        "transitions": [],
    }
    second = {
        "start_url": "https://vuc.test/login",
        "pages": [
            {"url": "https://vuc.test/login", "title": "Login",
             "actions": [], "fills": [], "controls": ["Sign In", "Forgot Password"], "fields": []},
            {"url": "https://vuc.test/vuc", "title": "Dashboard",
             "actions": [], "fills": [], "controls": ["Claim"], "fields": []},
        ],
        "transitions": [{"from": "https://vuc.test/login", "to": "https://vuc.test/vuc"}],
    }

    store.merge_and_save("https://vuc.test/login", first)
    merged = store.merge_and_save("https://vuc.test/login", second)

    assert merged["run_count"] == 2
    assert len(merged["pages"]) == 2
    login = next(p for p in merged["pages"] if p["url"].endswith("/login"))
    assert set(login["controls"]) == {"Sign In", "Forgot Password"}
    # A later docs-only run loads the same accumulated knowledge.
    assert len(store.load("https://vuc.test/anything")["pages"]) == 2


def test_cartographer_fallback_maps_requirements_to_pages(offline):
    from agents.cartographer import Cartographer

    cartographer = Cartographer(model_client=offline)
    requirements = [
        {"id": "r1", "feature": "provider availability", "title": "Provider availability board"},
        {"id": "r2", "feature": "admin configuration", "title": "Admin manages clinics"},
    ]
    app_map = {
        "pages": [
            {"url": "https://vuc.test/vuc/availability-board", "title": "Availability Board",
             "controls": ["Available", "Busy", "Provider"], "actions": []},
        ]
    }

    semantic_map = cartographer.map_features(requirements, app_map)

    by_id = {item["requirement_id"]: item for item in semantic_map["features"]}
    assert by_id["r1"]["status"] == "found"
    assert by_id["r1"]["pages"] == ["https://vuc.test/vuc/availability-board"]
    assert by_id["r2"]["status"] == "not_found"


def test_upload_step_is_valid_and_described(tmp_path):
    from agents.test_designer.models import describe_step
    from asset_manager import AssetManager

    case = normalise_case(
        {
            "title": "Create candidate from resume",
            "steps": [
                {"type": "navigate", "target": "/candidates"},
                {"type": "upload", "target": "Choose Files", "value": "resume"},
                {"type": "click", "target": "Save"},
            ],
            "expected": [{"type": "text_visible", "value": "Candidate created"}],
        },
        "up-1",
    )
    assert [s["type"] for s in case["steps"]] == ["navigate", "upload", "click"]
    assert case["steps"][1]["value"] == "resume"
    assert describe_step(case["steps"][1]) == "upload the stored test asset 'resume' via 'Choose Files'"

    manager = AssetManager(upload_dir=tmp_path)
    manager.save_asset("resume", "cv.pdf", b"pdf-bytes")
    manager.save_asset("invoice", "inv.pdf", b"pdf-bytes")
    assert sorted(manager.list_assets()) == ["invoice", "resume"]


def test_select_step_and_control_roles_flow_to_designer_context():
    from agents.test_designer.models import describe_step

    # A dropdown control must produce a valid select step…
    case = normalise_case(
        {
            "title": "Create PO with client",
            "steps": [
                {"type": "select", "target": "Client Name [dropdown]", "value": "Acme Corp"},
                {"type": "select", "target": "Empty option", "value": ""},
                {"type": "click", "target": "Create PO"},
            ],
            "expected": [{"type": "url_contains", "value": "purchase-order"}],
        },
        "sel-1",
    )
    assert case["steps"][0] == {"type": "select", "target": "Client Name", "value": "Acme Corp"}
    # …the empty-option select is dropped, the click survives.
    assert [s["type"] for s in case["steps"]] == ["select", "click"]
    assert describe_step(case["steps"][0]) == "choose 'Acme Corp' in the 'Client Name' dropdown"

    # Roles captured in the map are rendered as [dropdown] tags for the Designer.
    history = [{
        "action_type": "navigation", "success": True, "page_title": "PO",
        "url": "https://vms.test/po/create",
        "controls": ["Client Name", "Create PO"],
        "control_roles": {"Client Name": "combobox"},
    }]
    app_map = AppMapBuilder().build("https://vms.test", history)
    assert app_map["pages"][0]["roles"] == {"Client Name": "combobox"}
    text = AppMapBuilder.compact_text(app_map)
    assert "Client Name [dropdown]" in text and "Create PO" in text


def test_app_map_store_merges_roles(tmp_path):
    from app_map_store import AppMapStore

    store = AppMapStore(root=tmp_path)
    first = {"start_url": "https://vms.test", "pages": [
        {"url": "https://vms.test/po", "title": "PO", "actions": [], "fills": [],
         "controls": ["Client Name"], "fields": [], "roles": {"Client Name": "combobox"}}],
        "transitions": []}
    second = {"start_url": "https://vms.test", "pages": [
        {"url": "https://vms.test/po", "title": "PO", "actions": [], "fills": [],
         "controls": ["Status"], "fields": [], "roles": {"Status": "radio"}}],
        "transitions": []}
    store.merge_and_save("https://vms.test", first)
    merged = store.merge_and_save("https://vms.test", second)
    assert merged["pages"][0]["roles"] == {"Client Name": "combobox", "Status": "radio"}


def test_cartographer_synthesizes_requirements_without_docs(offline):
    from agents.cartographer import Cartographer

    app_map = {
        "pages": [
            {"url": "https://ats.test/login", "title": "Login",
             "controls": ["Log In", "Forgot Password?"], "fields": []},
            {"url": "https://ats.test/dashboard", "title": "Dashboard",
             "controls": ["Jobs", "Candidates", "Clients"], "fields": []},
        ]
    }
    requirements = Cartographer(model_client=offline).synthesize_requirements(app_map)

    assert len(requirements) == 2
    assert all(requirement["id"].startswith("app-R") for requirement in requirements)
    assert all(requirement["source_doc"] == "observed application" for requirement in requirements)
    assert "Candidates" in requirements[1]["description"]


def test_unexplored_hints_surface_unvisited_modules():
    from graphs.orchestrator_graph import QAOrchestrator

    stored = {
        "pages": [
            {"url": "https://ats.test/test/dashboard", "title": "Otomashen ATS",
             "controls": ["Dashboard", "Jobs", "Candidates", "Clients", "Talent Bench",
                          "Open navigation menu", "0 Total Open Jobs"]},
        ]
    }
    hints = QAOrchestrator._unexplored_hints(stored)

    # Dashboard is in the URL (visited); the other modules are frontier.
    assert "Dashboard" not in hints
    assert {"Candidates", "Clients", "Talent Bench"}.issubset(set(hints))
    context = QAOrchestrator._prd_seeded_context({"application_context": ""}, stored)
    assert "NEVER" in context and "Candidates" in context


def test_prd_seeded_context_lists_features():
    from graphs.orchestrator_graph import QAOrchestrator

    context = QAOrchestrator._prd_seeded_context(
        {
            "application_context": "Provider account only.",
            "requirements": [
                {"feature": "authentication", "title": "User can log in"},
                {"feature": "queue management", "title": "Requests are queued"},
            ],
        }
    )
    assert context.startswith("Provider account only.")
    assert "authentication: User can log in" in context
    assert "queue management: Requests are queued" in context


def test_critic_flags_untested_requirements_without_a_model(offline):
    critic = CoverageCritic(model_client=offline)
    requirements = [
        {"id": "prd-R1", "feature": "auth", "title": "Login works"},
        {"id": "prd-R2", "feature": "cart", "title": "Cart totals update"},
    ]
    cases = [{"id": "t1", "requirement_id": "prd-R1", "title": "login test"}]

    verdict = critic.review(requirements, cases)

    assert verdict["approved"] is False
    assert [gap["requirement_id"] for gap in verdict["gaps"]] == ["prd-R2"]

    verdict = critic.review(requirements, cases + [{"id": "t2", "requirement_id": "prd-R2", "title": "cart"}])
    assert verdict["approved"] is True


def test_doc_analyst_heading_fallback(tmp_path, offline):
    doc = tmp_path / "prd.md"
    doc.write_text(
        "# Authentication\nUsers must log in with email and password.\n\n"
        "## Checkout\nUsers can pay by card.\n",
        encoding="utf-8",
    )
    requirements = DocAnalyst(model_client=offline).analyze(str(doc))
    assert [requirement["title"] for requirement in requirements] == ["Authentication", "Checkout"]
    assert all(requirement["source_doc"] == "prd.md" for requirement in requirements)
    assert requirements[0]["id"].startswith("prd-R")


def test_reporter_writes_all_three_artifacts(tmp_path):
    reporter = Reporter(output_root=tmp_path)
    requirements = [{"id": "prd-R1", "feature": "auth", "title": "Login works", "priority": "critical"}]
    plan = [{"id": "t1", "requirement_id": "prd-R1", "title": "Valid login", "priority": "critical"}]
    results = [
        {"test_id": "t1", "requirement_id": "prd-R1", "title": "Valid login",
         "priority": "critical", "status": "passed", "duration_ms": 1200.0},
        {"test_id": "t2", "requirement_id": "", "title": "Broken flow", "priority": "medium",
         "status": "failed", "failure_reason": "expected text_visible 'Cart'",
         "duration_ms": 900.0, "evidence": {"screenshot": "shot.png"}},
    ]

    report = reporter.generate("testrun", requirements, plan, results)

    assert report["summary"]["total_tests"] == 2
    assert report["summary"]["passed"] == 1
    assert report["summary"]["requirements_tested"] == 1
    assert report["coverage"][0]["status"] == "passed"
    assert len(report["bugs"]) == 1
    # Every result carries a deterministic plain-language explanation.
    assert all(result["explanation"] for result in report["results"])

    output_dir = tmp_path / "testrun"
    saved = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == "testrun"
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Valid login" in markdown and "suspected bugs" in markdown
    junit = (output_dir / "junit.xml").read_text(encoding="utf-8")
    assert 'tests="2"' in junit and 'failures="1"' in junit


def test_orchestrator_graph_compiles_and_routes():
    from graphs.orchestrator_graph import QAOrchestrator
    from langgraph.types import Send

    orchestrator = QAOrchestrator()

    route = orchestrator._ingest_route({"doc_paths": []})
    assert route == "explore"
    sends = orchestrator._ingest_route({"doc_paths": ["a.md", "b.md"]})
    assert all(isinstance(send, Send) for send in sends) and len(sends) == 2

    collected = orchestrator._collect_tests(
        {"test_cases": [{"id": "t1", "title": "a"}, {"id": "t1", "title": "b"}]}
    )
    assert [case["id"] for case in collected["test_plan"]] == ["t1", "t1-2"]

    assert orchestrator._execute_route({"test_plan": []}) == "report"
    assert orchestrator._critic_route(
        {"critique": {"approved": True, "gaps": []}, "design_rounds": 1}
    ) == "collect_tests"
