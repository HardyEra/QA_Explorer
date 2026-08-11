from agents.flow_generator import FlowGenerator


def test_generates_clean_business_flow_from_discovery_history():
    history = [
        {"type": "navigation", "success": True, "page_title": "Login", "url": "https://shop.test/login"},
        {"action_type": "fill", "target": "Username", "value": "standard_user", "success": True, "page_title": "Login"},
        {"action_type": "click", "target": "Login", "success": False, "page_title": "Login"},
        {"action_type": "click", "target": "Login", "success": True, "page_title": "Login"},
        {"type": "telemetry", "success": True, "target": "ignored"},
        {"action_type": "click", "target": "Login", "success": True, "page_title": "Login", "retry": True},
        {"type": "navigation", "success": True, "page_title": "Inventory", "url": "https://shop.test/inventory"},
        {"action_type": "click", "target": "Sauce Labs Backpack", "success": True, "page_title": "Inventory"},
        {"action_type": "click", "target": "Add to cart", "success": True, "page_title": "Product Detail"},
    ]

    workflow = FlowGenerator().generate_json(history, flow_name="Purchase Product")

    assert workflow["flow_name"] == "Purchase Product"
    assert workflow["pages"] == ["Login", "Inventory", "Product Detail"]
    assert workflow["steps"] == [
        {"type": "fill", "target": "Username", "value": "standard_user"},
        {"type": "click", "target": "Login"},
        {"type": "click", "target": "Sauce Labs Backpack"},
        {"type": "click", "target": "Add to cart"},
    ]


def test_collapses_consecutive_duplicate_clicks():
    history = [
        {"action_type": "click", "target": "Continue", "success": True},
        {"action_type": "click", "target": "Continue", "success": True},
    ]

    assert FlowGenerator().generate_json(history)["steps"] == [{"type": "click", "target": "Continue"}]
