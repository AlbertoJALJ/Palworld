"""HTTP surface tests.

These run against the demo dataset, so they assert on shape, status codes and
the data-quality contract rather than on particular pals.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from palworld_api import api as api_module
from palworld_api.api import app, get_services
from palworld_api.ingest.demo import build_demo_dataset


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "demo.json"
    payload = build_demo_dataset().model_dump(mode="json", exclude_none=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def client(dataset_path: Path) -> TestClient:
    services = api_module.Services(dataset_path)
    app.dependency_overrides[get_services] = lambda: services
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_reports_dataset_contents(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pals"] > 0
    assert body["passives"] > 0


def test_demo_data_is_flagged_as_unverified(client: TestClient) -> None:
    quality = client.get("/health").json()["data_quality"]
    assert quality["verified"] is False
    assert "demo" in quality["warning"].lower()


def test_every_data_endpoint_carries_provenance(client: TestClient) -> None:
    endpoints = [
        "/pals/demo_forge",
        "/breed?parent_a=demo_wooly&parent_b=demo_sprout",
        "/routes/demo_sovereign",
        "/pals/demo_forge/passives?goal=combat",
        "/pals/demo_forge/parents",
        "/reachable",
    ]
    for endpoint in endpoints:
        body = client.get(endpoint).json()
        assert "data_quality" in body, endpoint
        assert body["data_quality"]["source"] == "demo", endpoint


def test_list_pals_paginates(client: TestClient) -> None:
    first = client.get("/pals?limit=5").json()
    assert len(first) == 5
    second = client.get("/pals?limit=5&offset=5").json()
    assert {p["id"] for p in first}.isdisjoint({p["id"] for p in second})


def test_list_pals_filters_by_element_and_work(client: TestClient) -> None:
    fire = client.get("/pals?element=fire").json()
    assert fire and all("fire" in p["elements"] for p in fire)
    miners = client.get("/pals?work=mining").json()
    assert miners


def test_list_pals_search_is_case_insensitive(client: TestClient) -> None:
    assert client.get("/pals?search=FORGE").json()[0]["id"] == "demo_forge"


def test_unknown_pal_is_404(client: TestClient) -> None:
    assert client.get("/pals/nope").status_code == 404
    assert client.get("/breed?parent_a=nope&parent_b=demo_wooly").status_code == 404
    assert client.get("/routes/nope").status_code == 404
    assert client.get("/pals/nope/passives").status_code == 404


def test_unknown_goal_is_422(client: TestClient) -> None:
    response = client.get("/pals/demo_forge/passives?goal=sandwich")
    assert response.status_code == 422


def test_breed_returns_the_rule_used(client: TestClient) -> None:
    body = client.get("/breed?parent_a=demo_wyrm&parent_b=demo_phoenix").json()
    assert body["child"] == "demo_sovereign"
    assert body["rule"] == "special"


def test_breed_accepts_display_names(client: TestClient) -> None:
    body = client.get("/breed?parent_a=Demo%20Wyrm&parent_b=Demo%20Phoenix").json()
    assert body["child"] == "demo_sovereign"


def test_route_steps_are_ordered_and_valid(client: TestClient) -> None:
    body = client.get("/routes/demo_sovereign").json()
    assert body["generations"] >= 1
    assert len(body["steps"]) == body["distinct_steps"]

    have = set()
    for step in body["steps"]:
        # Anything not produced by an earlier step must be something catchable.
        for parent in (step["parent_a"], step["parent_b"]):
            if parent not in have:
                pal = client.get(f"/pals/{parent}").json()["pal"]
                assert pal["wild_obtainable"] is True
        have.add(step["child"])
    assert body["target"] in have


def test_route_respects_an_explicit_owned_set(client: TestClient) -> None:
    response = client.get("/routes/demo_sovereign?owned=demo_wooly&owned=demo_sprout")
    # Both parents sit at the top of the rank range, and the formula only ever
    # interpolates, so nothing rarer can be reached from them.
    assert response.status_code == 422
    assert "not reachable" in response.json()["detail"]


def test_route_generation_cap_is_enforced(client: TestClient) -> None:
    assert client.get("/routes/demo_sovereign?max_generations=1").status_code == 422


def test_owning_the_target_short_circuits(client: TestClient) -> None:
    body = client.get("/routes/demo_sovereign?owned=demo_sovereign").json()
    assert body["already_owned"] is True
    assert body["steps"] == []


def test_parents_and_children_are_consistent(client: TestClient) -> None:
    pairs = client.get("/pals/demo_sovereign/parents").json()["pairs"]
    assert ["demo_phoenix", "demo_wyrm"] in [sorted(p) for p in pairs]
    children = client.get("/pals/demo_wyrm/children").json()["children"]
    assert children["demo_phoenix"] == "demo_sovereign"


def test_goals_endpoint_lists_the_weights(client: TestClient) -> None:
    body = client.get("/goals").json()
    assert set(body) == {"base", "combat"}
    assert body["base"]["weights"]["work_speed"] > 0
    assert body["combat"]["weights"]["attack"] > 0


def test_base_and_combat_give_different_loadouts(client: TestClient) -> None:
    base = client.get("/pals/demo_forge/passives?goal=base").json()
    combat = client.get("/pals/demo_forge/passives?goal=combat").json()
    assert {p["id"] for p in base["passives"]} != {p["id"] for p in combat["passives"]}
    assert len(base["passives"]) <= 4 and len(combat["passives"]) <= 4


def test_plan_with_passives_costs_out_the_route(client: TestClient) -> None:
    response = client.post(
        "/routes/plan",
        json={
            "target": "demo_sovereign",
            "desired_passives": ["demo_legend", "demo_ferocious"],
            "starting_passives": {
                "demo_titan": ["demo_legend"],
                "demo_shade": ["demo_ferocious"],
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["steps"]
    assert 0.0 < body["clean_run_probability"] <= 1.0
    assert body["total_expected_attempts"] >= len(body["steps"])
    for step in body["steps"]:
        assert 0.0 <= step["probability"] <= 1.0


def test_plan_with_unknown_passive_is_404(client: TestClient) -> None:
    response = client.post(
        "/routes/plan", json={"target": "demo_sovereign", "desired_passives": ["nope"]}
    )
    assert response.status_code == 404


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/routes/{target}" in schema["paths"]
    assert "/breed" in schema["paths"]


def test_ui_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Palworld" in response.text


def test_default_dataset_is_found_from_any_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting the server outside the repo root must still work.

    The lookup walks up from the package, so an editable install finds its own
    `data/` regardless of where the process was launched. Getting this wrong
    fails on every request rather than at startup, which is far harder to read.
    """
    from palworld_api.dataset import find_default_dataset

    monkeypatch.chdir(tmp_path)
    found = find_default_dataset()
    assert found.exists(), f"{found} should have been found from {tmp_path}"
    assert found.name == "pals.json"


def test_a_local_data_directory_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from palworld_api.dataset import find_default_dataset

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "pals.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert find_default_dataset() == Path("data/pals.json")


def test_work_overview_lists_best_pal_per_activity(client: TestClient) -> None:
    body = client.get("/work").json()
    assert "mining" in body["by_work"]
    assert body["by_work"]["mining"][0]["pal"] == "demo_titan"
    assert body["by_work"]["mining"][0]["level"] == 4
    assert body["data_quality"]["source"] == "demo"


def test_work_overview_omits_activities_nobody_does(client: TestClient) -> None:
    body = client.get("/work").json()
    # No demo pal has oil extraction.
    assert "oil_extraction" not in body["by_work"]


def test_work_overview_limit_applies_per_activity(client: TestClient) -> None:
    body = client.get("/work?limit=1").json()
    assert all(len(pals) <= 1 for pals in body["by_work"].values())


def test_best_pals_for_one_activity_are_ordered(client: TestClient) -> None:
    body = client.get("/work/mining/best").json()
    assert body["work"] == "mining"
    levels = [p["level"] for p in body["pals"]]
    assert levels == sorted(levels, reverse=True)
    assert body["pals"][0]["pal"] == "demo_titan"


def test_unknown_activity_is_422(client: TestClient) -> None:
    assert client.get("/work/sandwich/best").status_code == 422


def test_work_endpoints_carry_provenance(client: TestClient) -> None:
    for endpoint in ("/work", "/work/mining/best"):
        assert client.get(endpoint).json()["data_quality"]["source"] == "demo"
