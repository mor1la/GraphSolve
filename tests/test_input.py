import json

import networkx as nx
import pytest

from ptp.data import audit_and_template, load_instance
from ptp.solver import placement


def write_input(tmp_path):
    graph = nx.MultiGraph()
    graph.add_node("a", site_name="A", net_type="MBH")
    graph.add_node("b", site_name="B", net_type="MBH")
    graph.add_edge("a", "b", key="first")
    graph.add_edge("a", "b", key="second")
    nx.write_graphml(graph, tmp_path / "graph.graphml")
    config = {
        "graphml": "graph.graphml",
        "parameters": {
            "delays_us": {"MBH": 20, "RRN": 20, "optical": 50, "rrl": 150, "local": 3},
            "gm_capacity": 500,
            "minimum_gms": 2,
            "max_path_delay_us": 1500,
        },
        "stations": [
            {"id": "bs_a", "node": "a", "weight": 1},
            {"id": "bs_b", "node": "b", "weight": 2},
        ],
        "links": [
            {
                "source": "a",
                "target": "b",
                "key": key,
                "include": True,
                "kind": "optical",
                "cost": 5,
                "lag_group": "lag1",
            }
            for key in ("first", "second")
        ],
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def test_lag_aggregated_once_and_cli_input_is_solvable(tmp_path):
    path, _ = write_input(tmp_path)
    instance = load_instance(path)
    assert len(instance.links) == 1
    assert instance.links["lag1"].cost == 5
    assert placement(instance)["gm_count"] == 2


def test_missing_values_are_not_silently_imputed(tmp_path):
    path, config = write_input(tmp_path)
    config["parameters"]["delays_us"]["local"] = None
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(TypeError):
        load_instance(path)


def test_isolated_site_is_a_warning(tmp_path):
    graph = nx.MultiGraph()
    graph.add_node("a", site_name="A", net_type="MBH")
    graph.add_node("b", site_name="B", net_type="MBH")
    graph.add_node("c", site_name="C", net_type="MBH")
    graph.add_edge("a", "b")
    nx.write_graphml(graph, tmp_path / "graph.graphml")
    report = audit_and_template(tmp_path / "graph.graphml", tmp_path / "report")
    assert "placement_certificate" not in report
    assert report["warnings"][0]["sites"] == [{"site": "C", "nodes": ["c"]}]
    assert report["service_sites"] == 2


def test_isolated_site_does_not_block_placement_or_ranking(tmp_path):
    from ptp.solver import ranking

    path, config = write_input(tmp_path)
    graph = nx.read_graphml(tmp_path / "graph.graphml", force_multigraph=True)
    graph.add_node("c", site_name="C", net_type="MBH")
    graph.add_node("d", site_name="C", net_type="MBH")
    graph.add_edge("c", "d", key="local")
    nx.write_graphml(graph, tmp_path / "graph.graphml")
    config["links"].append(
        {
            "source": "c",
            "target": "d",
            "key": "local",
            "include": True,
            "kind": "local_optical",
            "cost": 1,
        }
    )
    config["stations"].append({"id": "excluded", "node": "c", "weight": 999})
    path.write_text(json.dumps(config), encoding="utf-8")
    instance = load_instance(path)
    result = placement(instance)
    assert result["proven_optimal"] and result["gm_count"] == 2
    assert result["excluded_stations"] == ["excluded"]
    assert result["warnings"][0]["sites"] == [{"site": "C", "nodes": ["c", "d"]}]
    ranked = ranking(instance, result["selected_gms"])
    assert ranked["status"] == "COMPLETE"
    assert ranked["rows"][-1]["effect"] == 3
    assert ranked["excluded_stations"] == ["excluded"]
    # Excluded sites do not require invented BS input.
    config["stations"].pop()
    path.write_text(json.dumps(config), encoding="utf-8")
    assert placement(load_instance(path))["gm_count"] == 2


def test_every_site_requires_bs_input(tmp_path):
    path, config = write_input(tmp_path)
    config["stations"].pop()
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="каждой площадке"):
        load_instance(path)
