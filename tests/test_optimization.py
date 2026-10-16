"""Tiny exhaustive oracles compare both MILPs with all feasible choices."""

import random
from itertools import combinations, pairwise, product

import networkx as nx
import pytest

from ptp.data import Instance, Link, Station
from ptp.solver import best_package, effect, placement, ranking, verify


def fixture(edges, demands, capacity=500, budget=1500, candidates=("a", "b")):
    graph = nx.MultiGraph()
    links = {}
    for eid, a, b, cost, delay in edges:
        for node in (a, b):
            graph.add_node(node, site=node, delay=20, kind="MBH")
        graph.add_edge(a, b, key=eid)
        links[eid] = Link(eid, a, b, "optical", cost, delay)
    return Instance(
        graph,
        links,
        [Station(*b) for b in demands],
        list(candidates),
        {n: 0 for n in graph},
        capacity,
        2,
        budget,
    )


def exhaustive_effect(instance, selected, ready, require_all=False):
    # Build independent delay graph and enumerate simple paths (tiny graphs only).
    graph = nx.Graph()
    graph.add_nodes_from(instance.graph)
    for eid in ready:
        e = instance.links[eid]
        if (
            not graph.has_edge(e.source, e.target)
            or e.delay < graph[e.source][e.target]["delay"]
        ):
            graph.add_edge(e.source, e.target, delay=e.delay)
    choices = []
    for bs in instance.stations:
        reachable = []
        for g in selected:
            paths = [[g]] if g == bs.node else nx.all_simple_paths(graph, g, bs.node)
            for path in paths:
                delay = sum(instance.graph.nodes[v]["delay"] for v in path)
                delay += sum(graph[a][b]["delay"] for a, b in pairwise(path))
                if delay <= instance.max_delay:
                    reachable.append(g)
                    break
        options = [
            (g, h) for g in reachable for h in reachable if instance.compatible(g, h)
        ]
        choices.append(options if require_all else [None, *options])
    best = -1
    for assignment in product(*choices):
        valid = True
        for failure in [None, *selected]:
            loads = dict.fromkeys(selected, 0)
            for pair in assignment:
                if pair is not None:
                    g, h = pair
                    loads[h if g == failure else g] += 1
            valid &= max(loads.values(), default=0) <= instance.capacity
        if valid:
            best = max(
                best,
                sum(
                    b.weight
                    for b, pair in zip(instance.stations, assignment)
                    if pair is not None
                ),
            )
    return best


def test_placement_matches_exhaustive_and_needs_three_gms():
    instance = fixture(
        [("ab", "a", "b", 0, 50), ("bc", "b", "c", 0, 50), ("ac", "a", "c", 0, 50)],
        [(f"bs{i}", "a", 1) for i in range(3)],
        capacity=2,
        candidates=("a", "b", "c"),
    )
    expected = next(
        k
        for k in range(2, 4)
        if any(
            exhaustive_effect(instance, subset, set(instance.links), True) == 3
            for subset in combinations(instance.candidates, k)
        )
    )
    result = placement(instance)
    assert result["proven_optimal"] and result["gm_count"] == expected == 3
    assert result["validation"]["verified"]


def test_milp_places_two_gms_in_each_disconnected_component():
    instance = fixture(
        [("ab", "a", "b", 0, 50), ("cd", "c", "d", 0, 50)],
        [(f"left{i}", "a", 1) for i in range(300)]
        + [(f"right{i}", "c", 1) for i in range(300)],
        capacity=500,
        candidates=("a", "b", "c", "d"),
    )
    result = placement(instance, fast=False)
    assert result["status"] == "OPTIMAL"
    assert result["gm_count"] == 4
    assert set(result["selected_gms"]) == {"a", "b", "c", "d"}
    assert result["validation"]["verified"]
    assert all(
        max(scenario["loads"].values(), default=0) <= 500
        for scenario in result["validation"]["failure_scenarios"]
    )


def test_capacity_includes_own_and_failed_gms_stations():
    instance = fixture(
        [("ab", "a", "b", 0, 50)],
        [("1", "a", 1), ("2", "b", 1), ("3", "a", 1)],
        capacity=2,
    )
    assert placement(instance)["status"] == "INFEASIBLE"
    with pytest.raises(ValueError, match="Перегрузка"):
        verify(
            instance, ["a", "b"], {"1": ["a", "b"], "2": ["b", "a"], "3": ["a", "b"]}
        )


def test_complementary_edges_and_shared_cost():
    instance = fixture(
        [
            ("ab", "a", "b", 0, 50),
            ("ad", "a", "d", 2, 50),
            ("dc", "d", "c", 3, 50),
            ("bc", "b", "c", 9, 50),
        ],
        [("bs", "c", 10)],
    )
    assert effect(instance, ["a", "b"], {"ab", "ad"})["effect"] == 0
    result = best_package(instance, ["a", "b"], {"ab"})
    assert result["proven_optimal"]
    assert set(result["package"]) == {"ad", "dc"}
    assert result["cost"] == 5 and result["ratio"] == 2
    ranked = ranking(instance, ["a", "b"])
    assert ranked["status"] == "COMPLETE"
    assert [r["gain"] for r in ranked["rows"]] == [0, 10]


@pytest.mark.parametrize("seed", range(8))
def test_best_package_matches_every_subset_and_assignment(seed):
    rng = random.Random(seed)
    instance = fixture(
        [
            ("ab", "a", "b", 0, 10),
            ("ac", "a", "c", rng.randint(1, 9), rng.randint(10, 70)),
            ("bc", "b", "c", rng.randint(1, 9), rng.randint(10, 70)),
            ("ad", "a", "d", rng.randint(1, 9), rng.randint(10, 70)),
            ("cd", "c", "d", rng.randint(1, 9), rng.randint(10, 70)),
        ],
        [("1", "a", 1), ("2", "c", 3), ("3", "d", 7)],
        capacity=2,
        budget=140,
    )
    ready = {"ab"}
    baseline = exhaustive_effect(instance, ["a", "b"], ready)
    expected = 0
    paid = sorted(set(instance.links) - ready)
    for count in range(1, len(paid) + 1):
        for subset in combinations(paid, count):
            value = exhaustive_effect(instance, ["a", "b"], ready | set(subset))
            expected = max(
                expected,
                (value - baseline) / sum(instance.links[e].cost for e in subset),
            )
    result = best_package(instance, ["a", "b"], ready)
    assert result["proven_optimal"]
    assert result.get("ratio", 0) == pytest.approx(expected)


def test_delay_counts_both_endpoints_and_infeasibility():
    instance = fixture([("ab", "a", "b", 0, 50)], [("bs", "a", 1)], budget=90)
    assert placement(instance)["proven_optimal"]
    instance.max_delay = 89
    assert placement(instance)["status"] == "INFEASIBLE"


def test_zero_importance_and_parallel_links():
    instance = fixture(
        [("ab", "a", "b", 0, 50), ("ac1", "a", "c", 2, 100), ("ac2", "a", "c", 3, 10)],
        [("bs", "c", 5)],
        budget=120,
    )
    result = best_package(instance, ["a", "b"], {"ab"})
    assert result["package"] == ["ac2"]
    instance.stations = [Station("bs", "c", 0)]
    assert best_package(instance, ["a", "b"], {"ab"})["status"] == "NO_IMPROVEMENT"


def test_same_site_cannot_be_primary_and_backup():
    instance = fixture([("ab", "a", "b", 0, 50)], [("bs", "a", 1)])
    instance.graph.nodes["b"]["site"] = "a"
    result = placement(instance)
    assert result["status"] == "NO_SERVICEABLE_STATIONS"
    assert result["warnings"][0]["sites"] == [{"site": "a", "nodes": ["a", "b"]}]


def test_disconnected_bs_cannot_be_covered():
    instance = fixture(
        [("ab", "a", "b", 0, 50), ("cd", "c", "d", 1, 50)], [("bs", "c", 1)]
    )
    assert placement(instance)["status"] == "INFEASIBLE"


@pytest.mark.parametrize("seed", range(6))
def test_placement_matches_exhaustive_with_varying_delay_and_capacity(seed):
    rng = random.Random(seed)
    instance = fixture(
        [
            ("ab", "a", "b", 0, rng.randint(10, 90)),
            ("bc", "b", "c", 0, rng.randint(10, 90)),
            ("ac", "a", "c", 0, rng.randint(10, 90)),
        ],
        [(f"bs{i}", rng.choice(["a", "b", "c"]), 1) for i in range(4)],
        capacity=rng.choice([2, 3, 4]),
        budget=rng.choice([70, 100, 180]),
        candidates=("a", "b", "c"),
    )
    expected = None
    for count in range(2, 4):
        if any(
            exhaustive_effect(instance, subset, set(instance.links), True) == 4
            for subset in combinations(instance.candidates, count)
        ):
            expected = count
            break
    result = placement(instance)
    if expected is None:
        assert result["status"] == "INFEASIBLE"
    else:
        assert result["proven_optimal"] and result["gm_count"] == expected


def test_time_limited_incumbent_is_not_reported_as_optimal(monkeypatch):
    from ptp.solver import Model

    solve = Model.solve

    def limited(self, objective, seconds):
        x, info = solve(self, objective, seconds)
        info.update(status="LIMIT", proven_optimal=False)
        return x, info

    monkeypatch.setattr(Model, "solve", limited)
    instance = fixture([("ab", "a", "b", 0, 50)], [("bs", "a", 1)])
    result = placement(instance, fast=False)
    assert result["gm_count"] == 2 and not result["proven_optimal"]
    instance.capacity = 2
    instance.stations.extend([Station("bs2", "a", 1), Station("bs3", "b", 1)])
    package = best_package(instance, ["a", "b"], {"ab"})
    assert package["status"] == "BASELINE_NOT_PROVEN"
    assert not package["proven_optimal"]
