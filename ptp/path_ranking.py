"""Bounded path-package heuristic; supports fixed verified failover pairs."""

import time
from itertools import pairwise, product

import networkx as nx

from .data import service_scope
from .solver import effect, verify


def path_ranking(original, selected, candidate_limit=32, assignments=None):
    instance = service_scope(original)
    fixed = len(selected) != 2 or len(instance.stations) > instance.capacity
    if fixed and (assignments is None or set(assignments) != {b.id for b in instance.stations}):
        raise ValueError("Ranking with more than two GM requires verified complete assignments")
    if not fixed and not instance.compatible(*selected):
        raise ValueError("Incompatible GM pair")
    if fixed:
        verify(instance, selected, assignments, require_all=True)
    started = time.perf_counter()
    graph = instance.routing_graph()
    delay_paths = {g: nx.single_source_dijkstra(graph, g)[1] for g in selected}
    weights = {b.id: b.weight for b in instance.stations}
    station_nodes = {b.id: b.node for b in instance.stations}
    ready = {e for e, link in instance.links.items() if link.cost == 0}
    rows, packages, steps = [], [], []
    cumulative = 0

    def fixed_effect(enabled, validate=False):
        routing = instance.routing_graph(enabled)
        reachable = {g: set(nx.single_source_dijkstra_path_length(
            routing, g, cutoff=instance.max_delay - instance.graph.nodes[g]["delay"]))
            for g in selected}
        served = {bid: pair for bid, pair in assignments.items()
                  if station_nodes[bid] in reachable[pair[0]]
                  and station_nodes[bid] in reachable[pair[1]]}
        result = {"effect": sum(weights[bid] for bid in served),
                  "assignments": served, "proven_optimal": False}
        if validate:
            result["validation"] = verify(instance, selected, served, enabled)
        return result

    def save_state():
        state = fixed_effect(ready, True) if fixed else effect(instance, selected, ready)
        steps.append({"ready": sorted(ready), "cost": cumulative,
                      "effect": state["effect"], "assignments": state["assignments"],
                      "validation": state["validation"], "proven_optimal": state["proven_optimal"]})
        return state

    def path_edges(path, routing):
        return {routing[a][b]["link"] for a, b in pairwise(path)}

    def value(enabled):
        if fixed:
            return fixed_effect(enabled)["effect"]
        routing = instance.routing_graph(enabled)
        reachable = [set(nx.single_source_dijkstra_path_length(
            routing, g, cutoff=instance.max_delay - instance.graph.nodes[g]["delay"]))
            for g in selected]
        return sum(b.weight for b in instance.stations
                   if b.node in reachable[0] and b.node in reachable[1])

    state = save_state()
    while len(state["assignments"]) < len(instance.stations):
        cost_graph = nx.DiGraph()
        cost_graph.add_nodes_from(instance.graph)
        for eid, link in instance.links.items():
            for a, b in ((link.source, link.target), (link.target, link.source)):
                delay = link.delay + instance.graph.nodes[b]["delay"]
                weight = (0 if eid in ready else link.cost) + delay * 1e-6
                if not cost_graph.has_edge(a, b) or weight < cost_graph[a][b]["weight"]:
                    cost_graph.add_edge(a, b, weight=weight, delay=delay, link=eid)
        cost_paths = {g: nx.single_source_dijkstra(cost_graph, g)[1] for g in selected}
        unserved = {b.id for b in instance.stations if b.id not in state["assignments"]}
        pool = {}
        for bid in sorted(unserved):
            node = station_nodes[bid]
            options = []
            for gm in (assignments[bid] if fixed else selected):
                paths = []
                if node in delay_paths[gm]:
                    path = delay_paths[gm][node]
                    paths.append(path_edges(path, graph))
                if node in cost_paths[gm]:
                    path = cost_paths[gm][node]
                    delay = instance.graph.nodes[gm]["delay"] + sum(
                        cost_graph[a][b]["delay"] for a, b in pairwise(path))
                    if delay <= instance.max_delay:
                        paths.append(path_edges(path, cost_graph))
                options.append(paths)
            for first, second in product(*options):
                package = frozenset((first | second) - ready)
                if package:
                    pool.setdefault(package, set()).add(bid)
        if not pool:
            raise RuntimeError("No path packages can serve the remaining BS")
        proposals = sorted(pool, key=lambda p: (
            -sum(weights[n] for n in pool[p]) / sum(instance.links[e].cost for e in p),
            tuple(sorted(p))))[:candidate_limit]
        best = None
        for package in proposals:
            cost = sum(instance.links[e].cost for e in package)
            gain = value(ready | package) - state["effect"]
            key = (gain / cost, gain, -cost)
            if gain > 0 and (best is None or key > best[0]):
                best = key, package, cost, gain
        if best is None:
            raise RuntimeError("No positive-gain package; zero-weight demands need a secondary objective")
        key, package, cost, gain = best
        packages.append({"status": "HEURISTIC", "proven_optimal": False,
                         "package": sorted(package), "cost": cost, "gain": gain,
                         "ratio": key[0], "candidates_evaluated": len(proposals)})
        for eid in sorted(package, key=lambda e: (instance.links[e].cost, e)):
            previous = state["effect"]
            ready.add(eid)
            cumulative += instance.links[eid].cost
            state = save_state()
            rows.append({"position": len(rows) + 1, "package": len(packages), "link": eid,
                         "cost": instance.links[eid].cost, "effect": state["effect"],
                         "gain": state["effect"] - previous, "package_ratio": key[0]})
        print(f"Package {len(packages)}: BS {len(state['assignments'])}/{len(instance.stations)}, "
              f"cost {cumulative:g}, effect {state['effect']:g}", flush=True)
    return {"status": "COMPLETE_HEURISTIC", "method": "bounded_path_packages",
            "each_package_optimal": False, "globally_optimal_ranking": False,
            "candidate_limit": candidate_limit, "packages": packages, "rows": rows,
            "remaining_links": sorted(set(instance.links) - ready),
            "warnings": instance.warnings, "excluded_stations": instance.excluded_stations,
            "seconds": time.perf_counter() - started,
            "total_cost": cumulative, "final_effect": state["effect"],
            "served_stations": len(state["assignments"]),
            "coverage_scope": "non_isolated_sites"}, steps
