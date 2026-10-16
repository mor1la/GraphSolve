"""Sparse MILP models, independently checked incumbents, and exact ratio search."""

import math
import time
from collections import Counter, defaultdict
from functools import wraps
from itertools import pairwise

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .data import service_scope


def scoped(function):
    @wraps(function)
    def call(instance, *args, **kwargs):
        instance = service_scope(instance)
        result = function(instance, *args, **kwargs)
        result.update(
            warnings=instance.warnings,
            excluded_stations=instance.excluded_stations,
            coverage_scope="non_isolated_sites",
            included_station_count=len(instance.stations),
        )
        return result

    return call


class Model:
    def __init__(self):
        self.upper = []
        self.rows, self.columns, self.values = [], [], []
        self.lower_rows, self.upper_rows = [], []
        self.constraint = None

    def var(self, upper=1):
        self.upper.append(upper)
        return len(self.upper) - 1

    def row(self, terms, lower=-np.inf, upper=np.inf):
        index = len(self.lower_rows)
        for column, value in terms.items():
            if value:
                self.rows.append(index)
                self.columns.append(column)
                self.values.append(value)
        self.lower_rows.append(lower)
        self.upper_rows.append(upper)
        self.constraint = None

    def solve(self, objective, seconds):
        if self.constraint is None:
            a = coo_matrix(
                (self.values, (self.rows, self.columns)),
                shape=(len(self.lower_rows), len(self.upper)),
            ).tocsc()
            self.constraint = LinearConstraint(a, self.lower_rows, self.upper_rows)
        c = np.zeros(len(self.upper))
        for index, value in objective.items():
            c[index] = value
        started = time.perf_counter()
        result = milp(
            c,
            integrality=np.ones(len(c)),
            bounds=Bounds(np.zeros(len(c)), self.upper),
            constraints=self.constraint,
            options={"time_limit": seconds, "mip_rel_gap": 0.0},
        )
        elapsed = time.perf_counter() - started
        incumbent_ok = False
        if result.x is not None:
            x = np.rint(result.x)
            ax = self.constraint.A @ x
            incumbent_ok = bool(
                np.all(np.abs(result.x - x) <= 1e-5)
                and np.all(x >= 0)
                and np.all(x <= self.upper)
                and np.all(ax >= np.array(self.lower_rows) - 1e-6)
                and np.all(ax <= np.array(self.upper_rows) + 1e-6)
            )
        status = {
            0: "OPTIMAL",
            1: "LIMIT",
            2: "INFEASIBLE",
            3: "UNBOUNDED",
            4: "ERROR",
        }.get(result.status, "ERROR")
        if result.x is not None and not incumbent_ok:
            status = "INVALID_INCUMBENT"
        info = {
            "status": status,
            "message": result.message,
            "proven_optimal": status == "OPTIMAL" and incumbent_ok,
            "seconds": elapsed,
            "variables": len(c),
            "constraints": len(self.lower_rows),
            "nonzeros": len(self.values),
            "objective": float(c @ x) if incumbent_ok else None,
            "dual_bound": finite(getattr(result, "mip_dual_bound", None)),
            "relative_gap": finite(getattr(result, "mip_gap", None)),
            "numerical_tolerance": 1e-6,
        }
        return (x if incumbent_ok else None), info


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def assignments_model(instance, pairs, candidates, placement=False, grouped=False):
    model = Model()
    groups = defaultdict(list)
    for b in instance.stations:
        groups[pairs[b.id] if grouped else b.id].append(b)
    xs = {g: model.var() for g in candidates} if placement else {}
    zs, ys = {}, {}
    primary, reserve = defaultdict(dict), defaultdict(dict)
    for members in groups.values():
        representative = members[0].id
        choices = pairs[representative]
        bound = len(members)
        total = {}
        for g, h in choices:
            z = model.var(bound)
            zs[representative, g, h] = (z, members)
            total[z] = 1
            primary[g][z] = 1
            reserve[g, h][z] = 1
            if placement:
                model.row({z: 1, xs[g]: -bound}, upper=0)
                model.row({z: 1, xs[h]: -bound}, upper=0)
        if placement:
            model.row(total, bound, bound)
        else:
            y = model.var()
            ys[representative] = y
            total[y] = -1
            model.row(total, 0, 0)
    for g in candidates:
        base = dict(primary[g])
        if placement:
            base[xs[g]] = -instance.capacity
        upper = 0 if placement else instance.capacity
        model.row(base, upper=upper)
        for f in candidates:
            if f != g and reserve[f, g]:
                row = dict(base)
                row.update(reserve[f, g])
                model.row(row, upper=upper)
    if placement:
        lower = max(
            instance.minimum_gms,
            1 + math.ceil(len(instance.stations) / instance.capacity),
        )
        model.row({x: 1 for x in xs.values()}, lower=lower)
    return model, xs, zs, ys


def decode(x, zs):
    assignments = {}
    offsets = Counter()
    for (representative, g, h), (z, members) in zs.items():
        count = int(x[z])
        for b in members[offsets[representative] : offsets[representative] + count]:
            assignments[b.id] = [g, h]
        offsets[representative] += count
    return assignments


@scoped
def verify(instance, selected, assignments, ready=None, require_all=False):
    """Independent graph traversal and explicit simulation of every GM failure."""
    selected = set(selected)
    if not selected <= set(instance.candidates):
        raise ValueError("Выбран недопустимый GM")
    stations = {b.id: b for b in instance.stations}
    if not set(assignments) <= set(stations):
        raise ValueError("Неизвестная БС в решении")
    if require_all and (
        set(assignments) != set(stations)
        or (stations and len(selected) < instance.minimum_gms)
    ):
        raise ValueError("Не обеспечено полное покрытие/минимальное число GM")
    graph = instance.routing_graph(ready)
    routes = {gm: nx.single_source_dijkstra(graph, gm) for gm in selected}
    paths = {}
    for bid, (g, h) in assignments.items():
        if not {g, h} <= selected or not instance.compatible(g, h):
            raise ValueError(f"Недопустимая пара для {bid}")
        paths[bid] = []
        for gm in (g, h):
            distances, route_paths = routes[gm]
            target = stations[bid].node
            if target not in distances:
                raise ValueError(f"Нет пути от {gm} к {bid}")
            distance, path = distances[target], route_paths[target]
            delay = distance + instance.graph.nodes[gm]["delay"]
            if delay > instance.max_delay + 1e-6:
                raise ValueError(f"Превышена задержка для {bid}")
            paths[bid].append(
                {
                    "gm": gm,
                    "delay_us": delay,
                    "nodes": path,
                    "links": [graph[a][b]["link"] for a, b in pairwise(path)],
                }
            )
    scenarios = []
    for failed in [None, *sorted(selected)]:
        loads = Counter(h if g == failed else g for g, h in assignments.values())
        if any(load > instance.capacity for load in loads.values()) or (
            failed in loads if failed else False
        ):
            raise ValueError(f"Перегрузка GM при отказе {failed}")
        scenarios.append({"failed_gm": failed, "loads": dict(loads)})
    return {"verified": True, "paths": paths, "failure_scenarios": scenarios}


@scoped
def placement(instance, seconds=60, fast=True):
    if not instance.stations:
        return {
            "status": "NO_SERVICEABLE_STATIONS",
            "proven_optimal": False,
            "selected_gms": [],
            "gm_count": 0,
            "assignments": {},
        }
    # Universal GM sites give an exact certificate when they attain the
    # single-failure capacity lower bound.
    lower = max(instance.minimum_gms, 1 + math.ceil(len(instance.stations) / instance.capacity))
    if fast and len(instance.candidates) >= lower:
        graph = instance.routing_graph()
        universal = []
        for gm in instance.candidates:
            distances = nx.single_source_dijkstra_path_length(
                graph, gm, cutoff=instance.max_delay - instance.graph.nodes[gm]["delay"])
            if all(b.node in distances for b in instance.stations):
                universal.append((max(distances[b.node] for b in instance.stations), gm))
        universal.sort()
        for _, backup in universal:
            primaries = []
            sites = {instance.graph.nodes[backup]["site"]}
            for _, gm in universal:
                site = instance.graph.nodes[gm]["site"]
                if site not in sites and instance.compatible(gm, backup):
                    primaries.append(gm)
                    sites.add(site)
                    if len(primaries) == lower - 1:
                        break
            if len(primaries) == lower - 1:
                selected = [*primaries, backup]
                assignments = {
                    b.id: [primaries[index % len(primaries)], backup]
                    for index, b in enumerate(instance.stations)
                }
                return {
                    "status": "OPTIMAL", "proven_optimal": True,
                    "gm_count": lower, "selected_gms": selected,
                    "assignments": assignments,
                    "sites": {gm: instance.graph.nodes[gm]["site"] for gm in selected},
                    "objective": lower, "dual_bound": lower, "relative_gap": 0,
                    "certificate": "Universal GM sites and verified failover loads attain the single-failure capacity lower bound",
                    "validation": verify(instance, selected, assignments, require_all=True),
                }
    pairs = instance.pairs()
    impossible = [b for b, choices in pairs.items() if not choices]
    if impossible:
        return {
            "status": "INFEASIBLE",
            "proven_optimal": False,
            "infeasibility_proven": True,
            "unserviceable_stations": impossible,
            "reason": "No compatible pair within delay budget even after all upgrades",
        }
    model, xs, zs, _ = assignments_model(
        instance, pairs, instance.candidates, True, True
    )
    x, info = model.solve({v: 1 for v in xs.values()}, seconds)
    info["infeasibility_proven"] = info["status"] == "INFEASIBLE"
    if x is not None:
        selected = [g for g, v in xs.items() if x[v]]
        assigned = decode(x, zs)
        info.update(
            selected_gms=selected,
            gm_count=len(selected),
            assignments=assigned,
            sites={g: instance.graph.nodes[g]["site"] for g in selected},
            validation=verify(instance, selected, assigned, require_all=True),
        )
    return info


@scoped
def effect(instance, selected, ready, seconds=60):
    if not instance.stations:
        return {
            "status": "OPTIMAL",
            "proven_optimal": True,
            "effect": 0,
            "assignments": {},
            "validation": verify(instance, selected, {}),
        }
    if len(selected) == 2 and len(instance.stations) <= instance.capacity:
        graph = instance.routing_graph(ready)
        distances = {g: nx.single_source_dijkstra_path_length(
            graph, g, cutoff=instance.max_delay - instance.graph.nodes[g]["delay"])
                     for g in selected}
        assignments = {}
        if instance.compatible(*selected):
            for b in instance.stations:
                if all(b.node in distances[g] for g in selected):
                    assignments[b.id] = sorted(selected, key=lambda g: (distances[g][b.node], g))
        return {"status": "OPTIMAL", "proven_optimal": True,
                "effect": sum(b.weight for b in instance.stations if b.id in assignments),
                "assignments": assignments,
                "certificate": "All jointly reachable BS fit on either of the two GM",
                "validation": verify(instance, selected, assignments, ready)}
    pairs = instance.pairs(selected, ready)
    model, _, zs, ys = assignments_model(instance, pairs, selected)
    x, info = model.solve({ys[b.id]: -b.weight for b in instance.stations}, seconds)
    if x is not None:
        assigned = decode(x, zs)
        info.update(
            effect=sum(b.weight for b in instance.stations if b.id in assigned),
            assignments=assigned,
            validation=verify(instance, selected, assigned, ready),
        )
    return info


def network_model(instance, selected, ready):
    """Two binary flows per BS; each physical edge is paid for only once."""
    pairs = instance.pairs(selected)
    model, _, zs, ys = assignments_model(instance, pairs, selected)
    upgrades = {eid: model.var() for eid in instance.links if eid not in ready}
    delay_graph = instance.routing_graph()
    for b in instance.stations:
        y = ys[b.id]
        if not pairs[b.id]:
            continue
        to_target = nx.single_source_dijkstra_path_length(
            delay_graph.reverse(copy=False), b.node, cutoff=instance.max_delay
        )
        for role in (0, 1):
            sources = {pair[role] for pair in pairs[b.id]}
            from_source = nx.multi_source_dijkstra_path_length(
                delay_graph, sources, cutoff=instance.max_delay
            )
            start_lower = min(instance.graph.nodes[g]["delay"] for g in sources)
            balance = {node: defaultdict(float) for node in instance.graph}
            delay = defaultdict(float)
            delay[y] = -instance.max_delay
            balance[b.node][y] += 1
            for g, h in pairs[b.id]:
                gm = (g, h)[role]
                z = zs[b.id, g, h][0]
                balance[gm][z] -= 1
                delay[z] += instance.graph.nodes[gm]["delay"]
            for eid, link in instance.links.items():
                for a, dest in ((link.source, link.target), (link.target, link.source)):
                    lower = (
                        start_lower
                        + from_source.get(a, math.inf)
                        + link.delay
                        + instance.graph.nodes[dest]["delay"]
                        + to_target.get(dest, math.inf)
                    )
                    if lower > instance.max_delay + 1e-9:
                        continue
                    f = model.var()
                    balance[a][f] += 1
                    balance[dest][f] -= 1
                    delay[f] = link.delay + instance.graph.nodes[dest]["delay"]
                    model.row({f: 1, y: -1}, upper=0)
                    if eid not in ready:
                        model.row({f: 1, upgrades[eid]: -1}, upper=0)
            for terms in balance.values():
                model.row(terms, 0, 0)
            model.row(delay, upper=0)
    return model, upgrades, zs, ys


@scoped
def best_package(instance, selected, ready, seconds=60, iterations=100):
    """Exact Dinkelbach steps if each MILP finishes; never certify a timed-out solve."""
    baseline = effect(instance, selected, ready, seconds)
    if not baseline["proven_optimal"]:
        return {
            "status": "BASELINE_NOT_PROVEN",
            "proven_optimal": False,
            "baseline": baseline,
        }
    remaining = set(instance.links) - ready
    if not remaining:
        return {"status": "NO_IMPROVEMENT", "proven_optimal": True, "package": []}
    if any(instance.links[e].cost <= 0 for e in remaining):
        raise ValueError("Все бесплатные линки должны входить в ready")
    model, upgrades, zs, ys = network_model(instance, selected, ready)
    model.row({v: 1 for v in upgrades.values()}, lower=1)
    ratio, history = 0.0, []
    answer = {}
    for _ in range(iterations):
        objective = {ys[b.id]: -b.weight for b in instance.stations}
        objective.update(
            {v: ratio * instance.links[e].cost for e, v in upgrades.items()}
        )
        x, info = model.solve(objective, seconds)
        history.append(info)
        if x is None or not info["proven_optimal"]:
            return {
                "status": "INNER_SOLVE_NOT_PROVEN",
                "proven_optimal": False,
                "history": history,
                "last_proven_candidate": answer,
            }
        package = sorted(e for e, v in upgrades.items() if x[v])
        assigned = decode(x, zs)
        validation = verify(instance, selected, assigned, ready | set(package))
        value = sum(b.weight for b in instance.stations if b.id in assigned)
        gain = value - baseline["effect"]
        cost = sum(instance.links[e].cost for e in package)
        residual = gain - ratio * cost
        answer = {
            "package": package,
            "cost": cost,
            "gain": gain,
            "ratio": gain / cost,
            "baseline_effect": baseline["effect"],
            "effect": value,
            "assignments": assigned,
            "validation": validation,
        }
        if gain <= 1e-7 and ratio == 0:
            return {
                "status": "NO_IMPROVEMENT",
                "proven_optimal": True,
                "package": [],
                "history": history,
            }
        if abs(residual) <= 1e-7 * max(1.0, abs(gain), abs(ratio * cost)):
            return {
                "status": "OPTIMAL",
                "proven_optimal": True,
                **answer,
                "history": history,
                "ratio_residual": residual,
            }
        ratio = gain / cost
    return {
        "status": "ITERATION_LIMIT",
        "proven_optimal": False,
        **answer,
        "history": history,
    }


@scoped
def ranking(instance, selected, seconds=60, max_packages=100):
    ready = {e for e, link in instance.links.items() if link.cost == 0}
    packages, rows = [], []
    for index in range(max_packages):
        result = best_package(instance, selected, ready, seconds)
        packages.append(result)
        if result["status"] == "NO_IMPROVEMENT":
            return {
                "status": "COMPLETE",
                "each_package_optimal": True,
                "globally_optimal_ranking": False,
                "packages": packages,
                "rows": rows,
                "remaining_links": sorted(set(instance.links) - ready),
            }
        if not result["proven_optimal"]:
            return {
                "status": "INCOMPLETE",
                "each_package_optimal": False,
                "globally_optimal_ranking": False,
                "packages": packages,
                "rows": rows,
            }
        previous = result["baseline_effect"]
        # Committed package, deterministic cost/ID order; no claim of optimal internal order.
        for eid in sorted(result["package"], key=lambda e: (instance.links[e].cost, e)):
            ready.add(eid)
            state = effect(instance, selected, ready, seconds)
            if not state["proven_optimal"]:
                return {
                    "status": "INCOMPLETE",
                    "each_package_optimal": True,
                    "globally_optimal_ranking": False,
                    "packages": packages,
                    "rows": rows,
                    "unproven_step": state,
                }
            rows.append(
                {
                    "position": len(rows) + 1,
                    "package": index + 1,
                    "link": eid,
                    "cost": instance.links[eid].cost,
                    "effect": state["effect"],
                    "gain": state["effect"] - previous,
                    "package_ratio": result["ratio"],
                }
            )
            previous = state["effect"]
    return {
        "status": "PACKAGE_LIMIT",
        "each_package_optimal": True,
        "globally_optimal_ranking": False,
        "packages": packages,
        "rows": rows,
    }
