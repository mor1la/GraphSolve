"""GraphML RRN/MBH nodes are BS; require explicit missing attributes."""

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import networkx as nx


@dataclass(frozen=True)
class Station:
    id: str
    node: str
    weight: float


@dataclass(frozen=True)
class Link:
    id: str
    source: str
    target: str
    kind: str
    cost: float
    delay: float


@dataclass
class Instance:
    graph: nx.MultiGraph
    links: dict[str, Link]
    stations: list[Station]
    candidates: list[str]
    optical: dict[str, int]
    capacity: int
    minimum_gms: int
    max_delay: float
    optical_rule: str = "path"
    optical_edges: set | None = None
    warnings: list = field(default_factory=list)
    excluded_stations: list = field(default_factory=list)

    def compatible(self, g, h):
        a, b = self.graph.nodes[g]["site"], self.graph.nodes[h]["site"]
        if a == b:
            return False
        if self.optical_rule == "direct":
            return frozenset((a, b)) in self.optical_edges
        return self.optical[a] == self.optical[b]

    def routing_graph(self, ready=None):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.graph.nodes)
        for link in self.links.values():
            if ready is not None and link.id not in ready:
                continue
            for a, b in ((link.source, link.target), (link.target, link.source)):
                weight = link.delay + self.graph.nodes[b]["delay"]
                if not graph.has_edge(a, b) or weight < graph[a][b]["weight"]:
                    graph.add_edge(a, b, weight=weight, link=link.id)
        return graph

    def pairs(self, candidates=None, ready=None):
        candidates = self.candidates if candidates is None else candidates
        graph = self.routing_graph(ready)
        reachable = {b.id: [] for b in self.stations}
        for g in candidates:
            cutoff = self.max_delay - self.graph.nodes[g]["delay"]
            distances = (
                nx.single_source_dijkstra_path_length(graph, g, cutoff=cutoff)
                if cutoff >= 0
                else {}
            )
            for b in self.stations:
                if b.node in distances:
                    reachable[b.id].append(g)
        return {
            b: tuple((g, h) for g in gs for h in gs if self.compatible(g, h))
            for b, gs in reachable.items()
        }


def isolated_sites(graph, site_attribute="site"):
    groups = {}
    for node, attrs in graph.nodes(data=True):
        groups.setdefault(attrs[site_attribute], []).append(node)
    external = set()
    for a, b in graph.edges():
        sa, sb = graph.nodes[a][site_attribute], graph.nodes[b][site_attribute]
        if sa != sb:
            external.update((sa, sb))
    return [
        {"site": site, "nodes": sorted(nodes)}
        for site, nodes in sorted(groups.items())
        if site not in external
    ]


def isolation_warning(sites):
    return {
        "code": "ISOLATED_SITES_EXCLUDED",
        "sites": sites,
        "message": "Изолированные площадки исключены из расчета покрытия: "
        + ", ".join(s["site"] for s in sites),
    }


def service_scope(instance):
    sites = isolated_sites(instance.graph)
    if not sites:
        return instance
    excluded = {n for s in sites for n in s["nodes"]}
    return replace(
        instance,
        graph=instance.graph.subgraph(set(instance.graph) - excluded).copy(),
        links={
            e: link
            for e, link in instance.links.items()
            if link.source not in excluded and link.target not in excluded
        },
        stations=[b for b in instance.stations if b.node not in excluded],
        candidates=[g for g in instance.candidates if g not in excluded],
        warnings=[*instance.warnings, isolation_warning(sites)],
        excluded_stations=[
            *instance.excluded_stations,
            *[b.id for b in instance.stations if b.node in excluded],
        ],
    )


def number(value, name, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name}: требуется явное числовое значение")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name}: недопустимое значение {value}")
    if integer and int(value) != value:
        raise ValueError(f"{name}: требуется целое число")
    return int(value) if integer else float(value)


def load_instance(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    raw = nx.read_graphml(path.parent / config["graphml"], force_multigraph=True)
    if raw.is_directed():
        raise ValueError("Поддерживается неориентированная L2-топология")
    p = config["parameters"]
    delays = {k: number(v, f"delay.{k}") for k, v in p["delays_us"].items()}
    capacity = number(p["gm_capacity"], "gm_capacity", True, True)
    minimum = number(p["minimum_gms"], "minimum_gms", True, True)
    max_delay = number(p["max_path_delay_us"], "max_path_delay_us", True)
    graph = nx.MultiGraph()
    for node, data in raw.nodes(data=True):
        kind = data.get("net_type")
        site = data.get("site_name")
        if not site or kind not in delays:
            raise ValueError(
                f"Узел {node}: неизвестны площадка или задержка типа {kind}"
            )
        graph.add_node(node, site=site, delay=delays[kind], kind=kind)
    optical = nx.Graph()
    optical.add_nodes_from(d["site"] for _, d in graph.nodes(data=True))
    links, seen, groups, candidates = {}, set(), {}, set()
    for row in config["links"]:
        a, b, key = row["source"], row["target"], row["key"]
        ref = (frozenset((a, b)), str(key))
        raw_keys = raw.get_edge_data(a, b, default={})
        if ref in seen or not any(str(actual_key) == str(key) for actual_key in raw_keys):
            raise ValueError(f"Неизвестное или повторное ребро: {row}")
        seen.add(ref)
        if not isinstance(row["include"], bool):
            raise TypeError(f"Ребро {a}/{b}/{key}: явно укажите include")
        if not row["include"]:
            continue
        kind = row["kind"]
        if kind not in ("optical", "rrl", "local_optical", "local_copper"):
            raise ValueError(f"Ребро {a}/{b}/{key}: явно укажите kind")
        local = graph.nodes[a]["site"] == graph.nodes[b]["site"]
        if local != kind.startswith("local_"):
            raise ValueError(
                f"Тип {kind} не соответствует площадкам ребра {a}/{b}/{key}"
            )
        cost = number(row["cost"], f"cost.{a}/{b}/{key}")
        lag = row.get("lag_group")
        link_id = str(lag) if lag else json.dumps([a, b, str(key)], ensure_ascii=False)
        signature = (frozenset((a, b)), kind, cost)
        if lag and lag in groups:
            if groups[lag] != signature:
                raise ValueError(
                    f"LAG {lag}: концы, тип и агрегированная стоимость должны совпадать"
                )
            continue
        if lag:
            groups[lag] = signature
        if link_id in links:
            raise ValueError(f"Повторный ID линка {link_id}")
        delay = delays["local"] if local else delays[kind]
        links[link_id] = Link(link_id, a, b, kind, cost, delay)
        graph.add_edge(a, b, key=link_id)
        if kind in ("optical", "local_optical"):
            for node in (a, b):
                if graph.nodes[node]["kind"] == "MBH":
                    candidates.add(node)
            if not local:
                optical.add_edge(graph.nodes[a]["site"], graph.nodes[b]["site"])
    if len(seen) != raw.number_of_edges():
        raise ValueError(
            "Требуется классификация всех исходных ребер, включая исключенные"
        )
    stations = []
    for row in config["stations"]:
        if row["node"] not in graph:
            raise ValueError(f"Неизвестный узел БС {row}")
        stations.append(
            Station(str(row["id"]), row["node"], number(row["weight"], "weight"))
        )
    if len({b.id for b in stations}) != len(stations):
        raise ValueError("Нужен список БС с уникальными ID")
    isolated = {s["site"] for s in isolated_sites(graph)}
    missing_sites = (
        set(optical) - isolated - {graph.nodes[b.node]["site"] for b in stations}
    )
    if missing_sites:
        raise ValueError(
            f"По условию на каждой площадке есть БС. Не заданы: {sorted(missing_sites)}"
        )
    rule = p.get("optical_rule", "path")
    if rule not in ("path", "direct"):
        raise ValueError("optical_rule должен быть path или direct")
    components = {
        s: i for i, comp in enumerate(nx.connected_components(optical)) for s in comp
    }
    return Instance(
        graph,
        links,
        stations,
        sorted(candidates),
        components,
        capacity,
        minimum,
        max_delay,
        rule,
        {frozenset(e) for e in optical.edges},
    )


def audit_and_template(graph_path, directory):
    graph_path, directory = Path(graph_path).resolve(), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    import os

    template = {
        "graphml": os.path.relpath(graph_path, directory.resolve()),
        "parameters": {
            "delays_us": {
                "MBH": 20,
                "RRN": 20,
                "optical": 50,
                "rrl": 150,
                "local": None,
            },
            "gm_capacity": 500,
            "minimum_gms": 2,
            "max_path_delay_us": 1500,
            "optical_rule": "path",
        },
        "stations": [
            {"id": node, "node": node, "weight": None}
            for node, attrs in sorted(graph.nodes(data=True))
            if attrs.get("net_type") in ("RRN", "MBH")
        ],
        "links": [
            {
                "source": a,
                "target": b,
                "key": k,
                "include": None,
                "kind": None,
                "cost": None,
                "lag_group": None,
                "source_hint": d.get("extra_info", ""),
            }
            for a, b, k, d in graph.edges(keys=True, data=True)
        ],
    }
    report = {
        "status": "INPUT_INCOMPLETE",
        "nodes": len(graph),
        "stations": sum(
            d.get("net_type") in ("RRN", "MBH") for _, d in graph.nodes(data=True)
        ),
        "edges": graph.number_of_edges(),
        "sites": len({d.get("site_name") for _, d in graph.nodes(data=True)}),
        "components": sorted(map(len, nx.connected_components(graph)), reverse=True),
        "isolates": [
            {"node": n, "site": graph.nodes[n].get("site_name")}
            for n in nx.isolates(graph)
        ],
        "lag_marked_records": sum(
            "LAG" in d.get("extra_info", "") for *_, d in graph.edges(data=True)
        ),
        "l3_marked_records": sum(
            "L3" in d.get("extra_info", "") for *_, d in graph.edges(data=True)
        ),
        "missing": [
            "BS importance",
            "upgrade costs",
            "confirmed link media",
            "local delay",
            "L2/LAG normalization decisions",
        ],
        "optimality": "Cannot evaluate tasks 2 and 3 on the real instance without these inputs.",
    }
    isolated = isolated_sites(graph, "site_name")
    report["warnings"] = [isolation_warning(isolated)] if isolated else []
    report["excluded_sites"] = [s["site"] for s in isolated]
    report["service_sites"] = report["sites"] - len(isolated)
    report["optimality"] = (
        "Isolated sites produce a warning and are excluded from coverage. "
        "Optimality for the remaining sites requires complete input."
    )
    for name, data in (("input.template.json", template), ("audit.json", report)):
        (directory / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report
