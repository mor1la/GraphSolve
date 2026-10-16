"""Generate reproducible SYNTHETIC inputs on the supplied, unchanged topology."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx


SEED = "mtc-synthetic-2026-10-16-v1"


def score(label):
    return int.from_bytes(hashlib.sha256(f"{SEED}:{label}".encode()).digest()[:8], "big")


def main():
    directory = Path(__file__).resolve().parent
    source = directory / "sample.graphml"
    graph = nx.read_graphml(source, force_multigraph=True)
    lag_records = defaultdict(list)
    for a, b, key, attrs in graph.edges(keys=True, data=True):
        hint = attrs.get("extra_info", "")
        if hint.startswith("L2") and "LAG" in hint:
            lag_records[tuple(sorted((a, b)))].append((key, hint))
    for pair, records in lag_records.items():
        if sum("MEMBER" not in hint for _, hint in records) != 1:
            raise ValueError(f"Ambiguous LAG: {pair}; classify manually")

    links = []
    for a, b, key, attrs in graph.edges(keys=True, data=True):
        hint = attrs.get("extra_info", "")
        local = graph.nodes[a]["site_name"] == graph.nodes[b]["site_name"]
        pair = tuple(sorted((a, b)))
        lag = hint.startswith("L2") and "LAG" in hint
        related = lag_records[pair] if lag else [(key, hint)]
        fiber = any("FO" in text.split() for _, text in related)
        # Media absent from GraphML is an explicit synthetic assumption.
        if local:
            kind = "local_optical" if fiber else "local_copper"
        else:
            kind = "optical" if fiber or all(
                graph.nodes[n]["net_type"] == "MBH" for n in (a, b)
            ) else "rrl"
        lag_group = f"lag:{pair[0]}:{pair[1]}" if lag else None
        identity = lag_group or json.dumps([*pair, str(key)])
        # One aggregate cost per LAG; member rows repeat the same value.
        cost = 0 if score(f"ready:{identity}") % 5 == 0 else (
            1 + score(f"cost:{identity}") % 100
        )
        include = hint.startswith("L2") and attrs.get("link_status") == "active"
        links.append({
            "source": a, "target": b, "key": key, "include": include,
            "kind": kind, "cost": cost, "lag_group": lag_group,
            "source_hint": hint,
        })

    config = {
        "graphml": "sample.graphml",
        "parameters": {
            "delays_us": {"MBH": 20, "RRN": 20, "optical": 50, "rrl": 150, "local": 0},
            "gm_capacity": 500, "minimum_gms": 2,
            "max_path_delay_us": 1500, "optical_rule": "path",
        },
        "stations": [
            {"id": f"bs_{node}", "node": node, "weight": 1 + score(f"weight:{node}") % 10}
            for node in sorted(graph)
        ],
        "links": links,
        "synthetic_metadata": {
            "synthetic": True, "seed": SEED,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "station_model": "One assumed BS per MBH/RRN node; not actual BS inventory",
            "weights": "Deterministic synthetic integers 1..10",
            "costs": "Synthetic units; hash modulo 5 selects ready links; other costs 1..100",
            "include_rule": "Only active records marked L2; L3 excluded explicitly",
            "lag_rule": "One LAG per endpoint pair, requiring exactly one non-member LAG record",
            "media_rule": "FO means optical; otherwise inter-site MBH-MBH assumed optical, other inter-site links assumed RRL; local links without FO assumed copper",
            "local_delay_us": "0, explicit assumption matching solve.pdf",
            "warning": "Results describe synthetic parameters on supplied topology, not measured operator data",
        },
    }
    output = directory / "sample.synthetic.json"
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Separate, explicitly labelled subset for experimenting with a feasible component.
    l2 = nx.Graph()
    l2.add_nodes_from(graph)
    l2.add_edges_from((e["source"], e["target"]) for e in links if e["include"])
    main_nodes = max(nx.connected_components(l2), key=len)
    nx.write_graphml(graph.subgraph(main_nodes).copy(), directory / "sample.main.graphml")
    subset = {
        **config,
        "graphml": "sample.main.graphml",
        "stations": [s for s in config["stations"] if s["node"] in main_nodes],
        "links": [e for e in links if e["source"] in main_nodes and e["target"] in main_nodes],
        "synthetic_metadata": {**config["synthetic_metadata"],
                               "scope": "Largest connected component of included L2 edges only; other components explicitly excluded"},
    }
    (directory / "sample.main.synthetic.json").write_text(
        json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "stations": len(config["stations"]),
                      "source_edges": len(links), "included_records": sum(e["include"] for e in links),
                      "lag_groups": len(lag_records),
                      "weight_distribution": dict(sorted(Counter(s["weight"] for s in config["stations"]).items()))}, indent=2))


if __name__ == "__main__":
    main()
