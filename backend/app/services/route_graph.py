"""Build a reaction route graph from linear steps (P1)."""

from __future__ import annotations

import re

from app.schemas.reaction_route import RouteGraph, RouteGraphEdge, RouteGraphNode, SynthesisRoute
from app.services.chemical_synonyms import extract_compound_id


def _node_key(name: str) -> str:
    base, cid = extract_compound_id(name or "")
    slug = re.sub(r"[^a-z0-9]+", "_", (base or name or "unknown").lower()).strip("_")
    if cid:
        return f"{slug}_{cid.lower()}"
    return slug or "unknown"


def build_route_graph(route: SynthesisRoute) -> RouteGraph:
    """Connect step products to downstream reactants as a directed graph."""
    nodes_by_key: dict[str, RouteGraphNode] = {}
    edges: list[RouteGraphEdge] = []

    def touch(name: str, *, step_index: int, role: str) -> str:
        key = _node_key(name)
        _, cid = extract_compound_id(name)
        if key not in nodes_by_key:
            nodes_by_key[key] = RouteGraphNode(
                node_id=key,
                name=name,
                canonical_name=extract_compound_id(name)[0],
                compound_ids=[cid] if cid else [],
                step_indices=[step_index],
                role=role,
            )
        else:
            node = nodes_by_key[key]
            if step_index not in node.step_indices:
                node.step_indices.append(step_index)
            if cid and cid not in node.compound_ids:
                node.compound_ids.append(cid)
        return key

    for step in route.steps:
        si = step.step_index
        product_keys = [touch(p.name, step_index=si, role="product") for p in step.products if p.name]
        reactant_keys = [touch(r.name, step_index=si, role="reactant") for r in step.reactants if r.name]
        for rk in reactant_keys:
            for pk in product_keys:
                edges.append(
                    RouteGraphEdge(
                        from_node=rk,
                        to_node=pk,
                        step_index=si,
                        relation="transforms_to",
                    )
                )
        # Chain: prior step products → this step reactants
        for inter in step.intermediates:
            touch(inter.name, step_index=si, role="intermediate")

    # Sequential chain edges when names align across steps
    for i in range(len(route.steps) - 1):
        cur = route.steps[i]
        nxt = route.steps[i + 1]
        for prod in cur.products:
            pk = _node_key(prod.name)
            for react in nxt.reactants:
                rk = _node_key(react.name)
                if pk == rk or pk in rk or rk in pk:
                    if not any(e.from_node == pk and e.to_node == rk and e.step_index == nxt.step_index for e in edges):
                        edges.append(
                            RouteGraphEdge(
                                from_node=pk,
                                to_node=rk,
                                step_index=nxt.step_index,
                                relation="intermediate_flow",
                            )
                        )

    target_key = ""
    if route.target_name:
        target_key = _node_key(route.target_name)
        if target_key not in nodes_by_key:
            touch(route.target_name, step_index=route.steps[-1].step_index if route.steps else 0, role="target")

    return RouteGraph(
        nodes=list(nodes_by_key.values()),
        edges=edges,
        target_node_id=target_key,
    )
