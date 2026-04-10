"""FlowGraph — directed acyclic graph for dynamic module connections."""

from __future__ import annotations

from collections import defaultdict, deque


class FlowGraph:
    """Parse and execute a flow specification.

    Flow spec format::

        "encoder_a -> encoder_b"        # encoder_b receives encoder_a's output
        "encoder_a -> actor"            # actor receives encoder_a's output
        "encoder_a -> actor"
        "encoder_b -> actor"            # actor receives concat of a and b

    Rules
    -----
    * Each node may have multiple upstream nodes; their latents are concatenated
      before being passed to that node.
    * A node with no upstream nodes in the graph receives the raw observation
      that matches its ``input_key``.
    * Cycles are forbidden (raises ``ValueError``).
    """

    def __init__(self, flow_specs: list[str], node_names: list[str]) -> None:
        self.nodes: list[str] = list(node_names)
        # edges[dst] = list of src nodes
        self.edges: dict[str, list[str]] = defaultdict(list)
        self._parse(flow_specs)
        self._order: list[str] = self._topological_sort()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, specs: list[str]) -> None:
        for spec in specs:
            spec = spec.strip()
            if not spec or spec.startswith("#"):
                continue
            if "->" not in spec:
                raise ValueError(f"Invalid flow spec (missing '->'): '{spec}'")
            parts = [p.strip() for p in spec.split("->")]
            if len(parts) != 2:
                raise ValueError(f"Flow spec must have exactly one '->': '{spec}'")
            src, dst = parts
            if src not in self.nodes:
                raise ValueError(f"Flow source '{src}' not declared in node list {self.nodes}")
            if dst not in self.nodes:
                raise ValueError(f"Flow destination '{dst}' not declared in node list {self.nodes}")
            if src not in self.edges[dst]:
                self.edges[dst].append(src)

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_sort(self) -> list[str]:
        in_degree: dict[str, int] = {n: 0 for n in self.nodes}
        # Build adjacency: for each dst, src adds an edge src→dst
        for dst, srcs in self.edges.items():
            for _ in srcs:
                in_degree[dst] = in_degree.get(dst, 0) + 1
        # Recompute cleanly
        in_degree = {n: 0 for n in self.nodes}
        for dst, srcs in self.edges.items():
            in_degree[dst] += len(srcs)

        queue: deque[str] = deque(n for n in self.nodes if in_degree[n] == 0)
        order: list[str] = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for dst, srcs in self.edges.items():
                if n in srcs:
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        queue.append(dst)

        if len(order) != len(self.nodes):
            raise ValueError(
                "FlowGraph has a cycle or disconnected nodes. "
                f"Processed {order}, declared {self.nodes}"
            )
        return order

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def execution_order(self) -> list[str]:
        return self._order

    def get_inputs(self, node: str) -> list[str]:
        """Return list of upstream nodes whose outputs feed into *node*."""
        return self.edges.get(node, [])

    def resolve_input_dim(self, node: str, latent_dims: dict[str, int]) -> int:
        """Compute the input dimension for *node* by summing upstream latents.

        Parameters
        ----------
        latent_dims:
            Map from node name → latent dimension.
        """
        inputs = self.get_inputs(node)
        if not inputs:
            # Root node — owns its own input_dim (from obs shape)
            return latent_dims.get(node, 0)
        return sum(latent_dims[src] for src in inputs)
