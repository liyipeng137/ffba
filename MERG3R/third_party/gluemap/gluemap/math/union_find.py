from collections.abc import Hashable


class UnionFind:
    """Disjoint-set / union-find with path compression.

    Elements may be any hashable value; sets are identified by their
    representative element.
    """

    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}

    def find(self, x: Hashable) -> Hashable:
        """Return the representative of ``x``.

        Inserts ``x`` as its own representative if it has not been seen
        before, and applies path compression on the traversed chain.
        """
        # Find the root in the first pass
        root = x
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[root] != root:
            root = self.parent[root]

        # Path compression in the second pass
        curr = x
        while self.parent[curr] != root:
            next_node = self.parent[curr]
            self.parent[curr] = root
            curr = next_node

        return root

    def union(self, x: Hashable, y: Hashable) -> None:
        """Merge the sets containing ``x`` and ``y``."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y

    def clear(self) -> None:
        """Remove all elements, resetting to an empty structure."""
        self.parent.clear()
