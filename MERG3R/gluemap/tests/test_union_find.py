"""Unit tests for gluemap.math.union_find.UnionFind."""

from gluemap.math.union_find import UnionFind


def test_initial_state_each_element_own_root():
    uf = UnionFind()
    for x in [0, 1, "a", (3, 4)]:
        assert uf.find(x) == x


def test_union_then_find_returns_common_root():
    uf = UnionFind()
    uf.union(1, 2)
    assert uf.find(1) == uf.find(2)

    uf.union(2, 3)
    root = uf.find(1)
    assert uf.find(2) == root
    assert uf.find(3) == root


def test_path_compression_flattens_chain():
    uf = UnionFind()
    uf.union(1, 2)
    uf.union(3, 2)
    uf.union(4, 3)
    uf.union(5, 4)

    root = uf.find(5)
    for x in [1, 2, 3, 4, 5]:
        assert uf.parent[x] == root, (
            f"after find(5), parent[{x}] should point directly to root {root}, "
            f"got {uf.parent[x]}"
        )


def test_idempotent_self_union():
    uf = UnionFind()
    uf.union(7, 7)
    uf.union(7, 7)
    assert uf.find(7) == 7


def test_disjoint_sets_remain_distinct():
    uf = UnionFind()
    uf.union(1, 2)
    uf.union(2, 3)

    uf.union(10, 11)
    uf.union(11, 12)

    assert uf.find(1) == uf.find(3)
    assert uf.find(10) == uf.find(12)
    assert uf.find(1) != uf.find(10)


def test_clear_resets_state():
    uf = UnionFind()
    uf.union(1, 2)
    uf.union(2, 3)
    assert uf.find(1) == uf.find(3)

    uf.clear()
    assert uf.parent == {}
    assert uf.find(1) == 1
    assert uf.find(3) == 3
