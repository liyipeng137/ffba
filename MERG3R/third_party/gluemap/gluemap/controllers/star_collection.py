import argparse
import logging
from typing import Any

import networkx as nx
import numpy as np
import torch

from gluemap.datasets.star import BaseStarDataset
from gluemap.datasets.twoview import BaseTwoViewDataset

logger = logging.getLogger(__name__)


class StarCollector:
    """
    Collects two-view outputs and builds a star dataset for multi-view use.

    Builds a covisibility graph from pairwise scores and initializes a
    BaseStarDataset with valid edges and metadata from the original dataset.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(
        self,
        dataset_pair: BaseTwoViewDataset,
        global_outputs: dict,
    ) -> BaseStarDataset:
        """Generate a star dataset from two-view global outputs.

        Args:
            dataset_pair: The two-view dataset whose images/metadata are reused
                for the new star dataset.
            global_outputs: Two-view inference outputs; must contain ``"pairs"``
                (N x 2 image-index pairs) and ``"scores"`` (per-pair scalar).

        Returns:
            A ``BaseStarDataset`` initialized with the valid edges, per-edge
            scores and metadata copied from ``dataset_pair``.
        """
        args = self.args
        dataset = BaseStarDataset(args)

        scores = global_outputs["scores"].clone()
        sequential_edges = getattr(dataset_pair, "sequential_edges", [])

        # extract the graph structure from the global outputs
        valid_edges, is_connected = self._construct_covisibility_graph(
            global_outputs["pairs"],
            scores,
            len(dataset_pair.images_list),
            threshold=args.valid_dg_threshold,
        )

        pairs_np = (
            global_outputs["pairs"].numpy()
            if torch.is_tensor(global_outputs["pairs"])
            else global_outputs["pairs"]
        )
        scores_np = scores.numpy() if torch.is_tensor(scores) else scores

        if is_connected:
            images_list = dataset_pair.images_list
            images_path = dataset_pair.images_path
            images_shape_ori = dataset_pair.images_shape_ori
            images_ori = (
                dataset_pair.images_ori
                if hasattr(dataset_pair, "has_preloaded")
                else None
            )
            N = len(dataset_pair.images_list)
        else:
            sub = self._restrict_to_largest_component(
                valid_edges,
                pairs_np,
                scores_np,
                dataset_pair,
                sequential_edges,
            )
            valid_edges = sub["valid_edges"]
            pairs_np = sub["pairs_np"]
            scores_np = sub["scores_np"]
            sequential_edges = sub["sequential_edges"]
            images_list = sub["images_list"]
            images_path = sub["images_path"]
            images_shape_ori = sub["images_shape_ori"]
            images_ori = sub["images_ori"]
            N = sub["N"]

        # Build edge score lookup from all (possibly filtered/re-indexed) pairs
        edge_scores = {}
        for k in range(len(pairs_np)):
            i, j = int(pairs_np[k, 0]), int(pairs_np[k, 1])
            key = (min(i, j), max(i, j))
            edge_scores[key] = float(scores_np[k])

        # Initialize the dataset with the global outputs
        logger.info(
            f"Initializing dataset with valid edges: {valid_edges.shape}"
        )
        dataset.valid_edges = valid_edges
        dataset.edge_scores = edge_scores
        dataset.N = N
        dataset.images_list = images_list
        dataset.images_path = images_path
        dataset.query_points = [None] * len(
            valid_edges
        )  # Initialize query points for each star structure
        dataset.images_shape_ori = images_shape_ori
        dataset.force_square = dataset_pair.force_square
        dataset.sequential_edges = sequential_edges

        if images_ori is not None:
            dataset.images_ori = images_ori

        # Post initialization to set up the star structure
        dataset.__post_init__()

        logger.info("Dataset initialized done.")

        return dataset

    def _construct_covisibility_graph(
        self,
        pairs: np.ndarray | torch.Tensor,
        scores: torch.Tensor,
        N: int,
        threshold: float = 0.8,
        lower_bound: float = 0.2,
    ) -> tuple[np.ndarray, bool]:
        """Select edges with ``score > threshold`` and connect all components.

        If the resulting graph is disconnected, the threshold is repeatedly
        lowered by 0.1 and the lower-scoring cross-component edges are added in
        until either a single connected component remains or the threshold
        reaches ``lower_bound``.

        Returns:
            Tuple ``(valid_edges, is_connected)``: ``valid_edges`` is the
            ``(M, 2)`` integer array of selected edges; ``is_connected`` is
            ``True`` iff the resulting graph spans all ``N`` nodes as a single
            connected component.
        """
        # Collect the valid edges based on the scores
        scores = scores.numpy()
        valid_edges = pairs[(scores > threshold)]

        G = nx.Graph()
        G.add_nodes_from(np.arange(N))
        G.add_edges_from(valid_edges)

        # Connect the unconnected components
        components = list(nx.connected_components(G))
        if len(components) > 1:
            while threshold > lower_bound:
                # Collect the cluster index for each node
                cluster_index = np.zeros(N, dtype=int)
                for idx, component in enumerate(components):
                    cluster_index[list(component)] = idx

                index = cluster_index[pairs[:, 0]] != cluster_index[pairs[:, 1]]

                threshold -= 0.1
                # Take the edges across the components
                G = nx.Graph()
                G.add_nodes_from(np.arange(N))
                G.add_edges_from(
                    np.concatenate(
                        [valid_edges, pairs[index * (scores > threshold)]],
                        axis=0,
                    )
                )
                components = list(nx.connected_components(G))

                logger.info(
                    f"Reducing threshold to {threshold:.2f} to connect "
                    f"components"
                )
                valid_edges = np.concatenate(
                    [valid_edges, pairs[index * (scores > threshold)]], axis=0
                )

                if len(components) == 1:
                    break

        is_connected = len(components) == 1
        return valid_edges, is_connected

    def _restrict_to_largest_component(
        self,
        valid_edges: np.ndarray,
        pairs_np: np.ndarray,
        scores_np: np.ndarray,
        dataset_pair: BaseTwoViewDataset,
        sequential_edges: list[tuple[int, int]],
    ) -> dict[str, Any]:
        """Filter every per-image structure to the largest connected component.

        Builds a graph from ``valid_edges`` over ``range(N)``, picks the
        largest connected component, and remaps all per-image data
        (``valid_edges``, two-view ``pairs_np``/``scores_np``,
        ``sequential_edges``, image list/path/shape, optional preloaded images)
        from original to compact ``0..len(kept)-1`` indexing.

        ``dataset_pair`` is not mutated; the filtered lists are returned in a
        dict for the caller to attach to the star dataset.
        """
        N = len(dataset_pair.images_list)

        G = nx.Graph()
        G.add_nodes_from(np.arange(N))
        G.add_edges_from(valid_edges)

        largest_cc = max(nx.connected_components(G), key=len)
        kept = sorted(largest_cc)
        old_to_new = {old: new for new, old in enumerate(kept)}
        keep_set = largest_cc  # set membership lookup

        logger.warning(
            f"Covisibility graph disconnected; keeping largest CC "
            f"({len(kept)}/{N} images)"
        )

        # Re-index valid edges (defensively filter; edges should lie within a
        # component)
        edge_mask = np.array(
            [
                (int(a) in keep_set) and (int(b) in keep_set)
                for a, b in valid_edges
            ],
            dtype=bool,
        )
        ve = valid_edges[edge_mask]
        valid_edges_new = np.array(
            [[old_to_new[int(a)], old_to_new[int(b)]] for a, b in ve],
            dtype=valid_edges.dtype,
        ).reshape(-1, 2)

        # Re-index two-view pairs/scores (for the edge_scores rebuild)
        pair_mask = np.array(
            [
                (int(a) in keep_set) and (int(b) in keep_set)
                for a, b in pairs_np
            ],
            dtype=bool,
        )
        pairs_filt = pairs_np[pair_mask]
        pairs_np_new = np.array(
            [[old_to_new[int(a)], old_to_new[int(b)]] for a, b in pairs_filt],
            dtype=pairs_np.dtype,
        ).reshape(-1, 2)
        scores_np_new = scores_np[pair_mask]

        # Re-index sequential edges
        sequential_edges_new = [
            (old_to_new[int(a)], old_to_new[int(b)])
            for a, b in sequential_edges
            if (int(a) in keep_set) and (int(b) in keep_set)
        ]

        # Build filtered image-level lists (no mutation of dataset_pair)
        images_list = [dataset_pair.images_list[i] for i in kept]
        images_path = [dataset_pair.images_path[i] for i in kept]
        images_shape_ori = [dataset_pair.images_shape_ori[i] for i in kept]
        if hasattr(dataset_pair, "has_preloaded"):
            images_ori = [dataset_pair.images_ori[i] for i in kept]
        else:
            images_ori = None

        return {
            "valid_edges": valid_edges_new,
            "pairs_np": pairs_np_new,
            "scores_np": scores_np_new,
            "sequential_edges": sequential_edges_new,
            "images_list": images_list,
            "images_path": images_path,
            "images_shape_ori": images_shape_ori,
            "images_ori": images_ori,
            "N": len(kept),
        }


def run_star_collection(
    dataset_pair: BaseTwoViewDataset,
    global_outputs: dict,
    args: argparse.Namespace,
) -> BaseStarDataset:
    """Convenience wrapper — instantiates StarCollector and runs it."""
    return StarCollector(args).run(dataset_pair, global_outputs)
