import argparse
import logging
from typing import ClassVar

import numpy as np
import torch
from scipy.special import softmax

from gluemap.controllers.base_inference import BaseInferencePipeline
from gluemap.datasets.twoview import BaseTwoViewDataset
from gluemap.utils.model_loader import load_models

logger = logging.getLogger(__name__)


class BatchInferenceDG:
    """Per-batch Doppelgangers two-view classifier.

    Forwards a batched pair of images through the DG model and returns a
    per-pair confidence score by voting across the two softmax outputs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype

    def main(self, batch: dict) -> dict[str, torch.Tensor]:
        """Run the DG model on a single batch.

        Args:
            batch: DataLoader batch; must contain ``"images"`` of shape
                ``(B, 2, ...)`` with the two views stacked along dim 1.

        Returns:
            ``{"scores": Tensor(B,)}`` — per-pair confidence in ``[0, 1]``.
        """
        images = batch["images"].to(self.device)

        view1 = {
            "img": images[:, 0],
            "instance": [i for i in range(images.shape[0])],
        }
        view2 = {
            "img": images[:, 1],
            "instance": [i for i in range(images.shape[0])],
        }

        res1, res2, pred1, pred2 = self.model(view1, view2)

        if isinstance(pred1, list):
            pred1 = torch.stack(pred1, dim=0)

        if isinstance(pred2, list):
            pred2 = torch.stack(pred2, dim=0)

        score_s1 = softmax(pred1.detach().cpu().numpy(), axis=1)
        score_s2 = softmax(pred2.detach().cpu().numpy(), axis=1)
        vote_0 = (score_s1[:, 0] > score_s1[:, 1]).astype(int) + (
            score_s2[:, 0] > score_s2[:, 1]
        ).astype(int)
        vote_1 = (score_s1[:, 1] > score_s1[:, 0]).astype(int) + (
            score_s2[:, 1] > score_s2[:, 0]
        ).astype(int)
        index_max = vote_1 > vote_0
        index_min = vote_1 < vote_0
        index_equal = vote_1 == vote_0
        score = np.zeros_like(score_s1[:, 0])
        score[index_max] = np.max(
            (score_s1[index_max, 1], score_s2[index_max, 1]), axis=0
        )
        score[index_min] = np.min(
            (score_s1[index_min, 1], score_s2[index_min, 1]), axis=0
        )
        score[index_equal] = np.mean(
            (score_s1[index_equal, 1], score_s2[index_equal, 1]), axis=0
        )

        result_dict = {
            "scores": torch.from_numpy(score),
        }

        return result_dict


class TwoViewInferencePipeline(BaseInferencePipeline):
    """Pipeline object for two-view (Doppelgangers) inference."""

    _index_key: ClassVar[str] = "pair_indexes"
    _rerun_from_triggers: ClassVar[frozenset[str] | None] = frozenset(
        {"retrieval", "twoview"}
    )
    _profiling_label: ClassVar[str] = "Two-view inference"

    def _load_models(self) -> dict[str, torch.nn.Module]:
        if self.models is not None:
            return self.models
        if self.preloaded_models is not None and "dg" in self.preloaded_models:
            self.models = {"dg": self.preloaded_models["dg"]}
            self.device = next(self.models["dg"].parameters()).device
        else:
            loaded, self.device = load_models(self.args, keys={"dg"})
            self.models = {"dg": loaded["dg"]}
            self._owns_models = True
        return self.models

    def _create_batch_inference(
        self, models: dict[str, torch.nn.Module]
    ) -> BatchInferenceDG:
        return BatchInferenceDG(
            models["dg"], device=self.device, dtype=self.dtype
        )

    def _run_batch_step(
        self, batch_inference: BatchInferenceDG, batch: dict
    ) -> tuple[dict, dict[str, float]]:
        return batch_inference.main(batch), {}

    def _pack_local_outputs(
        self, all_outputs: list[dict], all_indices: list[int]
    ) -> dict:
        output_keys = list(all_outputs[0].keys())
        return {
            "scores": torch.cat(
                [output["scores"] for output in all_outputs], dim=0
            ).contiguous(),
        } | {
            key: [
                output[key][i]
                for output in all_outputs
                for i in range(len(output[key]))
            ]
            for key in output_keys
            if key != "scores"
        }

    def _merge_gathered_outputs(
        self,
        data_list: list[dict],
        index_mapping: np.ndarray,
        dataset_size: int,
    ) -> dict:
        output_keys = [k for k in data_list[0] if k != self._index_key]
        global_outputs: dict = {}
        for key in output_keys:
            if key == "scores":
                gathered = torch.cat(
                    [out[key] for out in data_list], dim=0
                ).contiguous()
                global_outputs[key] = gathered[index_mapping]
            else:
                gathered_list = [
                    out[key][x]
                    for out in data_list
                    for x in range(len(out[key]))
                ]
                global_outputs[key] = [
                    gathered_list[index_mapping[i]]
                    for i in range(len(index_mapping))
                ]
        return global_outputs

    def _postprocess_global_outputs(
        self, global_outputs: dict, dataset: BaseTwoViewDataset
    ) -> dict:
        global_outputs["pairs"] = dataset.pairs
        return global_outputs


def run_twoview_inference(
    args: argparse.Namespace,
    dataset_pair: BaseTwoViewDataset,
    world_size: int,
    rank: int,
    file_name: str = "twoview_result.pth",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    preloaded_models: dict[str, torch.nn.Module] | None = None,
):
    """Module-level wrapper to instantiate TwoViewInferencePipeline and run."""
    pipeline = TwoViewInferencePipeline(
        args,
        world_size,
        rank,
        file_name=file_name,
        device=device,
        dtype=dtype,
        preloaded_models=preloaded_models,
    )
    return pipeline.run(dataset_pair)
