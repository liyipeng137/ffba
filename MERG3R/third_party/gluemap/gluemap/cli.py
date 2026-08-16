import logging
import os
import re

from gluemap.controllers.gluemap_impl import run_inference_pipeline
from gluemap.controllers.image_retrieval import (
    run_preprocessing_pipeline,
    run_preprocessing_pipeline_multi,
)
from gluemap.datasets.multi_sequence_twoview import MultiSequencePairs
from gluemap.datasets.sequential_twoview import SequentialTwoViewDataset
from gluemap.datasets.twoview import BaseTwoViewDataset
from gluemap.utils.cli import get_args_parser, parse_args_with_config
from gluemap.utils.gpu import init_distributed

logger = logging.getLogger(__name__)


def demo_main():
    parser = get_args_parser()
    args = parse_args_with_config(parser)

    rank, world_size, device, dtype = init_distributed(args)

    args.curr_processed = args.write_path
    args.curr_path = args.write_path

    if args.is_multi_sequence:
        # Multi-sequence inputs are sequential within each sequence.
        args.is_sequential = True

        subfolder_pattern = (
            re.compile(args.subfolder_regex) if args.subfolder_regex else None
        )
        datasets = [
            x
            for x in sorted(os.listdir(args.images_path))
            if os.path.isdir(os.path.join(args.images_path, x))
            and (subfolder_pattern is None or subfolder_pattern.match(x))
        ]
        logger.info(
            f"Multi-sequence mode: found {len(datasets)} subfolders "
            f"(regex='{args.subfolder_regex}')"
        )

        run_preprocessing_pipeline_multi(args, world_size, rank, datasets)
        dataset_pair = MultiSequencePairs(args, datasets)
    else:
        (_, _), _ = run_preprocessing_pipeline(args, world_size, rank)

        if getattr(args, "is_sequential", False):
            dataset_pair = SequentialTwoViewDataset(args)
        else:
            dataset_pair = BaseTwoViewDataset(args)

    # Inference pipeline: twoview -> star -> global mapping -> refinement
    run_inference_pipeline(args, dataset_pair, world_size, rank, device, dtype)
