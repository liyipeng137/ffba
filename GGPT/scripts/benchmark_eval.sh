export HYDRA_FULL_ERROR=1

# The evaluation consists of two steps. 
# (1) Running feed-forward regression and our SfM pipeline. 

match_model='romav2-base' # or ['roma','ufm-refine']
feedforward_model='vggt-point' # or dav3, pi3x, pi3
num_gpus=1
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$num_gpus --rdzv-endpoint=localhost:2026 \
    sfm/run_benchmark_sfm.py \
    match_config.models=$match_model \
    feedforward_config.model=$feedforward_model \
    hydra.run.dir=outputs/benchmark/sfm_${feedforward_model}-${match_model}
# Intermediate results are saved in outputs/benchmark/sfm_vggt-point-romav2-base/save and will be used for GGPT.

# (2) Running GGPT. 
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$num_gpus --rdzv-endpoint=localhost:2026 \
    ggpt/run_benchmark_ggpt.py \
    hydra.run.dir=outputs/benchmark/ggpt_${feedforward_model}-${match_model}
# The final metrics will be output as outputs/benchmark/ggpt_vggt-point-romav2-base/average.json

