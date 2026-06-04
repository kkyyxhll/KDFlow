# Installation

KDFlow can be installed either **from source** or run via a **prebuilt Docker image**.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.4 (`bf16`-capable GPU strongly recommended)
- CUDA-capable GPUs (the built-in examples assume 8× GPUs per node)
- [`sglang`](https://github.com/sgl-project/sglang) ≥ 0.5.10 for source installs
- [`ray`](https://docs.ray.io/) ≥ 2.0 (used for the Teacher / Rollout / Student actor groups)

The full Python dependency list is declared in
[`requirements.txt`](https://github.com/songmzhang/KDFlow/blob/main/requirements.txt).

## Install from source

```bash
git clone https://github.com/songmzhang/KDFlow.git
cd KDFlow
pip install -e ./

# install flash attention AFTER torch is installed
pip install flash_attn==2.8.3 --no-build-isolation
```

This installs the `kdflow` package (a *namespace* package; no console script — entry
points are run via `python -m kdflow.cli.<entry>`, see
[CLI Entry Points](../reference/cli.md)).

### Optional extras

The project exposes two extras in `pyproject.toml`:

```bash
# LoRA fine-tuning
pip install -e ".[lora]"

# Everything: peft + wandb + triton + flash_attn + ring_flash_attn
pip install -e ".[all]"
```

!!! note "Qwen3.5 / latest VLM support"
    To use **Qwen3.5** and similar models, install a recent SGLang that supports
    `transformers ≥ 5.3.0`.  
    `sglang==0.5.9` has a known critical VLM bug
    ([sglang#19335](https://github.com/sgl-project/sglang/issues/19335)) — for source
    installs please pin `sglang>=0.5.10`.

## Use the prebuilt Docker image

The simplest way to get a working environment:

```bash
docker pull songmzhang/kdflow:sgl059-torch291-cu128
```

Or build from the provided Dockerfile:

```bash
docker build -f docker/Dockerfile.sgl059.torch291.cu128 \
             -t kdflow:sgl059-torch291-cu128 .
```

Other Dockerfiles (different CUDA / torch / SGLang combinations) are available
under [`docker/`](https://github.com/songmzhang/KDFlow/tree/main/docker) and on
[Docker Hub](https://hub.docker.com/repository/docker/songmzhang/kdflow/tags).

## Verify the installation

```bash
python -c "import kdflow, kdflow.algorithms, kdflow.loss; \
           print('algorithms:', list(kdflow.algorithms.ALGO_DICT)); \
           print('losses    :', list(kdflow.loss.LOSS_DICT))"
```

You should see the four built-in algorithms (`sft`, `vanilla_kd`, `simple_ctkd`,
`dskd`) and the nine registered losses (`kl`, `rkl`, `jsd`, `akl`, `skl`, `srkl`,
`tvd`, `hrl`, `top1_ce`).

## Next steps

- Run your first job: [Quickstart](quickstart.md)
- Browse the example scripts under
  [`examples/`](https://github.com/songmzhang/KDFlow/tree/main/examples).
