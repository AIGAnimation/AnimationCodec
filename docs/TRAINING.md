# Training the entropy model

The learned model only decides how many bits each residual costs. It never changes the decoded motion. Training data
is therefore not motion directly: it is the codec's own residual streams. Dump them once, pack them into a pool, then
train. The released model (`curvecodec/data/model_float.pt`, exported to `curvecodec/data/model_int.npz`) was trained
with this recipe on the train side of split v2, about 900 h of motion, in about 1.5 h on one RTX 4090 (40 M windows).
Its training streams came from an earlier version of the lossy stage; the features and the model are the same.

Requirements: `pip install -e ".[train]"` (or `pip install -r requirements-train.txt`). Training needs PyTorch with a
CUDA GPU; dumping and pooling run on the CPU.

## 1. Dump the residual streams

```bash
python -m curvecodec.train.dump data/bvh --out dumps --jobs 16          # precisions 0.01, 0.05, 0.1, 0.3, 1 by default
```

The command encodes every BVH under `data/bvh` at each precision, exactly as `curvecodec encode` does: ACL reference,
mean contract and fallback margins. For every coded curve it keeps what the model's features are computed from: the
residuals, the decoded increments, the key gaps, the curve's static fields, and the bits the codec spent on each
residual. Output is one file per clip and precision, `dumps/<group>__<clip>__p<p>.npz`.

The **group** is the unit of the training budget, usually a corpus family. By default it is the first folder below
`data/bvh`; `--group NAME` sets it explicitly. Use `--scale` for rigs whose units the automatic rule gets wrong.
To reproduce our split, select the train side with `python -m curvecodec.split --side train`.

## 2. Build a pool

```bash
python -m curvecodec.train.pool build dumps --out pool                  # several folders, or --files-from LIST
python -m curvecodec.train.pool build dev_dumps --out dev_pool          # optional: a held-out dev pool
```

A pool is one memory-mapped image of many dumps. All training processes on a machine share one page-cached copy.
A training window has 128 positions of context and 384 scored positions. Its 36 features are computed on demand, in C,
by running the curve's feature recursion from its start.

## 3. Train

```bash
python -m curvecodec.train.train --pool pool --dev-pool dev_pool --out run \
       --layers 2 --hidden 64 --heads 4 --window 128 --budget sqrt --total-windows 40000000
```

These are the defaults and the released model's recipe:

- **Model:** causal transformer with 2 pre-norm blocks of width 64, 4 heads, a 128-position attention window, and a
  3-logistic mixture head.
- **Optimiser:** batch 128, AdamW with weight decay 1e-4, one-cycle learning rate to 2e-3 (10 % warm-up, cosine),
  gradient clip 5, TF32 matmuls.
- **Features:** normalised with the released statistics (`curvecodec/data/feature_norm.npz`).
- **Budget:** `--budget sqrt` gives each group a share of the scored symbols proportional to the square root of its
  size. `+group=0.04` raises one group's share to at least 4 %; the released model used `sqrt+wild=0.04` for game
  assets. `natural`, `equal` and `pow:A` are also available.

Outputs:

- `run/final.pt`: the float checkpoint.
- `run/hist.json`: train and dev bits per symbol.
- `run/model_int.npz`: the fixed-point export.

For a quick check, use `--max-steps 4000` on a small pool. Ours went from 18.4 to 5.9 bits per symbol on dev in
4,000 steps; the released model scores 5.8 on the same pool.

Lessons from our capacity sweep:

- Larger models (4–8 layers, width 64–256) save a few percent more bytes, at 2–27× the decode time.
- Deeper models need TF32 rather than bf16.
- At 4 layers or more, a one-pass schedule needs `--lr 5e-4`.
- Beyond about 80 M windows, more data barely helps.

## 4. Export and use

```bash
python -m curvecodec.train.export run/final.pt run/model_int.npz        # float -> fixed point (train does this too)
python -m curvecodec.train.check examples/cmu_27_07.bvh --float run/final.pt
```

The export quantizes to int16 weights with power-of-two scales and int64 accumulation. It also builds the
lookup-table activations. The result is bit-exact on every platform. Exporting the released `model_float.pt`
reproduces `model_int.npz` byte for byte.

`check` verifies on real clips that the training path matches the codec:

- the pool's features equal the encoder's own features, row for row;
- the integer model's code length on the pool equals the bits the codec spends;
- with `--float`, it also reports the float model's loss on the same windows.

To encode with your model, load it with `CurveCodec(model_path="run/model_int.npz")`. The blob does not record which
model produced it, so a stream must be decoded with the same model file it was encoded with.
