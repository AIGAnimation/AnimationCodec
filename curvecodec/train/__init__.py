"""Training the entropy model on the codec's own residual streams.

  dump    encode BVH files with the codec and keep, per coded curve, what the model's features are computed from
          (curvecodec.train.dump)
  pool    many dumps -> one memory-mapped pool; training windows with their features computed on demand (pool)
  train   the float transformer (net) on the pool's windows, per-family budget, one-cycle AdamW (train)
  export  float checkpoint -> the fixed-point model the codec loads (model.export_int_model; export)
  check   the pool's features equal the encoder's, and the model's code length on the pool equals the bits the
          codec spends (check)
"""
