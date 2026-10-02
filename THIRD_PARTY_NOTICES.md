# Third-party notices

**ACL – Animation Compression Library**, **RTM – Realtime Math**, **sjson-cpp** by Nicholas Frechette, MIT License.
Not vendored: `scripts/fetch_acl.sh` clones them into `third_party/acl` (ACL develop commit `3ee5685`, rtm `113ca0d`,
sjson-cpp `54b3985`). `cpp/acl_profile.cpp` and `cpp/acl_bench.cpp` are our tools,
built against ACL's public headers.

**Example clip** `examples/cmu_27_07.bvh` (and the test clips in `tests/golden/`, CMU 27_07 and 102_14), from the CMU Graphics Lab Motion Capture Database (http://mocap.cs.cmu.edu),
"created with funding from NSF EIA-0196217". The database may be used freely, including commercially.

**Model weights** in `curvecodec/data/` (`model_float.pt`, `model_int.npz`) were trained on the datasets listed in `docs/DATA.md`. Some of those
datasets are licensed for non-commercial research only. Check that the terms of the training data fit your use before
you use the weights in a product.
