# Data and the split

## The corpus

The paper uses 32 datasets: ~190,000 clips, ~900 h and 16.2 G joint samples. All of them are BVH, or were converted to
BVH. We do not redistribute any motion data. Get each dataset from its source under its own license. Many of them are
non-commercial and several forbid redistribution.

| `dataset` | name in the paper | clips | hours | test clips | source |
|---|---|---|---|---|---|
| `fc_hiphi` | HiPHI | 17,626 | 308.7 | 367 | [noitomrobotics/HiPHI](https://huggingface.co/datasets/noitomrobotics/HiPHI) (gated) |
| `fc_bones_seed` | BONES-SEED | 71,132 | 144.2 | 1,437 | [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) (originals, mirrors not used) |
| `fc_motionx` | Motion-X | 52,464 | 86.7 | 1,060 | [Motion-X](https://github.com/IDEA-Research/Motion-X) `smplx_322`, converted to BVH (SMPL-X 55 joints) |
| `fc_beat` | BEAT | 1,945 | 62.6 | 39 | [BEAT](https://github.com/PantoMatrix/BEAT) English v0.2.1 |
| `fc_motionpersona` | MotionPersona | 2,993 | 39.4 | 60 | [myshi/MotionPersona](https://huggingface.co/datasets/myshi/MotionPersona) |
| `fc_amass` | AMASS | 13,421 | 27.5 | 268 | [AMASS](https://amass.is.tue.mpg.de) as BVH (the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) crops, 20 fps) |
| `fc_animationgpt` | AnimationGPT | 14,762 | 26.4 | 295 | [AnimationGPT](https://github.com/fyyakaxyy/AnimationGPT) `CombatMotionRaw` (HumanML3D format), converted |
| `fc_for_elise` | for_elise (hands) | 306 | 23.7 | 6 | [rcwang/for_elise](https://huggingface.co/datasets/rcwang/for_elise) |
| `fc_100style` | 100STYLE | 810 | 22.1 | 16 | [100STYLE](https://www.ianxmason.com/100style/) |
| `fc_geno_100style` | Geno 100STYLE | 810 | 22.1 | 16 | [Geno retargets](https://theorangeduck.com/media/uploads/Geno/100style-retarget/bvh.zip) (D. Holden) |
| `fc_interact` | InterAct | 1,288 | 12.5 | 28 | [leohocs/interact](https://huggingface.co/datasets/leohocs/interact), 30 fps raw BVH |
| `fc_interact_r65` | InterAct 65-joint | 1,286 | 12.5 | 28 | [leohocs/interact](https://huggingface.co/datasets/leohocs/interact), 65-joint version |
| `fc_geno_interact_single` | Geno InterAct single | 433 | 9.3 | 10 | [Geno retargets](https://theorangeduck.com/media/uploads/Geno/interact-retarget/bvh_single.zip) |
| `fc_geno_interact_multi` | Geno InterAct multi | 432 | 9.3 | 10 | [Geno retargets](https://theorangeduck.com/media/uploads/Geno/interact-retarget/bvh_multi.zip), `bvh_multi` |
| `fc_cmu` | CMU | 2,180 | 9.1 | 63 | [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu) |
| `fc_geno_motorica` | Geno Motorica | 121 | 6.2 | 2 | [Geno retargets](https://theorangeduck.com/media/uploads/Geno/motorica-retarget/bvh.zip) of the Motorica dance data |
| `fc_lafan1` | LAFAN1 | 77 | 4.6 | 5 | [Ubisoft La Forge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) |
| `fc_geno_lafan1` | Geno LAFAN1 | 77 | 4.5 | 5 | [Geno re-solve](https://theorangeduck.com/media/uploads/Geno/lafan1-resolved/bvh.zip) |
| `fc_bandai` | Bandai | 3,077 | 3.9 | 62 | [Bandai-Namco-Research-Motiondataset](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset) |
| `fc_kid` | kid | 956 | 3.0 | 19 | from *Adult2child: Motion Style Transfer using CycleGANs* |
| `fc_multi_subject` | multi-subject | 269 | 2.3 | 5 | from *A Causal Convolutional Neural Network for Multi-Subject Motion Modeling and Generation* |
| `fc_zeroeggs` | ZeroEGGS | 67 | 2.2 | 1 | [ZeroEGGS](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS), original rig, 30 fps |
| `fc_zeroeggs_r65` | ZeroEGGS 65-joint | 67 | 2.2 | 1 | [ZeroEGGS](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS), 65-joint version |
| `fc_geno_zeroeggs` | Geno ZeroEGGS | 67 | 2.2 | 1 | [Geno retargets](https://theorangeduck.com/media/uploads/Geno/zeroeggs-retarget/bvh.zip) |
| `fc_bfa` | BFA | 64 | 2.2 | 5 | from *Unpaired Motion Style Transfer from Video to Animation* ([download](https://drive.google.com/drive/folders/1C-_iZJj-PSUWZwh25yAsQe1tLpPm9EZ5)) |
| `wild` | wild game assets | 3,488 | 2.1 | 549 | Truebones animation packs |
| `fc_humanact12` | HumanAct12 | 1,191 | 1.2 | 28 | the HumanAct12 part of [HumanML3D](https://github.com/EricGuo5513/HumanML3D), converted |
| `fc_pfnn` | PFNN | 40 | 1.1 | 8 | [PFNN data](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control) |
| `fc_dog` | dog (held out) | 52 | 0.7 | 52 | MANN quadruped ([AI4Animation](https://github.com/sebastianstarke/AI4Animation)), all test |
| `fc_edin` | Edinburgh | 47 | 0.2 | 5 | Edinburgh locomotion, from the PFNN paper ([pfnn.zip](http://theorangeduck.com/media/uploads/other_stuff/pfnn.zip)) |
| `fc_xia` | Xia | 572 | 0.2 | 11 | from *Realtime Style Transfer for Unlabeled Heterogeneous Human Motion* |
| `fc_mixamo` | Mixamo | 35 | 0.1 | 5 | [Mixamo](https://www.mixamo.com) |

KIT-ML (`fc_kit_ml`, 10.9 h) was converted and ingested but is excluded from both sides: its overlap with AMASS-KIT and
CMU could not be verified. In total, 389 byte duplicates and 99 files with NaN values are also excluded.

**Converted datasets.** Motion-X is written as SMPL-X 55-joint BVH, with a per-clip skeleton from the clip's mean shape;
the converted BVH matches the reference `smplx` model within 1.2e-3 cm. AnimationGPT and HumanAct12 use HumanML3D's 22
joints plus 5 zero-offset branch joints. KIT-ML uses its 21 joints. The conversion code is not part of this release.

**Header fixes**: `fc_amass` frame time 0.05 (the files claim 30 fps, the data is 20 fps); 9 BEAT takes
with a 24 fps header and 120 fps data; 249 BEAT takes whose `Frames:` count exceeds the motion lines. Units:
`fc_for_elise` and `fc_bones_seed` are physical centimetres (encode them with `--scale 1`). Every other
dataset uses `acl_profile`'s automatic rule.

## Split v2

`curvecodec/data/split_v2.json.gz`, read by `curvecodec/split.py` (`python -m curvecodec.split --side test`):

- per dataset, 2 of every 100 clips go to test, sampled deterministically (seed 0);
- the unit is the **take group**, never the clip: all clips of one take (retargets, crops, windows, the other actors of
  a multi-actor take) land on the same side;
- the dog (`fc_dog`) is held out entirely as the cross-species transfer test;
- the sources of our development and benchmark clips are forced to test.

The test side has about 4,500 clips (about 20 h); everything else is training data.