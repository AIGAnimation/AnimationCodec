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

**Media** in `media/` and in the online demo. The animals in `any_skeleton.webp` and the dragon in `hero_dragon.webp` are
Truebones Zoo characters (truebones.com, free motion-capture animal pack); the humans are the Geno character of the ZeroEGGS
dataset (Ubisoft La Forge, ZeroEGGS licence); the robot is a Unitree Go2 model (Unitree Robotics). The online demo
(https://playground.rubbly.cn/codec/) additionally shows "Wolf rigged low poly" by 3DHaupt
(https://sketchfab.com/models/f3769a474a714ebbbaca0d97f9b0a5a0) and the following Sketchfab models, all CC BY 4.0:

* “Animated Hovering Flying Hummingbird Loop” by LasquetiSpice, CC BY 4.0, https://sketchfab.com/3d-models/102884713a2742ce829e368d2a790c45
* “sparrow_upload” by faiyaz5yaz, CC BY 4.0, https://sketchfab.com/3d-models/144367de23534c28ad2e83fc8abbd9ea
* “Night Sky Down Under” by Miguelangelo Rosario, CC BY 4.0, https://sketchfab.com/3d-models/64fc160875ad4d71ae1cc5423cea8222
* “high poly bee modle” by shreebaghel72, CC BY 4.0, https://sketchfab.com/3d-models/bc6564c1221241eb9a06cc32526fb1b7
* “Mantis Twitch Walk” by JeffFleetwood, CC BY 4.0, https://sketchfab.com/3d-models/654a1d116efb40dc8b59ad635961287c
* “Animated Repticect” by DoubelFace, CC BY 4.0, https://sketchfab.com/3d-models/7a087f7cb4fc4019b0bfc547773c5ccd
* “Caterpillar Crawl” by michael l., CC BY 4.0, https://sketchfab.com/3d-models/54e7e37365ac4063ba2ba65151fd9f5d
* “Pangxie” by wsrttys, CC BY 4.0, https://sketchfab.com/3d-models/0846167f9302450cba1926c6fc5d6d42
* “Kraken v2” by lawtrigg, CC BY 4.0, https://sketchfab.com/3d-models/4691fa1b881b4d43936b706f18dba169
* “Tortuga verde (Chelonia mydas)” by Innoceana, CC BY 4.0, https://sketchfab.com/3d-models/1cb59946558d41ccbe1719c13238afd9
* “Fish Swimming” by geniusrahman155, CC BY 4.0, https://sketchfab.com/3d-models/657a6aeb04a64a90b9a2f3089d25422e
* “Cute Fish” by RickStikkelorum, CC BY 4.0, https://sketchfab.com/3d-models/43ead4b42ec64bff8dbdf81e12150598
* “Whale666” by a0976623059, CC BY 4.0, https://sketchfab.com/3d-models/32b24ee2dcc94671afadca8760492cef
* “Animated Elephant Character” by bgilgen, CC BY 4.0, https://sketchfab.com/3d-models/5db0b1b064f44b1fb9806f329d2295fd
* “Snowman” by Horizon Studio, CC BY 4.0, https://sketchfab.com/3d-models/f214d11ccd11444fa0d1e32ed9602c51
* “American Bison” by Damco, CC BY 4.0, https://sketchfab.com/3d-models/88fbc7cb875444669bd45ac3597bc579
* “Cow NPC - Now free to download” by Owlish Media, CC BY 4.0, https://sketchfab.com/3d-models/2ca1db4e890e4b24a68624597e7d2fc8
* “Low Poly wolf” by manoeldarochadeoliveira, CC BY 4.0, https://sketchfab.com/3d-models/c6baa6970e724c6a97d5e1ecb8823179
* “Fennec Fox Free” by Evil_Katz, CC BY 4.0, https://sketchfab.com/3d-models/68a1c810abee435a8207410b31c0456c
* “Roaring Stag ( deepdreamed )” by Miguelangelo Rosario, CC BY 4.0, https://sketchfab.com/3d-models/4798d8c87a0e4ad8835217fe93ddf67b
* “Don't overlook the Hippo” by Miguelangelo Rosario, CC BY 4.0, https://sketchfab.com/3d-models/f8ecf3f985d84be7b1cb5501f45bc688
* “Triceratops occultatum” by Miguelangelo Rosario, CC BY 4.0, https://sketchfab.com/3d-models/d8b6a381f36c46f8b1d59ed6e0b57c65
* “Cute Sci-Fi Dragon” by hare_ware, CC BY 4.0, https://sketchfab.com/3d-models/135d9d148a8f4b388b5c915f42b1abec
* “Flint Maw” by Spinnee, CC BY 4.0, https://sketchfab.com/3d-models/f301bd397cd6492b9df719313ecc51fe
* “Fire Elemental” by InaLaAtzu, CC BY 4.0, https://sketchfab.com/3d-models/05fe96ef7fca472ba0bf753686216002
* “Monster Plant Enemy” by Jacqueline Sweeney, CC BY 4.0, https://sketchfab.com/3d-models/02f912bbb5ea42958c30ecbf1cb949a6
* “Cactus1” by nathan.connell, CC BY 4.0, https://sketchfab.com/3d-models/16e379f271414c2cbf5a8806083c5b79
* “SkeletonBoss” by kennethcplace, CC BY 4.0, https://sketchfab.com/3d-models/0c668ecb7c384fc8a137bc7160164649
* “Robot Dinosaur Walking - First Mechanics Test” by Instinto Ideal Studio, CC BY 4.0, https://sketchfab.com/3d-models/03954e5eefbe4afd94ccf06ceaacf9ab
* “Ezaroid - The Chopping Killer Machine” by evilinvader, CC BY 4.0, https://sketchfab.com/3d-models/9035ed4c5cd44dd8aa70ed3b243a1791
* “Low poly mech walking” by WarlockStones, CC BY 4.0, https://sketchfab.com/3d-models/25ac94b7192d401f8ce7b7c072877502
* “Military Drone Low-Poly” by ToporEnterprise, CC BY 4.0, https://sketchfab.com/3d-models/24456f3cf17f4a12b05fbe9d95c3687d
* “Robot Error” by wamala, CC BY 4.0, https://sketchfab.com/3d-models/7351b854ca0d442aa31d68a551008e94
