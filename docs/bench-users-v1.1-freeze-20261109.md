# CoupledRIS-Bench generator v1.1 freeze — 2026-11-09

The user approved all 50 v1.1 previews after inspecting them in ParaView.
The frozen generator includes the native-ground rules, the approved footprint
overlay, site and scheduling rules, and the recorded recovered source snapshot.
The preview's `bs_link` stays unchanged; production derives its link classes
from the traced paths.

The [machine-readable lock](../configs/bench/users-v1.1-freeze-20261109.json)
is anchored by SHA-256 `e152e78cb7b66204ea4080339c29a80a53c94685f8c3743b4eddd0cbab6e0ecd`. Production must verify this
lock and every listed file before generation, tracing, or packaging. The
immutable generator is reused through imports by a separate case driver.

The [v1 parameters](bench-users-v1-design-20261108.md) and
[v1.1 amendment](bench-tracks-ground-v1.1-20261108.md) remain the source of
the geometry, ground, mobility and coverage decisions. Base seed
`202611081501` is the first SeedSequence entropy word. Preview streams use
`[seed,1,map_index,case_index]` for natural cases, `[seed,2,challenge_index]`
for challenges, and a separate seeded site plan (`[seed,5,map_index]`).
The production driver uses a new disjoint namespace while preserving the
same generator code. No data from either held-out city is opened.

| Frozen file | SHA-256 |
|---|---|
| `configs/bench/phase0b-area-proposals-20261108.json` | `2b96f9b35221445bc3601cb456d46f4451f1c9822fcd9d4c7c050854a5f34f85` |
| `docs/bench-tracks-ground-v1.1-20261108.md` | `393a631116d3d743b0b05f1e2c447082b75c44ffb97162df0a0edc243b3fef3d` |
| `docs/bench-users-v1-design-20261108.md` | `8a385e0a6fc03b7c14943ee1bd8dd4230661aa7be5e099bc924726c819940dfb` |
| `outputs/previews/bench-users-v1.1/case_manifest.json` | `a7847c61253375e51f8ace61d61ccf69e3e36e68138d5255787dee45052b5444` |
| `outputs/previews/bench-users-v1.1/generation_inputs.json` | `e85e80dd2a51c9cb3ed378ef2e24644cba842ad9d4e55aade01251b07037f4bb` |
| `outputs/previews/bench-users-v1.1/generator_slope_gate.json` | `82f0b42fd3c08c9c5ca365622ad3faa3967ce963409c0b2248620ec717810c3a` |
| `outputs/previews/bench-users-v1.1/geometry/maastricht-terrain.npz` | `4ada382c3cededd889a1cea0e987fdcd59eea12a252dc998c680ec88d4e66a0d` |
| `outputs/previews/bench-users-v1.1/geometry/maastricht.json` | `e6c2718b41fd0d32fb444c83fe078765aca00951e298c67ce57d7f6a8a7325f4` |
| `outputs/previews/bench-users-v1.1/geometry/munich-terrain.npz` | `77b1f2bd8e25388c7289a4de77b87af7d76aa0c4aba9331d258d8154a481c3fa` |
| `outputs/previews/bench-users-v1.1/geometry/munich.json` | `5bd8e4d74554e874c9523c26fc3c5b5dd0ee2dc9f81a97c5ce8193babd635ca9` |
| `outputs/previews/bench-users-v1.1/geometry/rotterdam-terrain.npz` | `c5508f9e3f2a8992697701a08788314552d5c919f532cee5e6826d7c42dc0368` |
| `outputs/previews/bench-users-v1.1/geometry/rotterdam.json` | `7a5a93a35905a59f4ba99245c13e7cba9d3c26d24d057efd24e8bdba05444e66` |
| `outputs/previews/bench-users-v1.1/geometry/slope_gate.json` | `2a38699de064fc552efda81e282e076923838adbf1c0166d17047564d688419c` |
| `outputs/previews/bench-users-v1.1/gis-requirements.txt` | `3f52511d4a78a746fabd8d930a8c47778f8db4a97e96a8ebb1f13e424fd3e399` |
| `outputs/previews/bench-users-v1.1/ground_fill/audit.json` | `e224565983c745f19e926ce2417e066e4ee83abf8726ed1fa04344b30cab991a` |
| `outputs/previews/bench-users-v1.1/ground_fill/maastricht-footprint-ground.npz` | `bee2cd8a930df50e831309d50b4263c12e6c22071ac452a0b0c6543a8e1fd4c0` |
| `outputs/previews/bench-users-v1.1/ground_fill/maastricht-owners.json` | `f9e2b1cf034892dc4855fa26f1e50cb8c839565f67c9896d03513c7dc7c2c070` |
| `outputs/previews/bench-users-v1.1/ground_fill/munich-footprint-ground.npz` | `41ad058b7bb53a99ca61b5f5a806a3f55322320e56f0aebedd9deb3c344831f8` |
| `outputs/previews/bench-users-v1.1/ground_fill/munich-owners.json` | `c29fb2c9baa5e0323a7ed660d1101950c4945bcf3a58ee176aefa8070118155d` |
| `outputs/previews/bench-users-v1.1/ground_fill/rotterdam-footprint-ground.npz` | `53fd02fccc85633f80036a63a49e1b24a26597f52f8e8fe18fb8de668a5a0973` |
| `outputs/previews/bench-users-v1.1/ground_fill/rotterdam-owners.json` | `9c90142c9ea62355ea72be0ea98f00bfdbb59be1af499117012d6d9222d6cb6d` |
| `outputs/previews/bench-users-v1.1/recovered-source-manifest.json` | `45559d88150e03a85865e6aa7805c3351e9534e9c273295e6c46d3adc8733cb3` |
| `outputs/previews/bench-users-v1.1/site_pool.json` | `1a65382f7baf7fffcbbc0804447a4eaaf06aedc0b6291dcf5f04a96926ff6dc2` |
| `outputs/previews/bench-users-v1.1/validation.json` | `722bf2b40ea0f067f06633be49b468e5cba38ecddcc31cd41e0e517ff17580b5` |
| `scripts/generate_bench_users_preview.py` | `7335c1a8e58782d3ce51634bdea69d45d11c4113f090fe240fdf9ba6e7523a0f` |
| `scripts/prepare_bench_users_geometry.py` | `1f6adb2aed38f6ad6c8bc98d92c60521e5b7c40491eab5e89905969cd0ce2316` |
| `scripts/render_bench_users_preview.py` | `f0d0877c1bb3ad14983219d7c35bf290022f08f700f983289e46265ab2730dc1` |
| `scripts/run_bench_users_preview.sh` | `7b1558f62bada852e97cc01f13888f15ce16ac36557837a53f12ca2c730eca0d` |
| `scripts/validate_bench_users_preview.py` | `7b8755c21495f41a465d6ba01e3042786ef45fdb1ff4345e94d55a848c3255a5` |
| `src/atomwise_ris/bench/ground_fill.py` | `cd46b43f0da449ebeac034b82b0ddd10d58bf50ecf328e0dea89430206532458` |
| `src/atomwise_ris/bench/users.py` | `398f326b5fd7177bd3fa90a05ee7ade179c0ab17cb72a89151df959e8ca5c359` |
| `src/atomwise_ris/bench/users_geometry.py` | `481da8535e137fd55b7a48cad79d4fde841cfa500b7d4dde2217c4483564d54d` |
