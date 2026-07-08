# Step1X Recovery Diagnostics

- Scope: Apple-local Step1X recovery evidence after fixing background-removal defaults, MPS guidance defaults, and the MLX-to-MPS cache handoff.
- Status: not a validated promotion proof; keep Step1X experimental.
- Main recovery wins:
  - same-source teapot `i23d` is now structurally recognizable instead of a shapeless blob
  - espresso-machine `i23d` is now boxy and recognizable instead of a dense amorphous mass
- Remaining gap:
  - composed `t23d` is now stable after the MLX cache handoff fix, but the checked teapot result is still below the production geometry bar

## Cases

- `teapot_i23d_same_source`: task=`image_to_scene3d`, total=`125.6824` s, guidance=`None`, steps=`8`, octree=`128`, verts=`94913`, faces=`164044`
- `espresso_i23d_guidance3`: task=`image_to_scene3d`, total=`97.3601` s, guidance=`None`, steps=`8`, octree=`128`, verts=`166652`, faces=`200000`
- `teapot_t23d_mlx_cache_fix`: task=`text_to_scene3d`, total=`57.9011` s, guidance=`3.0`, steps=`8`, octree=`128`, verts=`201446`, faces=`200000`

## Assets

- `contact_sheet.png`: improved-case recovery sheet
- `comparison_contact_sheet.png`: before/after comparison against the earlier bad Step1X proof on matched teapot and espresso cases
