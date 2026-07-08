# TripoSR Rocket Cleanup Comparison

- Raw contact sheet: `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d-triposr-cleanup/raw/contact_sheet.png`
- Cleaned contact sheet: `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d-triposr-cleanup/clean/contact_sheet.png`
- Comparison contact sheet: `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d-triposr-cleanup/summary/comparison_contact_sheet.png`

## Raw

- verts: `13471`
- faces: `26904`
- topology: `{'is_watertight': False, 'body_count': 1, 'euler_number': -15}`
- cleanup: `[]`

## Cleaned

- verts: `8999`
- faces: `17988`
- topology: `{'is_watertight': False, 'body_count': 1, 'euler_number': -4}`
- cleanup: `['remove_small_components:67', 'marching_cube_cleanup', 'taubin_smooth:4', 'repair_non_manifold_edges', 'repair_non_manifold_vertices', 'close_holes:24', 'merge_vertices', 'fix_normals']`