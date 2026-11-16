# Standalone workflow verification

Validated on 2026-11-16 using Python 3.11.14, Open3D 0.19.0 and CPU execution on macOS. The input was the same nine November 13 saved sessions used to validate the corrected J1 workflow. All 168 captures were processed by the public `reconstruct.py` entry point, with the default processing settings and explicit input/output paths.

| Object | Scan type | Frames | Exported triangles | Maximum vertex difference from reference |
|---|---|---:|---:|---:|
| Cube | Linear barrel | 20 | 490,617 | 0 mm |
| Cube | Oriented barrel | 20 | 457,519 | 0 mm |
| Cube | Planar | 16 | 524,590 | 0 mm |
| Cone | Linear barrel | 20 | 486,885 | 0 mm |
| Cone | Oriented barrel | 20 | 452,239 | 0 mm |
| Cone | Planar | 16 | 520,859 | 0 mm |
| Sphere | Linear barrel | 20 | 492,468 | 0 mm |
| Sphere | Oriented barrel | 20 | 455,101 | 0 mm |
| Sphere | Planar | 16 | 523,190 | 0 mm |

Reference meshes were the previously validated undistorted meshes with the 100-triangle disconnected-component cleanup. Both nearest-vertex distances and triangle connectivity matched exactly after accounting for vertex/index ordering. The oriented cube's measured vertex colors also matched its existing colored reference exactly; the other eight reference meshes did not contain vertex colors, so no reference-color comparison is claimed for those eight.

All nine runs exported a colored PLY mesh, STL, RGB and confidence point clouds, a table-plane estimate, aligned images, a preview, logs and completion metadata. Source image/calibration/manifest SHA-256 hashes matched before and after each run. The oriented-cube preview was visually inspected. The complete structured comparison is in `validation_results.json`; raw scans and generated geometry remain outside this repository.

The automated suite passes **51 tests**, including real synthetic TSDF runs with undistortion on and off, bias −5.1 mm and 0 mm, and an intentionally excessive extraction threshold that must fail rather than report success. Analytical radial-projection tests protect the undistortion math; component tests check retained coordinates/colors and attached details. Additional end-to-end tests cover nested print cycles sharing session-root calibration and restoring losslessly archived reconstruction artifacts.

These checks establish processing parity with the saved-scan reference. They do not establish absolute dimensional accuracy or a new depth calibration. Windows and CUDA have not been tested locally. GitHub Actions runs the test suite on Ubuntu/Python 3.11; its workflow status records that result separately.

The combiner tests verify exact stacked coordinates/colors/face indices, shared TSDF evidence and color accumulation across cycles, rejection of mismatched coordinate frames and changed calibration, reading archived results, and relocation of raw data.

## Combining a growing print

A 34-cycle sequence (544 depth frames) was replayed on CPU. In the printed region, disabling the 100-triangle component cleanup restored 14,078 triangles; cleanup had removed 3.55% of the raw region's triangles. Lowering the TSDF extraction threshold from 3 to 1 with cleanup enabled added only 127 triangles (0.03%). Neither change repaired the large gaps.

Capping accumulated TSDF weight at 8 after each cycle substantially improved visible completeness while retaining threshold 3 and the original cleanup. On 50,000 sampled vertices from the voxel-sampled union of per-cycle surfaces in that region, the fraction within 3 mm of the fused mesh rose from 42.7% to 73.5%. This measures retention of reconstructed surfaces, including intermediate print stages and possible noise; it is **not** ground-truth accuracy. Both models were compared with identical camera and shading settings. Surface remeshing with ball pivoting also improved retention in a separate region-only experiment; that method is not part of the production combiner.

The new automated tests check static and moving planes against analytical weighted-depth expectations, cross-cycle evidence/color accumulation with the cap enabled, and rejection of caps that would make the extraction threshold unreachable. The optional `fused_history_weight_cap` defaults to `None`; ordinary fusion remains unchanged.
