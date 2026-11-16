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

The automated suite passes **41 tests**, including real synthetic TSDF runs with undistortion on and off, bias −5.1 mm and 0 mm, and an intentionally excessive extraction threshold that must fail rather than report success. Analytical radial-projection tests protect the undistortion math; component tests check retained coordinates/colors and attached details. Additional end-to-end tests cover nested print cycles sharing session-root calibration and restoring losslessly archived reconstruction artifacts.

These checks establish processing parity with the saved-scan reference. They do not establish absolute dimensional accuracy or a new depth calibration. Windows and CUDA have not been tested locally. GitHub Actions runs the test suite on Ubuntu/Python 3.11; its workflow status records that result separately.

The combiner tests verify exact stacked coordinates/colors/face indices, shared TSDF evidence and color accumulation across cycles, rejection of mismatched coordinate frames and changed calibration, reading archived results, and relocation of raw data.
