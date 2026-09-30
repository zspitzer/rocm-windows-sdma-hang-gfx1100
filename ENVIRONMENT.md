# Environment

- RX 7900 XT 20 GB (gfx1100), HIP device 1; Ryzen 9950X3D iGPU (gfx1036) hidden with `HIP_VISIBLE_DEVICES=1`
- Windows 11 Pro 26200, AMD driver 32.0.21030.2001
- TheRock `rocm-sdk 10.1.0a20260822` (rocm-systems `hip-version_10.1.62330`), `torch 2.13.0+rocm10.1.0a20260822`, Python 3.12. Server-side hang also on `torch 2.14.0+rocm10.2`

- CLR source for the rebuilt runtime: rocm-systems at tag `hip-version_10.1.62330`, `projects/clr`, RelWithDebInfo with PDB. The rebuilt `amdhip64_7.dll` hangs at the same rate as the shipped one and has the same stack shapes
- The integrated GPU is enabled in firmware on this machine and only hidden with `HIP_VISIBLE_DEVICES=1`. Untested with it disabled

## `HIP_VISIBLE_DEVICES`

`HIP_VISIBLE_DEVICES=1` is only there because the iGPU is device 0 on this machine. On a box where the gfx1100 card is the only (or first) GPU, leave it unset or use `0`.

## Installing the same stack

`standalone.py` and `loop.py` need only torch. The wheels come from AMD's nightly index, all on one nightly stamp. This is how the venv was built (download, then install with `--no-deps` so pip does not pull a CUDA torch from PyPI):

```
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip setuptools wheel
$env:INDEX = "https://rocm.nightlies.amd.com/whl-multi-arch/"
$env:STAMP = "10.1.0a20260822"
venv\Scripts\python -m pip download --index-url $env:INDEX -d wheels "rocm[libraries,devel,device-gfx1100]==$env:STAMP"
venv\Scripts\python -m pip download --no-deps --index-url $env:INDEX -d wheels "torch==2.13.0+rocm$env:STAMP" "amd-torch-device-gfx1100==2.13.0+rocm$env:STAMP"
venv\Scripts\python -m pip install (Get-ChildItem wheels\*.whl) --no-deps --force-reinstall
venv\Scripts\python -m pip install "wheels\rocm-$env:STAMP.tar.gz" --no-deps --no-build-isolation
venv\Scripts\python -m pip install typing_extensions sympy filelock networkx jinja2 fsspec numpy
```

The block is PowerShell. Everything is installed with `--no-deps` so pip cannot pull a CUDA torch from PyPI, so torch's pure-Python dependencies go in by hand on the last line (the versions here: typing_extensions 4.16.0, sympy 1.14.0, filelock 4.0.6, networkx 3.7, jinja2 3.1.6, fsspec 2026.9.0, numpy 2.4.6). The resulting packages, with the sha256 of the files installed here:

| Package | sha256 |
| --- | --- |
| `torch-2.13.0+rocm10.1.0a20260822-cp312-cp312-win_amd64.whl` | `5e0d7215b6c6585403e862db2fe48d1aadf254459cdc847037464cf9841e2be0` |
| `amd_torch_device_gfx1100-2.13.0+rocm10.1.0a20260822-cp312-cp312-win_amd64.whl` | `d72c0fd21a55f24331686d3fe5859694723abfee8ac5691ecbe3bdcf5386356b` |
| `rocm_sdk_core-10.1.0a20260822-py3-none-win_amd64.whl` | `6211f8b693b900fef3a2d4facb3ff995077acfa782aab89db21cb0874449a690` |
| `rocm_sdk_devel-10.1.0a20260822-py3-none-win_amd64.whl` | `fbe70f3606bd86c5684a1d1e2830bd1865ea0ed92ee4e7a2165bb5c038b927e9` |
| `rocm_sdk_device_gfx1100-10.1.0a20260822-py3-none-win_amd64.whl` | `c896ad3307466b0534f05134ab03d6f2fda3d4d2c7699fe584816f4f032231ac` |
| `rocm_sdk_libraries-10.1.0a20260822-py3-none-win_amd64.whl` | `4210ffe35dbcbe4aa5038364ee9321cc2cb3f27354d55d0f925c24a140da2728` |
| `rocm-10.1.0a20260822.tar.gz` | `a716097245003c1de6531e495be9ad1948feacfff9c606c7b80659d6a26569cc` |

The in-process harness also needs the FreeToken fork's own dependencies (`triton-windows==3.7.1.post27` and the rest of its installer's list).
