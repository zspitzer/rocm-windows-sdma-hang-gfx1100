# Results

Every run table from the investigation, oldest first. Server-side rows are fresh server starts, one first request each; in-process rows are the harness in `in-process/`; standalone rows are `standalone.py` in a fresh process. Hung counts are hung/total.

### Cures (workaround off unless stated, fresh start each, n=60 first request)

| Setting | Hung | Notes |
| --- | --- | --- |
| baseline | 6/8 | |
| `PAL_DISABLE_SDMA=1` | 0/12 | full perf suite clean, prefill 5024 tok/s at 4.5k, no speed cost. now the server default |
| `GPU_CP_DMA_COPY_SIZE=1024` | 0/6 | value is in KB, so copies under 1 MiB go CP DMA on the main queue, SDMA stays on for bigger ones. No perf run |
| `AMD_SERIALIZE_COPY=3` | 0/6 | full perf suite clean. Adds a CPU round trip to every copy |
| `AMD_SERIALIZE_KERNEL=3`, `HIP_LAUNCH_BLOCKING=1` | 0/6 | cure the hang, then the scheduler exits silently on the first decode request (a separate issue) |
| per-op `synchronize()` (`FT_H2D_TRACE=1`) | 0/8 | masks it |
| `HIP_HOST_COHERENT=1` | 6/6 | no effect |
| `DEBUG_CLR_MAX_BATCH_SIZE=1` | 4/6 | noise |
| `AMD_DIRECT_DISPATCH=1` | n/a | breaks startup, Linux-only default |

### Read-back probe results (2026-09-30)

Caveat: the probe object is created on its first call, which is inside the window between the SDMA copies and `index_put`, and its constructor does two small `torch.zeros(..., device=)` fills there. Every probe mode gets the same fills, so the host-store vs device-store comparison stands, but it means no mode leaves the window untouched and the command-buffer boundaries are not known.

| Run | Hung | mismatches a / b | Notes |
| --- | --- | --- | --- |
| control, SDMA off (`probe-rb-ctl`) | 0/2 | 0 / 0 | marks `[1, 1, 1]` both starts, requests 2.0 s / 1.7 s |
| SDMA on (`probe-rb`) | 0/6 | 0 / 0 | marks `[1, 1, 1]` every start, requests 1.7-1.8 s. **Fourth cure**: two mirror kernels plus three mark kernels between the SDMA copies and `index_put` |
| SDMA on, marks only (`probe-rb-marks`, `FT_H2D_PROBE_MARKS_ONLY=1`) | 0/6 | n/a | marks `[1, 1, 1]` every start, requests 1.7-1.8 s. Three single-program Triton kernels that each store one int64 into pinned host memory, no read of the SDMA-written tensors, still cure it |
| SDMA on, no probe (`probe-rb-noprobe`, `FT_H2D_PROBE` unset) | 2/3 | n/a | same session, same script, straight after the marks-only loop. Starts 1 and 2 hung (client timeout at 60 s), start 3 answered in 1.7 s |
| SDMA on, one mark kernel storing to a **device** int64 tensor (`FT_H2D_PROBE_MODE=devmark`) | 6/6 | n/a | same kernel as the marks, VRAM destination: no cure |
| SDMA on, torch `fill_` on an unrelated 4-element device tensor (`FT_H2D_PROBE_MODE=torchfill`) | 6/6 | n/a | an extra launch in that position is not enough |
| SDMA on, one pinned-host mark kernel **after** `index_put` only (`FT_H2D_PROBE_MODE=after`) | 6/6 | n/a | a pinned-host store before `index_put` cures, one after it does not |

### Bisect from the real request (2L model, SDMA on)

| Step | Removed | Hung | Notes |
| --- | --- | --- | --- |
| baseline | nothing | 6/6, +1 logged | SDMA off 0/2 |
| 1 | warmup (`FT_SKIP_WARMUP=1`) | 3/3 | SDMA off 1/1 OK (5.2 s incl. kernel compile). Settles the inconclusive server `probe-nowarm` |
| 2 | + in-flight init work (`MINI_PRESYNC=1`) | 3/3 | all load / init work drained before the request |
| 3 | + second stream (`MINI_ONESTREAM=1`) | 3/3 | no scheduler stream, no cross-stream event wait |
| 4 | + overlap loop (`FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`, instead of 3) | 3/3 | `normal_loop` on the engine stream |
| 5 | 1 + 2 + 4 combined: **the lean baseline** from here on | 3/3 | |
| - | lean baseline with `AMD_LOG_LEVEL=4` | **0/4** | requests 4.4-4.7 s. Logging lowers the rate: level 4 hung on the warmup config (the log in `evidence/logs/`), not on the lean baseline |
| - | lean baseline with `AMD_LOG_LEVEL=5` | **0/4** | requests 4.4-4.6 s instead of ~2 s. Two full-model server level-5 logs did hang, so this is a rate change, not a guarantee |

### Sync-point bisect inside the real request path (lean baseline, SDMA on)

| Sync at | Hung | Notes |
| --- | --- | --- |
| `token_pool_copy` (right after the 4604 B input-id copy) | 0/3 | requests 4.4-4.6 s |
| `prepare_batch:start` | 0/3 | |
| `allocate_paged` (after the 2 x 9208 B page-table copies + `index_put<4>`) | 2/2 | stuck inside the sync itself. Run 3 (rc 127) was the loop being stopped, not a result |
| `make_positions` | 2/2 | ran on after the stop, same reading |
| later labels | not run | after `allocate_paged`, can only hang |

### Swap and takeover bisect (lean baseline, SDMA on)

| Run | Hung | Reading |
| --- | --- | --- |
| control, same session (4 over the block) | 4/4 | |
| `MINI_SWAP=pagetable` (twin `_write_page_table` in the real path) | 3/3 | the page-table code is not special |
| `MINI_SWAP=presync-pt` (sync right before the real `_write_page_table`) | 0/3 | drain after `alloc_swa`, before the page-table copies, cures |
| takeover at `prepare_batch:start`, hand-written rest with an extra `arange` | 0/3 | |
| takeover at `token_pool_copy`, hand-written lookalike kernels | 0/3 | |
| takeover `realalloc` (real `_allocate` + `alloc_swa`), extra `arange` before the copies | 0/3, 0/3 | once with a log print in the window, once without |
| takeover `clonealloc` (same 3 kernels on full-size clones), extra `arange` | 0/3 | |
| takeover `smallalloc` (4096-element copies), extra `arange` | 0/3 | |
| **takeover `realalloc`, no extra kernel** | **3/3** | first hand-written tail that hangs |
| takeover `clonealloc`, no extra kernel | 0/3 | real buffers needed |
| takeover `realalloc` + `MINI_TO_EXTRA=1` | 0/3 | the extra kernel cures |

### Block reuse (lean baseline, SDMA on, 6 runs per row, takeover at `prepare_batch:start` unless stated)

| Run | Hung | Reading |
| --- | --- | --- |
| takeover `realalloc`, same session | 4/4 | control |
| takeover `halfreal` (hand-written `_allocate` slice, real `alloc_swa`) | 4/4 | `_allocate`'s pool-state change is not it |
| takeover `mix`, real order, `MINI_TO_FREE=1` (temporaries released right after `index_put`) | **6/6** | the hand-written ops hang once their blocks are freed |
| takeover `halfreal` + `MINI_TO_GRAB=1` (two `torch.empty` of 9208 B right after `alloc_swa`, no kernel) | **0/6** | the copies land elsewhere: cured |
| first `standalone.py` version, pinned host tensors held alive to the end (today's `SA_HOLD_HOST=1`), fresh process | 0/6 | block reuse confirmed (`data_ptr` of the copy destinations = the freed temporaries); the missing ingredient was the pinned frees, see Standalone below |
| `MINI_MODE=sa`: that sequence on FreeToken's real buffers right after init, no request path | 0/6 | the request path before `prepare_batch:start` contributes something |

### In-process takeover by label (lean baseline, `mix`, `MINI_TO_ORDER=real`, `MINI_TO_FREE=1`)

| Takeover at | Hung | Notes |
| --- | --- | --- |
| `token_pool_copy` | **5/6** | the 5-of-6 variant in `in-process/README.md`: the standalone sequence from step 2 on, inside the real request |
| `prepare_batch:start` | 2/2 | same session |

### Standalone (`standalone.py`, fresh process per run)

| Run | Hung | Notes |
| --- | --- | --- |
| first version, pinned host tensors alive to the end | 0/6 | block reuse confirmed by `data_ptr` |
| host tensors freed at FreeToken's points, SDMA on | **2/12** | runs 2 and 3 of the first six; stuck at the 4-byte copy (`standalone.py:84`) |
| same, `PAL_DISABLE_SDMA=1` | 0/6 | |
| same, `SA_HOLD_HOST=1` (host tensors alive) | 0/6 | |
| batch `b2`, 12 interleaved rounds: default | **6/12** | all at the 4-byte copy |
| `b2` `PAL_DISABLE_SDMA=1` / `SA_HOLD_HOST=1` / `GPU_CP_DMA_COPY_SIZE=1024` | 0/12 each | |
| `loop.py 12` (the AMD-facing driver), default | 2/12 | |
| batch `b3`, 12 interleaved rounds: default | 4/12 | |
| `b3` `SA_VRAM_GB=12` | 4/12 | VRAM occupancy is not the in-process difference |
| `b3` `SA_STREAM2=1` | 4/12 | nor the second stream |
| `b3` both | 2/12 | |
