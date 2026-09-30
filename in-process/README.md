# In-process harness (needs the FreeToken ROCm fork)

The higher-rate variant: the server's scheduler driven directly in one process, no API, tokenizer or HTTP, a 2-layer cut of gpt-oss-20b with real weights. About 15 s per cold start. Same stuck stack as the server. A plain first request hangs 6 of 6 with warmup; the hand-written tail below hangs 5 of 6.

Needs: the [Windows-ROCm FreeToken fork](https://github.com/Maxritz/FreeToken-ROCm) at commit `e8545da` with `patches/freetoken-fork.diff` applied (`git apply`), installed in the same venv as torch, and a 2-layer model cut. The diff is exactly the tree every in-process and server run used: the inert `_T(label)` trace hook the harness rebinds (`h2dtrace.py`, off unless `FT_H2D_TRACE=1`), the read-back probe (`h2dprobe.py`, off unless `FT_H2D_PROBE=1`), the `FT_SKIP_WARMUP` gate in `engine/engine.py`, and the port fixes the model needs to run on this stack (MoE align kernel, CPU MoE extension, kernel utils).

```
python make_small_model.py <gpt-oss-20b dir> <out dir> 2
set MINI_MODEL_PATH=<out dir>
set HIP_VISIBLE_DEVICES=1
python mini_engine.py
```

Prints `MINI OK <s>` or a faulthandler dump of every thread after `MINI_TIMEOUT` seconds (default 60) on a hang.

## The 5-of-6 variant

Lean baseline plus a takeover at the `token_pool_copy` label (right after the input-id copy), with `alloc_swa`'s ops written by hand on the server's real buffers, in `alloc_swa`'s evaluation order, temporaries freed straight after:

```
set FT_SKIP_WARMUP=1
set MINI_PRESYNC=1
set FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1
set MINI_TAKEOVER=token_pool_copy
set MINI_TO_VARIANT=mix
set MINI_TO_ORDER=real
set MINI_TO_FREE=1
python mini_engine.py
```

This is the standalone script's sequence from step 2 on, run inside the real request.

## Knobs

Real request path:

- unset: a real first request
- `FT_SKIP_WARMUP=1`, `MINI_PRESYNC=1` (device sync after init), `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`: together, the lean baseline used for every bisect. `MINI_ONESTREAM=1` puts the scheduler on the engine stream
- `MINI_SYNC_AT=<label>[,<label>]`: device sync at trace-hook labels, the sync-point bisect. Labels in request order: `token_pool_copy`, `prepare_batch:start`, `allocate_paged`, `make_positions`, `make_input_tuple`, `make_write_tuple`, `page_table_index`, `prepare_metadata:pre229`
- `MINI_SWAP=pagetable`: the real `_write_page_table` replaced by a hand-written twin. `MINI_SWAP=presync-pt`: device sync right before the real `_write_page_table`

Takeover (`MINI_TAKEOVER=<label>`): real request up to the label, then a hand-written rest of the window from inside the hook, then exit. `MINI_TO_VARIANT` picks what runs `alloc_swa`'s three kernels (`direct_copy` x2, `index_put`):

- `realalloc`: the real `_allocate` + `alloc_swa`
- `halfreal`: hand-written `_allocate` slice, real `alloc_swa`. `MINI_TO_GRAB=1` then allocates two 9208 B tensors so the page-table copies land elsewhere
- `mix`: the three ops by hand. `MINI_TO_CLONE=fs,sf,mp` swaps `free_slots`, `_swa_free` and the mapping for clones; `MINI_TO_ORDER=real` uses `alloc_swa`'s evaluation order; `MINI_TO_FREE=1` releases the temporaries
- `clonealloc` / `smallalloc`: the same ops on full-size clones / 4096-element copies
- unset: lookalike kernels (only at `token_pool_copy`)
- `MINI_TO_EXTRA=1`: build the page-table values on the device, one extra kernel before the copies (a cure)

Hand-written modes that never hung, kept for the record (`MINI_MODE`): `ops`, `trace` (`MINI_STEPS`), `real`, `real3` (`MINI_EMPTY_CACHE`, `MINI_HOLD_PINNED`, `MINI_GAP_US`), `stale` (`MINI_STALE_FILL`, `MINI_STALE_NOCOPY`), `sa`. See the comments in the file.

The read-back probe in the same diff is the server-side instrumentation behind the "host store cures, device store does not" result: `FT_H2D_PROBE=1`, `FT_H2D_PROBE_MARKS_ONLY=1`, `FT_H2D_PROBE_MODE=full|devmark|torchfill|after`.

Open question: why this hangs 5 of 6 where `standalone.py` hangs about 1 in 3. Tried and ruled out: 12 GB of VRAM already in use (`SA_VRAM_GB`), a second stream with an event wait (`SA_STREAM2`), both together.
