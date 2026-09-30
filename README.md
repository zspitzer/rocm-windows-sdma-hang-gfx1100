# rocm-windows-sdma-hang-gfx1100

First compute submission after an SDMA host-to-device copy into recently freed memory, with the pinned source tensors freed straight after (torch records a stream event for each), never signals its fence. Windows 11, RX 7900 XT (gfx1100), HIP runtime from the TheRock `rocm-sdk 10.1` nightly. Plain PyTorch reproduces it. `PAL_DISABLE_SDMA=1` cures it.

**The ask:** this looks like a PAL / WDDM fence or ordering bug between an SDMA copy and the next compute submission on gfx1100. The stuck queue's counters are in [`evidence/cdb/hung-start-r1-queue.txt`](evidence/cdb/hung-start-r1-queue.txt), the stacks are below.

## Run it

```
set HIP_VISIBLE_DEVICES=1
python loop.py 20
```

`HIP_VISIBLE_DEVICES=1` is only because this machine's iGPU is device 0; with the gfx1100 card as the only GPU, leave it unset. Installing the exact wheels: [ENVIRONMENT.md](ENVIRONMENT.md).

**Expect a few hangs in 20 attempts; a single run usually passes.** On this machine the default hung 14 of 48 fresh processes; with `PAL_DISABLE_SDMA=1` 0 of 18. `loop.py` starts `standalone.py` in a fresh process per attempt and counts. A pass prints `SA OK`; a hang prints a faulthandler stack after 20 s, stuck at the final `torch.tensor([0], dtype=torch.int32, device=dev)`, and the process exits. `loop.py` writes each hung run's output to `hangs/hung-<run>.txt` for attaching. Nothing needs a reboot afterwards.

Controls: `PAL_DISABLE_SDMA=1`, `GPU_CP_DMA_COPY_SIZE=1024`, or `SA_HOLD_HOST=1` (keeps the pinned host tensors alive so no stream events are recorded between the copies and the consumer). All three have been 0 hangs.

## What the script does

All on one stream, sizes as in the real server's first request:

1. A 4.6 KB pinned host-to-device copy (large enough for the SDMA engine) into a long-lived buffer; the pinned host tensor is freed
2. Two `direct_copy` kernels and an `index_put` whose int64 temporaries (9.2 KB each) are freed at the end of the statement
3. Two 9.2 KB pinned host-to-device copies into fresh tensors, which reuse the step-2 blocks (the script prints the `data_ptr`s), an `index_put` consuming them, and the pinned host tensors freed
4. A 4-byte pageable host-to-device copy, which returns only when the queue drains. On a hang it never does

## What a hang looks like

App thread:

```
amdhip64_7!amd::Event::awaitCompletion
amdhip64_7!amd::HostQueue::finishCommand
amdhip64_7!hip::ihipMemcpy
amdhip64_7!hipMemcpyWithStream
c10_hip!c10::cuda::memcpy_and_sync
```

HIP host queue thread:

```
amdhip64_7!Pal::Wddm::Fence::WaitForFences
amdhip64_7!amd::pal::VirtualGPU::Queue::waitForFence<0>       result = NotReady
amdhip64_7!amd::pal::VirtualGPU::Queue::waitForEvent
amdhip64_7!amd::pal::VirtualGPU::awaitCompletion
amdhip64_7!amd::pal::VirtualGPU::submitMarker
amdhip64_7!amd::HostQueue::loop
```

Main queue `cmdBufIdCurrent_` 15, `cmbBufIdRetired_` (sic, the CLR spelling) 8; SDMA queue 5 / 4 with nothing outstanding. Main-engine command buffer 9, the first after the SDMA copies, is the one that never retires. `AMD_LOG_LEVEL=4` repeats `PAL fence isn't ready! result:3` every ~6 s.

Files: `evidence/cdb/` has one full cdb capture of a hung server start on the rebuilt runtime, its `GPU_ANALYZE_HANG` queue dump and locals. `evidence/logs/` has an `AMD_LOG_LEVEL=4` log of a hung start of the in-process harness (2-layer model, with warmup; logging makes the standalone hang rare, so there is no log of a hung standalone run). `evidence/results.md` has every run table. `clr-sdma-acquire.patch` is the runtime-side acquire we tried; it did not help. `in-process/patches/freetoken-fork.diff` is the FreeToken tree every server and in-process run used, including the read-back probe behind the host-store result.

## Workaround

`PAL_DISABLE_SDMA=1`. No measurable cost on this workload (prefill about 4900 tok/s, decode about 112 tok/s on gpt-oss-20b).

## Related

- [ROCm/rocm-systems#11605](https://github.com/ROCm/rocm-systems/issues/11605): same `PAL fence isn't ready` loop on Windows
- [ROCm/rocm-systems#10553](https://github.com/ROCm/rocm-systems/issues/10553): CPU drain before a non-P2P staged copy, same file, different path
- [ROCm/rocm-systems#12067](https://github.com/ROCm/rocm-systems/issues/12067): SDMA user queues on Windows, gfx12+ only, so gfx11 stays on the PAL SDMA path
- [ggml-org/llama.cpp#28178](https://github.com/ggml-org/llama.cpp/issues/28178): same `GPU_CP_DMA_COPY_SIZE` routing, RGP capture of the SDMA barriers
- [pytorch/pytorch#196377](https://github.com/pytorch/pytorch/issues/196377): Windows stream left un-drained, no event and no TDR

## How we got here

**The goal.** Serve gpt-oss-20b locally on an RX 7900 XT (gfx1100) on Windows 11, using the [Windows-ROCm port of the FreeToken inference server](https://github.com/Maxritz/FreeToken-ROCm) on the TheRock nightly torch wheels. By 29 September that worked: correct output, about 112 tok/s decode with the fused MoE backend, prefill around 4900 tok/s. One remaining problem: about three fresh server starts in four hung on the first real request. Warmup passed every time. The scheduler's main thread sat forever in a 4-byte host-to-device copy, GPU at 0% on every engine, no TDR, no driver reset, nothing in Event Viewer. The process had to be killed.

**What did not matter (29 September).** CUDA graphs on or off, prompt length and chunking, torch 2.13 vs 2.14, overlap scheduling on or off, fused vs offload MoE backend, the expert host banks and `hipHostRegister`. A ladder of pure-torch host-to-device scripts (plain copies, pinned copies, memory pressure, extra streams, a second process) ran 0 hangs in over a hundred attempts. So plain H2D was not it; something the server queued was.

**The knobs.** `AMD_SERIALIZE_COPY=3` cured it and became the first workaround. `AMD_SERIALIZE_KERNEL=3` and `HIP_LAUNCH_BLOCKING=1` also cured the hang but the scheduler then died silently on a long prefill, a separate problem. `PAL_DISABLE_SDMA=1` cured it with no measurable cost and is now the default. `GPU_CP_DMA_COPY_SIZE=1024` also cured it, which pointed at the routing decision between CP DMA on the main queue and the SDMA engine. `HIP_HOST_COHERENT=1` did nothing.

**Reading the runtime (29 to 30 September).** Level-4 runtime logs of a hang repeated `PAL fence isn't ready! result:3` every few seconds. Level-5 logs showed the API call sequence identical between a passing and a hanging start. We cloned rocm-systems at the exact commit the wheel was built from, rebuilt `amdhip64_7.dll` with symbols, and captured hung processes with cdb. Four captures (one is in `evidence/cdb/`), two MoE backends, identical numbers: the app thread polling in `hipMemcpyWithStream`, the HIP host queue thread in `Pal::Wddm::Fence::WaitForFences` on the main compute queue, command buffer 9 never retiring, the SDMA queue idle and fully retired. The last kernel in the stuck buffer was torch's `index_put`, consuming two int64 index tensors that the SDMA engine had just written from pinned memory.

**Two wrong theories, both tested.** First, the CLR PAL backend's SDMA copy branch issues no `CopyToKernel` acquire where the CP DMA branch does, so a stale compute cache looked plausible. We patched the rebuilt DLL to add the acquire: still hung 6 of 6. Second, stale data more generally: a read-back probe in the server copied the index tensors to pinned host memory from a kernel before `index_put`. It never hung, and the values were correct every time. Variants showed that any kernel in that position which stores to pinned host memory cures it, while the same kernel storing to VRAM still hangs. So the consumer kernel and its data were not the point: the first main-engine submission after the SDMA copies never signals, whatever kernel it holds.

**The sharpest clue we have.** A kernel that stores to pinned host memory, placed before `index_put`, cures it. The same store to VRAM does not. The same host store placed after `index_put` does not. One caveat: the probe object is created inside that window on the first request and does two small device fills there, in every mode, so the comparison between modes holds but the exact command-buffer boundary is not known.

**Replay does not reproduce it.** A C++ player replayed the full HIP API sequence of a hung start (7254 calls: same allocations, pinned vs pageable copies, streams, events, graph captures, timing) with dummy kernels. About 70 runs with busy kernels, cache-dirtying traffic, host reads and scratch use: never hung. In hindsight the reason is that dummy kernels do not touch the same allocator blocks as the real ones, and the trigger depends on exactly that.

**Bisecting from the hanging side (30 September).** We built an in-process harness: the server's scheduler driven directly in one process, a 2-layer cut of the model with real weights, about 15 seconds per cold start, same hang rate as the server. Then we removed ingredients one at a time while it still hung: warmup, all pending init work, the second stream, the overlap loop. All removable. A sync-point bisect inside the request placed the edge exactly: a device sync before the page-table write cures, a sync straight after it hangs inside the sync. Swapping pieces of the real path for hand-written twins found that the twins only hang when their int64 temporaries are freed straight after use, so the SDMA copies land in the blocks those kernels just wrote; allocating two tensors of the same size first, so the copies land elsewhere, cures it. The last ingredient was the lifetime of the pinned host tensors: freeing them at the same points the server does, which records a stream event for each, made a plain torch script hang.

**Why it was slow.** Instrumentation cures it. Level-4 and level-5 logging, a sync per op, a print in the window, an extra kernel, all reduce or remove the hang, so every probe needed an unmodified control in the same session, and there is still no runtime log of a hung standalone run. And an API log is not enough to copy a sequence: every visible difference between the hung window and the hand-written copy was restored before the real difference turned out to be allocator block reuse and pinned-tensor lifetimes, neither of which appears as an API call.

**What is in this repo.** The standalone script and loop driver, the in-process harness and its takeover modes, the model-cut script, the level-4 log of a hung in-process start, one cdb capture of a hung server start with the queue dump and locals, the CLR patch that did not help, the FreeToken diff (trace hook, read-back probe, warmup gate, port fixes), and the results tables for every negative above. The C++ replay player is not included.

## Lessons, for anyone chasing something like this

- Start from something that hangs and remove ingredients. Hand-written sequences and API replays produced about a hundred clean runs with nothing to compare against; bisecting the real path found the trigger in an afternoon
- Instrumentation cures it. Runtime logging at level 4 or 5, a sync per op, a print or an extra kernel in the window all reduce or remove the hang. Every probe needs an unmodified control in the same session
- An API log is not enough to copy a sequence. The trigger depends on which caching-allocator blocks the kernels and copies touch and on when pinned tensors are freed, neither of which appears as an API call
- Dummy kernels cannot reproduce it, for the same reason
- Rates, not certainties. At one hang in three, a negative needs 12 or more runs
- `os._exit` drops the runtime log's buffered tail; sleep a few seconds first
- `long` is 32-bit on Windows; 2 GiB sizes in C++ tooling overflow
