"""Standalone attempt at the first-request hang, plain torch, no FreeToken.

usage: python standalone.py      (one attempt per fresh process; python loop.py [N] runs N and counts)

Mirrors what the bisect found in FreeToken's first request:
  1. an SDMA-sized pinned H2D copy (input ids into a big buffer made at "init")
  2. small kernels whose int64 temporaries are freed straight away (FreeToken's alloc_swa:
     mapping[free[:n].to(int64)] = swa_free[:n].to(int64))
  3. pinned H2D copies (> 1 KB, so SDMA) into fresh tensors of the same size, which reuse those
     just-freed caching-allocator blocks, then a kernel consuming them (page-table index_put)
  4. a 4-byte pageable copy, which blocks forever on a wedge

env: SA_N (default 1151), SA_TIMEOUT (default 20 s), SA_HOLD_HOST=1 keeps the pinned host tensors alive (old behaviour), SA_KEEP=1 keeps the step-2 temporaries alive (control:
the copies then land elsewhere), SA_NOFIRST=1 skips step 1
rate knobs (in-process differences): SA_VRAM_GB=<n> allocates an n GB dummy tensor first (weights / KV cache occupancy),
SA_STREAM2=1 builds the init buffers on a "scheduler" stream and runs the request on an "engine" stream that waits on its event
prints "SA OK" or a faulthandler dump on a wedge
"""
import faulthandler
import os
import time

import torch

n = int(os.environ.get("SA_N", "1151"))
dev = torch.device("cuda")
print("SA PAL_DISABLE_SDMA=%s GPU_CP_DMA_COPY_SIZE=%s n=%d keep=%s nofirst=%s hold=%s vram=%s stream2=%s" % (
	os.environ.get("PAL_DISABLE_SDMA", "unset"), os.environ.get("GPU_CP_DMA_COPY_SIZE", "unset"), n, bool(os.environ.get("SA_KEEP")),
	bool(os.environ.get("SA_NOFIRST")), bool(os.environ.get("SA_HOLD_HOST")), os.environ.get("SA_VRAM_GB", "0"), bool(os.environ.get("SA_STREAM2"))), flush=True)

with torch.inference_mode():
	if os.environ.get("SA_VRAM_GB"):
		dummy = torch.empty(int(float(os.environ["SA_VRAM_GB"]) * (1 << 30)), dtype=torch.uint8, device=dev)
	if os.environ.get("SA_STREAM2"):
		sched_stream = torch.cuda.Stream()
		eng_stream = torch.cuda.Stream()
		torch.cuda.set_stream(sched_stream)
	# "init": long-lived buffers, sizes as in the 2-layer FreeToken run
	token_pool = torch.full((5, 1 << 20), -1, dtype=torch.int32, device=dev)
	free_slots = torch.arange(5993821, dtype=torch.int32, device=dev)
	swa_free = torch.arange(1, 1198765, dtype=torch.int32, device=dev)
	mapping = torch.zeros(5993824, dtype=torch.int64, device=dev)
	page_table = torch.zeros((5, 131072), dtype=torch.int32, device=dev)
	torch.cuda.synchronize()
	if os.environ.get("SA_STREAM2"):
		# the scheduler's request path: engine stream waits on the scheduler stream, then runs the request
		eng_stream.wait_event(sched_stream.record_event())
		torch.cuda.set_stream(eng_stream)

	faulthandler.dump_traceback_later(int(os.environ.get("SA_TIMEOUT", "20")), exit=True)
	t = time.time()
	# 1. input ids, pinned, non_blocking (4604 B: SDMA)
	if not os.environ.get("SA_NOFIRST"):
		ids_host = torch.empty(n, dtype=torch.int32, pin_memory=True)
		ids_host.fill_(7)
		token_pool[0, :n].copy_(ids_host, non_blocking=True)
		if not os.environ.get("SA_HOLD_HOST"):
			# FreeToken's _maybe_pinned temporary dies on the copy line: torch records a stream event on the free
			del ids_host
	# 2. alloc_swa: two direct_copy kernels + index_put<8>, temporaries freed at the end of the statement
	full = free_slots[:n]
	if os.environ.get("SA_KEEP"):
		fi = full.to(torch.int64)
		sw = swa_free[:n].to(torch.int64)
		mapping[fi] = sw
	else:
		a64 = full.to(torch.int64)
		b64 = swa_free[:n].to(torch.int64)
		ptrs = (a64.data_ptr(), b64.data_ptr())
		mapping[a64] = b64
		del a64, b64
	# 3. page-table write: two pinned int64 copies (9208 B each: SDMA) into fresh tensors, then index_put<4>
	tidx_host = torch.empty(n, dtype=torch.int64, pin_memory=True)
	pos_host = torch.empty(n, dtype=torch.int64, pin_memory=True)
	tidx_host.fill_(0)
	torch.arange(0, n, out=pos_host)
	tidx_dev = tidx_host.to(dev, non_blocking=True)
	pos_dev = pos_host.to(dev, non_blocking=True)
	page_table[tidx_dev, pos_dev] = full
	if not os.environ.get("SA_HOLD_HOST"):
		# FreeToken's _write_page_table returns here, freeing its two pinned host tensors (event records on the stream)
		del tidx_host, pos_host
	# 4. pageable 4-byte copy: returns only if the queue drains
	prefix = torch.tensor([0], dtype=torch.int32, device=dev)
	torch.cuda.synchronize()
	faulthandler.cancel_dump_traceback_later()
	print("SA OK %.3fs" % (time.time() - t), flush=True)
	if not os.environ.get("SA_KEEP"):
		print("SA temps %s, copy dsts %s, reused=%s" % ([hex(p) for p in ptrs], [hex(tidx_dev.data_ptr()), hex(pos_dev.data_ptr())],
			sorted(ptrs) == sorted((tidx_dev.data_ptr(), pos_dev.data_ptr()))), flush=True)
os._exit(0)
