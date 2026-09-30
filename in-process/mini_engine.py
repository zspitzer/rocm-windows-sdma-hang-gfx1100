"""In-process FreeToken scheduler, one cold first request, no server / tokenizer processes.

usage: python mini_engine.py [extra serve args...]      (env knobs: see README.md)

Parses the server's argv with its own parse_args, switches to offline mode and
drives the Scheduler through the LLM offline hooks. Same prompt as the server hang loop
(1151 tokens, chat template applied, max_tokens 1). A faulthandler watchdog dumps every thread's stack
and exits if the request has not finished after MINI_TIMEOUT seconds (default 60).

prints one of:
  MINI OK <seconds>        request finished
  (faulthandler dump)      wedged, process exits via the watchdog
"""
import dataclasses
import faulthandler
import os
import sys
import time

import torch

from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.scheduler import Scheduler
from freetoken.server.args import parse_args

SERVE_ARGV = [
	"--model-path", os.environ.get("MINI_MODEL_PATH", "models/gpt-oss-20b-2L"),
	"--moe-backend", "offload",
	"--moe-cache-auto",
	"--cuda-graph-max-bs", "0",
]


class MiniLLM(LLM):
	# LLM.__init__ builds its own config; take the server's parsed one instead
	def __init__(self, config):
		Scheduler.__init__(self, config)
		self.pending_requests = []
		self.status_map = {}
		self.mm_embeds_map = {}
		self.counter = 0


def log(msg):
	print(msg, flush=True)


def main():
	args, _ = parse_args(SERVE_ARGV + sys.argv[1:])
	cfg = dataclasses.replace(args, offline_mode=True)
	log("MINI PAL_DISABLE_SDMA=%s" % os.environ.get("PAL_DISABLE_SDMA", "unset"))
	t = time.time()
	with torch.inference_mode():
		llm = MiniLLM(cfg)
	log("MINI ready %.1fs" % (time.time() - t))
	# bisect knobs for the real request path
	if os.environ.get("MINI_ONESTREAM"):
		# scheduler works on the engine's stream: no second stream, no cross-stream waits
		llm.stream = llm.engine.stream
		torch.cuda.set_stream(llm.engine.stream)
		log("MINI onestream")
	sync_at = [x for x in os.environ.get("MINI_SYNC_AT", "").split(",") if x]
	takeover = os.environ.get("MINI_TAKEOVER", "")
	if takeover:
		# real request up to the _T label, then the hand-written rest of the window from inside the hook, then exit.
		# token_pool_copy: after the input-id SDMA copy; prepare_batch:start: after its 3 kernels too
		import freetoken.attention.triton as m_attn
		import freetoken.scheduler.prefill as m_prefill
		import freetoken.scheduler.scheduler as m_sched
		from freetoken.core import get_global_ctx

		def _T(label, *a, **k):
			if label != takeover:
				return
			dev = llm.device
			page_table = get_global_ctx().page_table
			n = len(ids)
			t = time.time()  # no prints inside the window: a print is a syscall gap
			variant = os.environ.get("MINI_TO_VARIANT", "")
			if variant:
				# alloc_swa's three kernels (direct_copy x2, index_put<8>) on: the real buffers (realalloc),
				# full-size clones made before the request (clonealloc), or n-element copies (smallalloc)
				cm = llm.cache_manager
				if variant == "realalloc":
					allocated = cm._page_to_token(cm._allocate(n // cm.page_size))
					cm.swa_pool.alloc_swa(allocated)
				elif variant == "halfreal":
					# hand-written _allocate (slice only, pool state untouched), then the real alloc_swa
					allocated = cm.free_slots[:n]
					cm.swa_pool.alloc_swa(allocated)
					if os.environ.get("MINI_TO_GRAB"):
						# take the blocks alloc_swa just freed (no kernel launched), so the page-table copies land elsewhere
						grab = [torch.empty(n, dtype=torch.int64, device=dev) for _ in range(2)]
				else:
					fs, sf, mp = clones
					if os.environ.get("MINI_TO_ORDER") == "real":
						# alloc_swa's evaluation order: mapping[full.to(int64)] = swa.to(int64) evaluates the RHS first
						sw = sf[:n].to(torch.int64)  # direct_copy (_swa_free)
						fi = fs[:n].to(torch.int64)  # direct_copy (free_slots)
					else:
						fi = fs[:n].to(torch.int64)  # direct_copy
						sw = sf[:n].to(torch.int64)  # direct_copy
					mp[fi] = sw  # index_put<8>
					if os.environ.get("MINI_TO_FREE"):
						# release the temporaries now, like alloc_swa's one-statement form: the page-table copies'
						# destinations can then reuse these just-freed caching-allocator blocks
						del fi, sw
					allocated = fs[:n]
			if label == "token_pool_copy" and not variant:
				ids_dev = llm.token_pool[0, :n]
				a64 = ids_dev.to(torch.int64)  # direct_copy
				b64 = ids_dev.to(torch.int64)  # direct_copy
				tok = torch.empty((5, 4096), dtype=torch.int64, device=dev)
				tok[torch.zeros_like(b64), torch.arange(n, device=dev)] = a64  # index_put<8> (+ its index builders)
			# value tensor must already exist: building it here (e.g. torch.arange on the device) puts an extra kernel
			# between alloc_swa's kernels and the page-table copies. MINI_TO_EXTRA=1 keeps that extra kernel
			if variant and not os.environ.get("MINI_TO_EXTRA"):
				vals = allocated.to(page_table.dtype) if allocated.dtype != page_table.dtype else allocated
			else:
				vals = torch.arange(n, dtype=page_table.dtype, device=dev)
			m = len(vals)
			tidx = torch.zeros(m, dtype=torch.int64).pin_memory()
			pos = torch.arange(m, dtype=torch.int64).pin_memory()
			page_table[tidx.to(dev, non_blocking=True), pos.to(dev, non_blocking=True)] = vals  # 2 x 9208 B + index_put<4>
			prefix_lens = torch.tensor([0], dtype=torch.int32, device=dev)  # 4 B pageable
			torch.cuda.synchronize()
			faulthandler.cancel_dump_traceback_later()
			log("MINI OK %.1fs (takeover %s %s)" % (time.time() - t, label, os.environ.get("MINI_TO_VARIANT", "")))
			os._exit(0)
		for m in (m_attn, m_prefill, m_sched):
			m._T = _T
	if sync_at:
		# rebind the fork's inert _T(label) trace hook in the modules that call it: device sync at the chosen labels.
		# Labels in request order: token_pool_copy, prepare_batch:start, allocate_paged, make_positions,
		# make_input_tuple, make_write_tuple, page_table_index, prepare_metadata:pre229
		import freetoken.attention.triton as m_attn
		import freetoken.scheduler.prefill as m_prefill
		import freetoken.scheduler.scheduler as m_sched

		def _T(label, *a, **k):
			if label in sync_at:
				torch.cuda.synchronize()
				log("MINI sync at %s" % label)
		for m in (m_attn, m_prefill, m_sched):
			m._T = _T
	swaps = [x for x in os.environ.get("MINI_SWAP", "").split(",") if x]
	if "pagetable" in swaps:
		# real request, but scheduler/cache.py's _write_page_table replaced by a hand-written twin
		# (same ops: two pinned int64 host tensors, non_blocking H2D, index_put)
		import freetoken.scheduler.cache as m_cache

		def _write_page_table_twin(page_table, allocated, allocation_info, page_size):
			n = len(allocated)
			tidx = torch.empty(n, dtype=torch.int64, pin_memory=True)
			pos = torch.empty(n, dtype=torch.int64, pin_memory=True)
			off = 0
			for table_idx, first_page, last_page in allocation_info:
				first, last = first_page * page_size, last_page * page_size
				tidx[off:off + last - first].fill_(table_idx)
				torch.arange(first, last, out=pos[off:off + last - first])
				off += last - first
			page_table[tidx.to(page_table.device, non_blocking=True), pos.to(page_table.device, non_blocking=True)] = allocated
		m_cache._write_page_table = _write_page_table_twin
		log("MINI swap pagetable")
	if "presync-pt" in swaps:
		# real request, device sync right before the real _write_page_table (after _allocate / _page_to_token / alloc_swa)
		import freetoken.scheduler.cache as m_cache
		orig_wpt = m_cache._write_page_table

		def _wpt_presync(*a, **k):
			torch.cuda.synchronize()
			log("MINI sync before _write_page_table")
			return orig_wpt(*a, **k)
		m_cache._write_page_table = _wpt_presync
	clones = None
	to_variant = os.environ.get("MINI_TO_VARIANT", "")
	if to_variant == "mix":
		# per-buffer bisect: MINI_TO_CLONE=comma list of fs (free_slots), sf (_swa_free), mp (mapping) to swap for clones
		cm = llm.cache_manager
		sp = cm.swa_pool
		which = [x for x in os.environ.get("MINI_TO_CLONE", "").split(",") if x]
		with torch.inference_mode():
			clones = (
				cm.free_slots.clone() if "fs" in which else cm.free_slots,
				sp._swa_free.clone() if "sf" in which else sp._swa_free,
				sp.full_to_swa_index_mapping.clone() if "mp" in which else sp.full_to_swa_index_mapping,
			)
		log("MINI mix cloned=%s" % which)
	if to_variant in ("clonealloc", "smallalloc"):
		cm = llm.cache_manager
		sp = cm.swa_pool
		with torch.inference_mode():
			if to_variant == "clonealloc":
				clones = (cm.free_slots.clone(), sp._swa_free.clone(), sp.full_to_swa_index_mapping.clone())
			else:
				m = 4096
				clones = (cm.free_slots[:m].clone(), sp._swa_free[:m].clone(), torch.zeros(m, dtype=sp.full_to_swa_index_mapping.dtype, device=llm.device))
		log("MINI clones %s: %s" % (to_variant, [(tuple(c.shape), c.dtype) for c in clones]))
	if os.environ.get("MINI_PRESYNC"):
		# drain everything load / warmup / scheduler init queued before the request starts
		torch.cuda.synchronize()
		log("MINI presync")

	ns = int(os.environ.get("PROBE_NSENT", "60"))
	p = " ".join(f"Report P60-{i}: the survey team logged {(i*37)%991} observations near site {(i*13)%97}." for i in range(ns)) + " Reply with OK."
	ids = llm.tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
	if not isinstance(ids, list):
		ids = ids["input_ids"]
	log("MINI prompt %d tokens" % len(ids))

	# every path below ends in os._exit, which drops the AMD runtime log's buffered tail: flush it first
	real_exit = os._exit
	def _exit(code):
		time.sleep(3)
		real_exit(code)
	os._exit = _exit

	faulthandler.dump_traceback_later(int(os.environ.get("MINI_TIMEOUT", "60")), exit=True)
	t = time.time()
	if os.environ.get("MINI_MODE") == "ops":
		# only the first request's page-table write (scheduler/cache.py _write_page_table) and the
		# pageable copy that blocks behind it (attention/triton.py prepare_metadata), on the scheduler stream
		from freetoken.core import get_global_ctx
		page_table = get_global_ctx().page_table
		n = len(ids)
		log("MINI ops: page_table %s %s, n=%d, stream=%s" % (tuple(page_table.shape), page_table.dtype, n, torch.cuda.current_stream()))
		with torch.inference_mode():
			table_idx_host = torch.empty(n, dtype=torch.int64, pin_memory=True)
			positions_host = torch.empty(n, dtype=torch.int64, pin_memory=True)
			table_idx_host.fill_(0)
			torch.arange(0, n, out=positions_host)
			allocated = torch.arange(0, n, dtype=page_table.dtype, device=page_table.device)
			table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
			offsets = positions_host.to(page_table.device, non_blocking=True)
			page_table[table_idxs, offsets] = allocated
			prefix_lens = torch.tensor([0], dtype=torch.int32, device=page_table.device)
		faulthandler.cancel_dump_traceback_later()
		log("MINI OK %.1fs (ops)" % (time.time() - t))
		os._exit(0)
	if os.environ.get("MINI_MODE") == "stale":
		# stale-data test: kernels pre-fill the copy destinations with MINI_STALE_FILL (default out of range),
		# SDMA copies valid indices over them, index_put consumes them right after. Reading stale values
		# means an out-of-bounds write (fault, looks like a hang here); fill 0 is the harmless control
		from freetoken.core import get_global_ctx
		dev = llm.device
		page_table = get_global_ctx().page_table
		n = len(ids)
		fill = int(os.environ.get("MINI_STALE_FILL", str(1 << 40)))
		log("MINI stale fill=%d n=%d" % (fill, n))
		with torch.inference_mode():
			allocated = torch.arange(n, dtype=page_table.dtype, device=dev)
			dst_idx = torch.full((n,), fill, dtype=torch.int64, device=dev)
			dst_off = torch.full((n,), fill, dtype=torch.int64, device=dev)
			src_idx = torch.zeros(n, dtype=torch.int64).pin_memory()
			src_off = torch.arange(n, dtype=torch.int64).pin_memory()
			if not os.environ.get("MINI_STALE_NOCOPY"):  # NOCOPY: sanity check that reading the fill really faults
				dst_idx.copy_(src_idx, non_blocking=True)
				dst_off.copy_(src_off, non_blocking=True)
			page_table[dst_idx, dst_off] = allocated
			prefix_lens = torch.tensor([0], dtype=torch.int32, device=dev)
			torch.cuda.synchronize()
		faulthandler.cancel_dump_traceback_later()
		log("MINI OK %.1fs (stale fill=%d)" % (time.time() - t, fill))
		os._exit(0)
	if os.environ.get("MINI_MODE") == "sa":
		# standalone.py's tail on FreeToken's real buffers right after init, no request path:
		# input-id copy into the token pool, alloc_swa-style kernels with temporaries freed, page-table copies
		# reusing those blocks, index_put, 4 B pageable copy
		from freetoken.core import get_global_ctx
		dev = llm.device
		cm = llm.cache_manager
		sp = cm.swa_pool
		page_table = get_global_ctx().page_table
		n = len(ids)
		with torch.inference_mode():
			t = time.time()
			ids_host = torch.empty(n, dtype=torch.int32, pin_memory=True)
			ids_host.fill_(7)
			llm.token_pool[0, :n].copy_(ids_host, non_blocking=True)
			full = cm.free_slots[:n]
			sp.full_to_swa_index_mapping[full.to(torch.int64)] = sp._swa_free[:n].to(torch.int64)
			tidx_host = torch.empty(n, dtype=torch.int64, pin_memory=True)
			pos_host = torch.empty(n, dtype=torch.int64, pin_memory=True)
			tidx_host.fill_(0)
			torch.arange(0, n, out=pos_host)
			page_table[tidx_host.to(dev, non_blocking=True), pos_host.to(dev, non_blocking=True)] = full
			prefix = torch.tensor([0], dtype=torch.int32, device=dev)
			torch.cuda.synchronize()
		faulthandler.cancel_dump_traceback_later()
		log("MINI OK %.1fs (sa)" % (time.time() - t))
		os._exit(0)
	if os.environ.get("MINI_MODE") == "real3":
		# like real2, but pinned host tensors are created inline right before each copy, the way FreeToken does
		# (torch.empty(pin_memory=True) + fill, then .to(dev, non_blocking=True)), so torch's caching host allocator
		# records an event after each copy and queries events on each new pinned block, as in the hung log
		from freetoken.core import get_global_ctx
		dev = llm.device
		page_table = get_global_ctx().page_table
		sched = torch.cuda.current_stream()
		n = len(ids)
		log("MINI real3 n=%d" % n)

		# MINI_HOLD_PINNED=1: keep pinned tensors alive to the end (FreeToken frees them after the consumer kernel,
		# so torch records their events after it). MINI_EMPTY_CACHE=1: empty torch's device cache before the window
		# so the window's first allocation is a fresh hipMalloc, as in the hung log
		held = []
		gap_s = int(os.environ.get("MINI_GAP_US", "0")) / 1e6

		def gap():
			# MINI_GAP_US: busy-wait between window ops, like the Python work between them in the real path
			if gap_s:
				end = time.perf_counter() + gap_s
				while time.perf_counter() < end:
					pass

		def pinned(num, dtype, arange=False):
			h = torch.empty(num, dtype=dtype, pin_memory=True)
			if os.environ.get("MINI_HOLD_PINNED"):
				held.append(h)
			if arange:
				torch.arange(0, num, out=h)
			else:
				h.fill_(0)
			return h

		with torch.inference_mode():
			buf20 = torch.full((5 << 20,), -1, dtype=torch.int32, device=dev)
			tok = torch.zeros((5, 4096), dtype=torch.int64, device=dev)
			idx0 = torch.zeros(n, dtype=torch.int64, device=dev)
			ar = torch.arange(n, dtype=torch.int64, device=dev)
			allocated = torch.arange(n, dtype=page_table.dtype, device=dev)
			torch.cuda.synchronize()
			if os.environ.get("MINI_EMPTY_CACHE"):
				torch.cuda.empty_cache()
			t = time.time()
			ev = torch.cuda.Event()
			ev.record(llm.engine.stream)
			sched.wait_event(ev)
			gap()
			ids_dev = buf20[393216:393216 + n]
			ids_dev.copy_(pinned(n, torch.int32), non_blocking=True)  # 4604 B
			gap()
			a64 = ids_dev.to(torch.int64)  # direct_copy
			gap()
			b64 = ids_dev.to(torch.int64)  # direct_copy
			gap()
			tok[idx0, ar] = a64  # index_put<8>
			gap()
			table_idxs = pinned(n, torch.int64).to(dev, non_blocking=True)  # 9208 B
			gap()
			offsets = pinned(n, torch.int64, arange=True).to(dev, non_blocking=True)  # 9208 B
			gap()
			page_table[table_idxs, offsets] = allocated  # index_put<4>
			gap()
			pos32 = pinned(n, torch.int32).to(dev, non_blocking=True)  # 4604 B
			gap()
			loc64 = pinned(n, torch.int64, arange=True).to(dev, non_blocking=True)  # 9208 B
			gap()
			pos64 = pos32.to(torch.int64)  # direct_copy
			gap()
			r8 = pinned(1, torch.int64).to(dev, non_blocking=True)  # 8 B
			gap()
			c8 = pinned(1, torch.int64).to(dev, non_blocking=True)  # 8 B
			gap()
			out_loc = page_table[idx0, loc64]  # index<4>
			gap()
			prefix_lens = torch.tensor([0], dtype=torch.int32, device=dev)  # 4 B pageable, blocks on a wedge
			torch.cuda.synchronize()
		faulthandler.cancel_dump_traceback_later()
		log("MINI OK %.1fs (real3)" % (time.time() - t))
		os._exit(0)
	if os.environ.get("MINI_MODE") == "real":
		# kernel-for-kernel copy of the hung 2-layer run's request section (level-4 log), all on the scheduler stream.
		# Every tensor that needs a device kernel to build is made before the section (and synced), so between the
		# first copy and the blocking 4-byte copy only the real request's kernels run: direct_copy x2, index_put<8>,
		# index_put<4>, direct_copy, index<4>
		from freetoken.core import get_global_ctx
		dev = llm.device
		page_table = get_global_ctx().page_table
		sched = torch.cuda.current_stream()
		n = len(ids)
		log("MINI real n=%d" % n)
		with torch.inference_mode():
			buf20 = torch.full((5 << 20,), -1, dtype=torch.int32, device=dev)
			tok = torch.zeros((5, 4096), dtype=torch.int64, device=dev)
			idx0 = torch.zeros(n, dtype=torch.int64, device=dev)
			ar = torch.arange(n, dtype=torch.int64, device=dev)
			allocated = torch.arange(n, dtype=page_table.dtype, device=dev)
			host = {k: torch.zeros(n, dtype=d).pin_memory() for k, d in (("ids", torch.int32), ("tidx", torch.int64), ("pos32", torch.int32))}
			host["off"] = torch.arange(n, dtype=torch.int64).pin_memory()
			host["loc"] = torch.arange(n, dtype=torch.int64).pin_memory()
			host["r8"] = torch.zeros(1, dtype=torch.int64).pin_memory()
			host["c8"] = torch.zeros(1, dtype=torch.int64).pin_memory()
			torch.cuda.synchronize()
			t = time.time()
			ev = torch.cuda.Event()
			ev.record(llm.engine.stream)
			sched.wait_event(ev)
			ids_dev = buf20[393216:393216 + n]
			ids_dev.copy_(host["ids"], non_blocking=True)  # 4604 B
			a64 = ids_dev.to(torch.int64)  # direct_copy
			b64 = ids_dev.to(torch.int64)  # direct_copy
			tok[idx0, ar] = a64  # index_put<8>
			table_idxs = host["tidx"].to(dev, non_blocking=True)  # 9208 B
			offsets = host["off"].to(dev, non_blocking=True)  # 9208 B
			page_table[table_idxs, offsets] = allocated  # index_put<4>
			pos32 = host["pos32"].to(dev, non_blocking=True)  # 4604 B
			loc64 = host["loc"].to(dev, non_blocking=True)  # 9208 B
			pos64 = pos32.to(torch.int64)  # direct_copy
			r8 = host["r8"].to(dev, non_blocking=True)  # 8 B
			c8 = host["c8"].to(dev, non_blocking=True)  # 8 B
			out_loc = page_table[idx0, loc64]  # index<4>
			prefix_lens = torch.tensor([0], dtype=torch.int32, device=dev)  # 4 B pageable, blocks on a wedge
			torch.cuda.synchronize()
		faulthandler.cancel_dump_traceback_later()
		log("MINI OK %.1fs (real)" % (time.time() - t))
		os._exit(0)
	if os.environ.get("MINI_MODE") == "trace":
		# the request section of the hung c3 HIP trace, op for op in torch, steps picked by MINI_STEPS
		from freetoken.core import get_global_ctx
		steps = os.environ.get("MINI_STEPS", "abcdefg")
		dev = llm.device
		page_table = get_global_ctx().page_table
		sched = torch.cuda.current_stream()
		n = len(ids)
		log("MINI trace steps=%s n=%d sched=%s engine=%s" % (steps, n, sched, llm.engine.stream))

		def pinned(num, dtype):
			return torch.zeros(num, dtype=dtype, pin_memory=True)

		with torch.inference_mode():
			if "a" in steps:
				ev = torch.cuda.Event()
				ev.record(llm.engine.stream)
				sched.wait_event(ev)
			if "b" in steps:
				ids_dev = pinned(n, torch.int32).to(dev, non_blocking=True)
			if "B" in steps:
				# like the real run: SDMA copy into a block a compute kernel just filled (scheduler-init zeros)
				buf = torch.zeros(20 << 20, dtype=torch.uint8, device=dev)
				ids_dev = buf[1572864:1572864 + 4 * n].view(torch.int32)
				ids_dev.copy_(pinned(n, torch.int32), non_blocking=True)
			if "c" in steps:
				small = torch.zeros(3 * 128, dtype=torch.int32, device=dev)
				for _ in range(3):
					small.add_(1)
			if "d" in steps:
				table_idxs = pinned(n, torch.int64).to(dev, non_blocking=True)
				offsets = torch.arange(n, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
				page_table[table_idxs, offsets] = torch.arange(n, dtype=page_table.dtype, device=dev)
			if "e" in steps:
				pos = pinned(n, torch.int32).to(dev, non_blocking=True)
				loc = pinned(n, torch.int64).to(dev, non_blocking=True)
				out_loc = page_table[torch.zeros_like(loc), loc]
			if "f" in steps:
				a8 = pinned(1, torch.int64).to(dev, non_blocking=True)
				b8 = pinned(1, torch.int64).to(dev, non_blocking=True)
			if "g" in steps:
				prefix_lens = torch.tensor([0], dtype=torch.int32, device=dev)
			torch.cuda.synchronize()
		faulthandler.cancel_dump_traceback_later()
		log("MINI OK %.1fs (trace %s)" % (time.time() - t, steps))
		os._exit(0)
	with torch.inference_mode():
		out = llm.generate([ids], SamplingParams(max_tokens=1))
	faulthandler.cancel_dump_traceback_later()
	log("MINI OK %.1fs tokens=%s" % (time.time() - t, out[0]["token_ids"]))
	os._exit(0)


if __name__ == "__main__":
	main()
