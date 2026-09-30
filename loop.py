"""Run standalone.py in N fresh processes and count hangs.

usage: python loop.py [N=20]      (env passes through, e.g. PAL_DISABLE_SDMA=1 for the control)
Each hung run's full output (faulthandler stack included) is written to hangs/hung-<run>.txt for attaching.
"""
import os
import subprocess
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
here = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(here, "standalone.py")
hang_dir = os.path.join(here, "hangs")
hung = 0
for i in range(1, n + 1):
	r = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=120)
	out = r.stdout + r.stderr
	if "SA OK" in out:
		res = "ok"
	elif "Timeout" in out:
		hung += 1
		os.makedirs(hang_dir, exist_ok=True)
		dump = os.path.join(hang_dir, "hung-%d.txt" % i)
		with open(dump, "w", encoding="utf-8") as f:
			f.write(out)
		res = "HUNG, output in %s" % dump
	else:
		raise RuntimeError("run %d: neither pass nor hang, rc=%d\n%s" % (i, r.returncode, out))
	print("run %d/%d %s" % (i, n, res), flush=True)
print("hung %d of %d fresh processes" % (hung, n), flush=True)
