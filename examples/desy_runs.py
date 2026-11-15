"""DESY's published tuning runs on ARES (simulation and the real machine), as small arrays.

Source: Kaiser et al., "Learning to Do or Learning While Doing: Reinforcement Learning and
Bayesian Optimisation for Online Continuous Tuning", data at Zenodo record 7853721
(doi 10.5281/zenodo.7853721, CC BY 4.0): simulation.zip (5.4 GB) and real.zip (17.2 GB).

Each run is one pickled episode (recorded_episode_1.pkl): the observations (beam on the
screen, magnet settings, target; in simulation also the incoming beam and misalignments),
the actions and per-step infos. The real-machine files also hold a screen image per step,
which is why real.zip is large. This script never stores the archives. It reads the zip
directory and then one member at a time over HTTP range requests, unpickles it with an
allow-list (numpy arrays, numpy scalars and datetime only, so no code from the file can
run), keeps the small arrays and drops the images.

    python examples/desy_runs.py list                      # groups in both archives
    python examples/desy_runs.py fetch simulation bo rl random nelder-mead
    python examples/desy_runs.py fetch real bo bo_ucb rl rl_bo_takeover_0.00015 [--workers 4]

Output: ~/.cache/metalsq/desy_runs/<archive>/<group>__<run>.npz with
    magnets (T+1, 5)   settings (k_Q1, k_Q2, theta_CV, k_Q3, theta_CH) in 1/m^2 and rad
    beam    (T+1, 4)   measured (mu_x, sigma_x, mu_y, sigma_y) in m
    target  (4,)       target beam in m
    incoming, misalignments   (simulation only)
and anything else in the infos that is small. load_runs(archive, group) returns them.
"""
import datetime
import glob
import io
import json
import os
import pickle
import sys
import time
import urllib.request
import zipfile
from multiprocessing import Pool

import numpy as np

URL = "https://zenodo.org/records/7853721/files/{}.zip?download=1"
CACHE = os.path.expanduser("~/.cache/metalsq/desy_runs")
ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("datetime", "datetime"),
    ("datetime", "timedelta"), ("collections", "OrderedDict"),
}
KEEP_BYTES = 4096          # arrays larger than this (the screen images) are dropped


class RangeFile:
    """A seekable read-only file over HTTP range requests. It keeps one streaming response
    open and serves sequential reads from it, so reading a zip member costs one or two
    requests however large it is (Zenodo allows about 130 requests a minute)."""
    SKIP = 1 << 20          # a forward seek shorter than this is read through, not reopened

    def __init__(self, url):
        self.url, self.pos, self.stream, self.stream_pos = url, 0, None, -1
        r = self._open(0, 0)
        self.size = int(r.headers["Content-Range"].split("/")[1])
        r.close()

    def _open(self, a, b=None):
        req = urllib.request.Request(self.url, headers={
            "Range": f"bytes={a}-{'' if b is None else b}", "User-Agent": "python-urllib"})
        for attempt in range(8):
            try:
                r = urllib.request.urlopen(req, timeout=300)
                if r.status != 206:
                    raise OSError(f"server ignored the range request (status {r.status})")
                return r
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(10 * (attempt + 1))

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _align(self):
        gap = self.pos - self.stream_pos
        if self.stream is not None and 0 < gap <= self.SKIP:
            self.stream_pos += len(self.stream.read(gap))
        if self.stream is None or self.stream_pos != self.pos:
            if self.stream is not None:
                self.stream.close()
            self.stream, self.stream_pos = self._open(self.pos), self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        retries = 0
        while n > 0 and self.pos < self.size:
            self._align()
            try:
                chunk = self.stream.read(n)
            except OSError:
                chunk = b""
            if not chunk:                                   # dropped connection: reopen here
                self.stream.close()
                self.stream = None
                retries += 1
                if retries > 8:
                    raise OSError("connection kept dropping")
                time.sleep(5 * retries)
                continue
            out += chunk
            self.pos += len(chunk)
            self.stream_pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        return super().find_class(module, name)


def directory(archive):
    """[(member, compressed size, offset)] of the episode files, cached as json."""
    path = os.path.join(CACHE, f"{archive}.members.json")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        z = zipfile.ZipFile(RangeFile(URL.format(archive)))
        members = sorted(((i.filename, i.compress_size, i.header_offset) for i in z.infolist()
                          if i.filename.endswith(".pkl") and not i.filename.startswith("__MACOSX")),
                         key=lambda m: m[2])                  # archive order, so neighbours share blocks
        json.dump(members, open(path, "w"))
    return json.load(open(path))


def _small(v):
    """Keep numbers and small arrays; drop images and anything else that is big."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, datetime.datetime):
        return v.isoformat()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v if v.nbytes <= KEEP_BYTES else None
    if isinstance(v, (list, tuple)) and len(v) <= 64:
        return [_small(x) for x in v]
    return None


def episode_arrays(ep):
    obs = ep["observations"]
    out = {"magnets": np.stack([np.asarray(o["magnets"], float) for o in obs]),
           "beam": np.stack([np.asarray(o["beam"], float) for o in obs]),
           "target": np.asarray(obs[0]["target"], float)}
    for key in ("incoming", "misalignments"):
        if key in obs[0]:
            out[key] = np.asarray(obs[0][key], float)
    if "actions" in ep:
        out["actions"] = np.asarray(ep["actions"], float)
    infos = ep.get("infos") or []
    keys = sorted({k for info in infos for k in info})
    meta = {"info_keys": keys, "steps_taken": _small(ep.get("steps_taken")),
            "t_start": _small(ep.get("t_start")), "t_end": _small(ep.get("t_end"))}
    for k in keys:
        vals = [_small(info.get(k)) for info in infos]
        if all(isinstance(v, (bool, int, float)) for v in vals):
            out[f"info_{k}"] = np.asarray(vals, float)
        elif all(isinstance(v, np.ndarray) for v in vals) and len({v.shape for v in vals}) == 1:
            out[f"info_{k}"] = np.stack(vals).astype(float)
        elif vals and isinstance(vals[0], (int, float, str, list)):
            meta[f"first_{k}"] = vals[0] if not isinstance(vals[0], np.ndarray) else None
    out["meta"] = np.asarray(json.dumps(meta, default=str))
    return out


_ZIPS = {}


def _fetch(args):
    archive, member = args
    group, run = member.split("/")[1], member.split("/")[2]
    run = os.path.splitext(run)[0]
    dest = os.path.join(CACHE, archive, f"{group}__{run}.npz")
    if os.path.exists(dest):
        return member, "have", 0.0
    t0 = time.time()
    try:
        if archive not in _ZIPS:                            # one open archive per process
            _ZIPS[archive] = zipfile.ZipFile(RangeFile(URL.format(archive)))
        z = _ZIPS[archive]
        with z.open(member) as f:
            ep = SafeUnpickler(io.BufferedReader(f, 1 << 22)).load()
        if not (isinstance(ep, dict) and "observations" in ep):
            return member, "skipped: not an episode", time.time() - t0
        arrays = episode_arrays(ep)
        del ep
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        np.savez_compressed(dest + ".tmp.npz", **arrays)
        os.replace(dest + ".tmp.npz", dest)
        return member, "ok", time.time() - t0
    except Exception as e:                                   # keep going; report at the end
        return member, f"failed: {type(e).__name__}: {e}", time.time() - t0


def fetch(archive, groups, workers=4, limit=None):
    members = [(m, c) for m, c, _ in directory(archive)
               if m.split("/")[1] in groups and m.count("/") == 3]
    members = members[:limit] if limit else members
    total = sum(c for _, c in members)
    print(f"{archive}: {len(members)} runs, {total / 1e9:.2f} GB to read", flush=True)
    done, t0, failed = 0, time.time(), []
    with Pool(workers) as pool:
        jobs = [(archive, m) for m, _ in members]
        for member, status, dt in pool.imap_unordered(_fetch, jobs, chunksize=max(1, len(jobs) // (workers * 8))):
            done += 1
            if status.startswith(("failed", "skipped")):
                failed.append((member, status))
            if status != "have" and (done % 10 == 0 or dt > 20 or status != "ok"):
                print(f"  {done}/{len(members)} {member.split('/', 1)[1]}: {status} ({dt:.0f}s, "
                      f"{time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"done: {done - len(failed)} ok, {len(failed)} not read", flush=True)
    for member, status in failed:
        print("  ", member, status)


def load_runs(archive, group):
    """[dict of arrays] for one group, sorted by run name."""
    runs = []
    for path in sorted(glob.glob(os.path.join(CACHE, archive, f"{group}__*.npz"))):
        with np.load(path, allow_pickle=False) as d:
            run = {k: d[k] for k in d.files}
        run["meta"] = json.loads(str(run["meta"]))
        run["name"] = os.path.basename(path)[len(group) + 2:-4]
        runs.append(run)
    return runs


if __name__ == "__main__":
    if sys.argv[1] == "list":
        for archive in ("simulation", "real"):
            groups = {}
            for m, c, _ in directory(archive):
                g = groups.setdefault(m.split("/")[1], [0, 0])
                g[0] += 1
                g[1] += c
            for name, (n, c) in sorted(groups.items()):
                print(f"{archive:10s} {name:36s} {n:5d} files {c / 1e6:9.1f} MB")
    elif sys.argv[1] == "fetch":
        args = sys.argv[2:]
        workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 4
        limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
        names = [a for i, a in enumerate(args[1:]) if not a.startswith("--")
                 and args[i] not in ("--workers", "--limit")]
        fetch(args[0], names, workers, limit)
