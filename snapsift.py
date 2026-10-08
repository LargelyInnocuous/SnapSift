#!/usr/bin/env python3
"""
SnapSift - browse files that still live in ZFS snapshots but are gone from the
live filesystem, and see exactly what you would lose by destroying snapshots.

Single file, Python 3.9+ standard library only. Run it on the NAS (as root, so
`zfs diff` works) and open the printed URL in a browser:

    sudo python3 snapsift.py                      # all mounted datasets
    sudo python3 snapsift.py --exclude 'scratch'  # extra exclude regex
    python3 snapsift.py --demo                    # synthetic data, any OS

SnapSift only reads, with one exception you trigger yourself: "Recover" copies
files out of snapshots into a folder you choose (never overwriting anything).
Otherwise it reads snapshots, runs `zfs diff`, and (only when you click
"Estimate") `zfs destroy -n`, which is a dry run. Destroying snapshots is done
by you, with commands it generates.
"""

import argparse
import collections
import faulthandler
import gzip
import io
import json
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VERSION = "1.1"

DEFAULT_EXCLUDE = r"^boot-pool(/|$)|/\.system(/|$)|/ix-applications(/|$)|/\.ix-|/ix-apps(/|$)"

KIND_DELETED, KIND_MOVED, KIND_REPLACED = 0, 1, 2


LOG = True
LOG_FILE = None                                 # set in main(): next to the cache
_log_lock = threading.Lock()
_log_lines = collections.deque(maxlen=5000)     # served to the browser's Console view
_log_seq = 0


def log(msg):
    """Timestamped line to the console (the tmux window), the log file and the browser's
    Console view. One write per line under a lock, so parallel diffs never interleave."""
    global _log_seq
    if LOG:
        line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg + "\n"
        with _log_lock:
            _log_seq += 1
            _log_lines.append((_log_seq, line.rstrip("\n")))
            if LOG_FILE:
                try:
                    with open(LOG_FILE, "a", encoding="utf-8", errors="backslashreplace") as f:
                        f.write(line)
                except OSError:
                    pass
            try:
                # Consoles that aren't UTF-8 (or filenames that aren't) must never break a scan:
                # anything unprintable is written as an escape instead.
                out = sys.stdout
                out.flush()
                enc = getattr(out, "encoding", None) or "utf-8"
                data = line.encode(enc, "backslashreplace")
                if hasattr(out, "buffer"):
                    out.buffer.write(data)
                    out.buffer.flush()
                else:
                    out.write(data.decode(enc, "replace"))
                    out.flush()
            except Exception:  # noqa: BLE001 - logging is best-effort
                pass


STARTUP_WARNINGS = []  # shown on the Scan tab


def fs_source(path):
    """(mount source, fs type) of the filesystem that holds `path`, from /proc/mounts."""
    best = ("", None, None)
    try:
        real = os.path.realpath(path)
        with open("/proc/mounts", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                src, mnt, fstype = parts[0], parts[1].replace("\\040", " "), parts[2]
                if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best[0]):
                    best = (mnt, src, fstype)
    except OSError:
        pass
    return best[1], best[2]


def warn_if_boot_pool(path):
    """TrueNAS's boot pool is a small OS drive; big diff caches and records can fill it."""
    src, fstype = fs_source(path)
    if src and fstype == "zfs" and src.split("/")[0] in ("boot-pool", "freenas-boot"):
        msg = (f"SnapSift's cache folder {os.path.dirname(path)} is on the boot pool ({src}), TrueNAS's "
               "small OS drive. Large scans and saved records can fill it. Restart with "
               "--cache /mnt/<pool>/<dataset>/snapsift/snapsift.json.gz to keep everything on a data pool.")
        STARTUP_WARNINGS.append(msg)
        log("WARNING " + msg)


def fmt_dur(sec):
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m {sec % 60:02d}s"
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m"


# Progress-panel task kinds -> their step-strip code, and console names.
STEP_CODE = {"zfs": "d", "tree": "t", "save": "s", "process": "p", "resync": "d", "stat": "r", "list": "l",
             "load": "p"}
KIND_NAME = {"zfs": "ZFS DIFF", "tree": "FOLDER LISTING", "save": "SAVING", "process": "PROCESSING",
             "load": "LOADING FROM CACHE",
             "resync": "ZFS DIFF (vs live)", "stat": "FILE DETAILS", "list": "LISTING"}


class Task:
    """One visible unit of work: shown in the progress panel and logged to the console.
    Calling it (task(n)) updates its counter; zfs-diff tasks also record the zfs pid so the
    panel can show that process's CPU and disk activity."""
    _ids = iter(range(1, 1 << 62))

    def __init__(self, kind, label, unit="", total=None):
        self.id = next(Task._ids)
        self.kind, self.label, self.unit, self.total = kind, label, unit, total
        self.n = 0
        self.started = self.last = time.time()
        self.rep_n, self.rep_t = 0, self.started  # last periodic console report
        self.pid = None
        self.tid = threading.get_native_id()
        log(f"start  {KIND_NAME[kind]:<18} {label}")

    def __call__(self, n):
        if n != self.n:
            self.n, self.last = n, time.time()

    def activity(self, who="ui"):
        if self.pid:  # zfs diff: its own process (kernel work is accounted to it)
            return proc_activity(f"/proc/{self.pid}", who)
        return proc_activity(f"/proc/{os.getpid()}/task/{self.tid}", who)  # the thread doing the work

    def report(self):
        """Periodic console line: progress since the last report, so speed changes are visible."""
        now = time.time()
        dt = now - self.rep_t
        parts = [f"running {fmt_dur(now - self.started)}"]
        if self.unit:
            delta = self.n - self.rep_n
            pct = f" of {self.total:,} ({self.n / self.total:.0%})" if self.total else ""
            parts.insert(0, f"{self.n:,}{pct} {self.unit}, +{delta:,} in last {fmt_dur(dt)} "
                            f"({delta / dt:,.0f}/s)" if dt > 0 else f"{self.n:,} {self.unit}")
            if now - self.last >= 60:
                parts.append(f"count unchanged for {fmt_dur(now - self.last)}")
        act = self.activity("report")  # averaged over the whole interval since the last report
        if act:
            if act["cpu"] is not None:
                parts.append(f"CPU {act['cpu']}%")
            if act["read"] is not None:
                parts.append(f"disk read {act['read'] / 1e6:,.1f} MB/s")
            if act["state"] == "D":
                parts.append("waiting on disk")
        self.rep_n, self.rep_t = self.n, now
        log(f"status {KIND_NAME[self.kind]:<18} {self.label}  ({', '.join(parts)})")

    def finish(self, failed=False):
        took = fmt_dur(time.time() - self.started)
        count = f"{self.n:,} {self.unit}, " if self.unit else ""
        log(f"{'FAILED' if failed else 'done  '} {KIND_NAME[self.kind]:<18} {self.label}  ({count}{took})")


_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_samples = {}


def proc_activity(path, who="ui"):
    """CPU %, disk read bytes/s and state letter for a /proc/<pid> or /proc/<pid>/task/<tid>
    entry, measured since the previous call by the same `who`. None where /proc isn't available."""
    try:
        with open(path + "/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        state, cpu = fields[0], (int(fields[11]) + int(fields[12])) / _CLK
        rd = None
        try:
            with open(path + "/io") as f:
                for line in f:
                    if line.startswith("read_bytes:"):
                        rd = int(line.split()[1])
        except OSError:
            pass
    except (OSError, ValueError, IndexError):
        return None
    now = time.time()
    key = (path, who)
    prev = _samples.get(key)
    if prev and now - prev[0] < 0.5:
        return prev[3]
    res = {"state": state, "cpu": None, "read": None}
    if prev:
        dt = now - prev[0]
        res["cpu"] = round((cpu - prev[1]) / dt * 100)
        if rd is not None and prev[2] is not None:
            res["read"] = max(0, int((rd - prev[2]) / dt))
    _samples[key] = (now, cpu, rd, res)
    return res


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

_OCTAL = re.compile(rb"\\0([0-7]{3})")


def zfs_unescape(raw: bytes) -> str:
    """zfs diff prints unusual bytes (space, tab, backslash, non-ASCII) as \\0ooo."""
    return os.fsdecode(_OCTAL.sub(lambda m: bytes([int(m.group(1), 8)]), raw))


def parse_zfs_diff_line(line: bytes):
    """Parse one line of `zfs diff -F -H`. Returns (change, ftype, path, newpath|None)."""
    parts = line.rstrip(b"\r\n").split(b"\t")
    if len(parts) < 3:
        return None
    change = parts[0].decode("ascii", "replace")
    ftype = parts[1].decode("ascii", "replace")
    path = zfs_unescape(parts[2])
    newpath = zfs_unescape(parts[3]) if len(parts) > 3 else None
    return change, ftype, path, newpath


XATTR_DIR = "<xattrdir>"


def is_xattr_path(p):
    """zfs diff reports changes to directory-style extended attributes (e.g. Samba's DOSATTRIB)
    as '<file>/<xattrdir>/<attr>'. Those aren't files: the attributes travel with their file."""
    return bool(p) and XATTR_DIR in p.split("/")


def clean_result(res):
    """Drop xattr pseudo-entries from a saved result (scans and records made before they were
    filtered). Renumbers entry ids. Returns True if anything changed."""
    dirs, keep = res["dirs"], []
    for e in res["entries"]:
        path = f"{dirs[e[2]][1]}/{e[3]}" if dirs[e[2]][1] else e[3]
        if is_xattr_path(path) or (len(e) > 9 and is_xattr_path(e[9])):
            continue
        keep.append(e)
    if len(keep) == len(res["entries"]):
        return False
    for i, e in enumerate(keep):
        e[0] = i
    res["entries"] = keep
    return True


def rel_to(mountpoint: str, path: str):
    mp = mountpoint.rstrip("/")
    if path.startswith(mp + "/"):
        return path[len(mp) + 1:]
    return None


class DiffStalled(RuntimeError):
    pass


class Backend:
    name = "base"
    # True when diff_lines() can compare two snapshots; the scanner then diffs
    # consecutive snapshots instead of every snapshot against live (see Scanner).
    incremental = False

    def list_datasets(self):
        """[{name, mountpoint, used, snaps:[{name, creation, used, referenced}]}] (snaps oldest first)."""
        raise NotImplementedError

    def diff(self, ds, snap, cancel, tick=None):
        """Yield (kind, ftype, rel, newrel, statinfo|None) for things in `snap` missing from live."""
        raise NotImplementedError

    def diff_lines(self, ds, a, b, cancel, tick=None, stall=None):
        """Yield raw (change, ftype, rel, newrel) between snapshot a and snapshot b (None = live)."""
        return self.tree_diff(ds, a, b, cancel, tick)

    def live_exists(self, ds, rel):
        return os.path.lexists(self.live_path(ds, rel))

    # -- directory-tree comparison (fallback for when `zfs diff` crawls) -----
    def tree_root(self, ds, snap):
        return ds["mountpoint"] if snap is None else self.snap_root(ds, snap)

    def listdir(self, root, rel):
        """[(name, inode, is_dir)] for one directory. Inode = ZFS object number, which is the
        same for a file in every snapshot and in live, so it identifies renames."""
        path = os.path.join(root, *rel.split("/")) if rel else root
        dev = self._devs.get(root) if hasattr(self, "_devs") else None
        if dev is None:
            dev = os.stat(root).st_dev
            self.__dict__.setdefault("_devs", {})[root] = dev
        out = []
        with os.scandir(path) as it:
            for e in it:
                if not rel and e.name == ".zfs":
                    continue
                is_dir = e.is_dir(follow_symlinks=False)
                if is_dir and os.lstat(e.path).st_dev != dev:
                    out.append((e.name, -e.inode(), True))  # child dataset mount: don't descend
                    continue
                out.append((e.name, e.inode(), is_dir))
        return out

    def file_sig(self, root, rel):
        try:
            st = os.lstat(os.path.join(root, *rel.split("/")) if rel else root)
            return st.st_size, st.st_mtime_ns
        except OSError:
            return None

    def tree_diff(self, ds, a, b, cancel, tick=None):
        """Same output as `zfs diff a b` for files, computed by listing both trees.
        Only folder listings are read (no per-file stat), and the cost grows linearly with
        the number of files, unlike zfs diff's per-file name lookups."""
        roots = (self.tree_root(ds, a), self.tree_root(ds, b))
        # Entries that differ, per side: inode -> (rel, is_dir, parent_inode, name).
        # Identical (same name + inode) entries in folders reachable by the same path are
        # never recorded, so memory scales with what changed, not with the tree size.
        sides = ({}, {})
        expanded = (set(), set())   # folders whose full listing was recorded, per side
        paired = set()              # folders present on both sides, already reconciled
        listed = [0]

        def join(p, n):
            return f"{p}/{n}" if p else n

        def ls(side, rel):
            if cancel.is_set():
                raise RuntimeError("cancelled")
            listed[0] += 1
            if tick and listed[0] % 50 == 0:
                tick(listed[0])
            return self.listdir(roots[side], rel)

        def compare(pa, pb, ka, kb):
            stack = [(pa, pb, ka, kb)]
            while stack:
                x, y, kx, ky = stack.pop()
                A = {n: (i, d) for n, i, d in ls(0, x)}
                B = {n: (i, d) for n, i, d in ls(1, y)}
                # A file deleted and recreated under the same name can get the old object number
                # back. Its folder's mtime changes when that happens, so only in changed folders
                # are same-name/same-inode files checked by size + mtime.
                touched = self.file_sig(roots[0], x) != self.file_sig(roots[1], y)
                for n, (i, d) in A.items():
                    if B.get(n) == (i, d) and (d or not touched or
                                               self.file_sig(roots[0], join(x, n)) ==
                                               self.file_sig(roots[1], join(y, n))):
                        if d and i >= 0:
                            stack.append((join(x, n), join(y, n), i, i))
                    else:
                        sides[0][i] = (join(x, n), d, kx, n)
                for n, (i, d) in B.items():
                    if A.get(n) != (i, d) or (not d and i in sides[0] and sides[0][i][0] == join(x, n)):
                        sides[1][i] = (join(y, n), d, ky, n)

        def expand(side, ino):
            expanded[side].add(ino)
            p = sides[side][ino][0]
            for n, i, d in ls(side, p):
                sides[side][i] = (join(p, n), d, ino, n)

        compare("", "", "/", "/")
        # Folders that moved, appeared or vanished hide their contents from the path walk.
        # Reconcile until stable: a folder on both sides is compared (or fully listed on both
        # sides if one side already was); a folder on one side only is fully listed, so files
        # moved into or out of it can be matched up by inode.
        changed = True
        while changed:
            changed = False
            A, B = sides
            for ino in [i for i, v in A.items() if v[1] and i >= 0 and i not in paired]:
                if ino in B and B[ino][1]:
                    paired.add(ino)
                    changed = True
                    if ino in expanded[0] or ino in expanded[1]:
                        for sd in (0, 1):
                            if ino not in expanded[sd]:
                                expand(sd, ino)
                    else:
                        compare(A[ino][0], B[ino][0], ino, ino)
            for sd in (0, 1):
                other = sides[1 - sd]
                for ino in [i for i, v in sides[sd].items()
                            if v[1] and i >= 0 and i not in paired and i not in expanded[sd]
                            and not (i in other and other[i][1])]:
                    expand(sd, ino)
                    changed = True

        # Same semantics as zfs diff: per object, '-' gone, '+' new, 'R' own parent/name changed.
        out = []
        A, B = sides
        for ino, (pa, da, ka, na) in A.items():
            other = B.get(ino)
            if other is None:
                if not da:
                    out.append(("-", "F", pa, None))
                continue
            pb, db, kb, nb = other
            if da != db or (not da and self.file_sig(roots[0], pa) != self.file_sig(roots[1], pb)):
                # same inode number, different object (ZFS reuses freed object numbers)
                if not da:
                    out.append(("-", "F", pa, None))
                if not db:
                    out.append(("+", "F", pb, None))
            elif (ka, na) != (kb, nb) and pa != pb:
                # Note: if ZFS gave a deleted folder's object number to an unrelated new folder,
                # this reports a rename. Folders have no reliable identity check beyond the
                # number, and real renames are far more common than that kind of reuse.
                out.append(("R", "/" if da else "F", pa, pb))
        for ino, (pb, db, kb, nb) in B.items():
            if ino not in A and not db:
                out.append(("+", "F", pb, None))
        if tick:
            tick(listed[0])
        return out

    def snap_root(self, ds, snap):
        return os.path.join(ds["mountpoint"], ".zfs", "snapshot", snap["name"])

    def snap_path(self, ds, snap, rel):
        return os.path.join(self.snap_root(ds, snap), *rel.split("/"))

    def live_path(self, ds, rel):
        return os.path.join(ds["mountpoint"], *rel.split("/"))

    def stat(self, path):
        st = os.lstat(path)
        return st.st_size, int(st.st_mtime)

    def estimate(self, ds, snap_spec):
        return None


class ZfsBackend(Backend):
    name = "zfs-diff"

    def __init__(self, exclude):
        self.exclude = re.compile(exclude) if exclude else None
        if not shutil.which("zfs"):
            raise SystemExit("`zfs` not found on PATH. Run this on the NAS, or use --demo.")

    def _zfs(self, *args):
        r = subprocess.run(["zfs", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise RuntimeError(f"zfs {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout.decode(errors="surrogateescape")

    def list_datasets(self):
        out = []
        by_name = {}
        for line in self._zfs("list", "-H", "-p", "-t", "filesystem",
                              "-o", "name,mountpoint,mounted,used").splitlines():
            name, mp, mounted, used = line.split("\t")
            if mounted != "yes" or not mp.startswith("/"):
                continue
            if self.exclude and self.exclude.search(name):
                continue
            d = {"name": name, "mountpoint": mp, "used": int(used), "snaps": []}
            out.append(d)
            by_name[name] = d
        for line in self._zfs("list", "-H", "-p", "-t", "snapshot", "-s", "creation",
                              "-o", "name,creation,used,referenced,guid").splitlines():
            full, creation, used, refd, guid = line.split("\t")
            dsname, _, snapname = full.partition("@")
            if dsname in by_name:
                by_name[dsname]["snaps"].append({
                    "name": snapname, "creation": int(creation),
                    "used": int(used), "referenced": int(refd), "guid": guid})
        return [d for d in out if d["snaps"]]

    incremental = True

    def diff(self, ds, snap, cancel, tick=None):
        for change, ftype, rel, newrel in self.diff_lines(ds, snap, None, cancel, tick):
            if ftype == "/" or is_xattr_path(rel) or is_xattr_path(newrel):
                continue  # directories (their files are listed individually) and extended attributes
            if change == "-":
                yield KIND_DELETED, ftype, rel, None, None
            elif change == "R" and newrel:
                yield KIND_MOVED, ftype, rel, newrel, None

    def diff_lines(self, ds, a, b, cancel, tick=None, stall=None):
        """`zfs diff` between a and b (None = live). If `stall` seconds pass with fewer than
        100 new lines, zfs diff is killed and DiffStalled raised so the caller can fall back."""
        mp = ds["mountpoint"]
        args = ["zfs", "diff", "-F", "-H", f"{ds['name']}@{a['name']}"]
        if b is not None:
            args.append(f"{ds['name']}@{b['name']}")
        with tempfile.TemporaryFile() as errf:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errf)
            read_all = False
            if isinstance(tick, Task):
                tick.pid = proc.pid  # lets the progress panel show zfs's own CPU and disk reads
            count = [0]
            stalled = threading.Event()

            def watchdog():
                mark_t, mark_n = time.time(), 0
                while proc.poll() is None and not cancel.is_set():
                    time.sleep(5)
                    if count[0] - mark_n >= 100:
                        mark_t, mark_n = time.time(), count[0]
                    elif time.time() - mark_t > stall:
                        stalled.set()
                        proc.kill()
                        return
            if stall:
                threading.Thread(target=watchdog, daemon=True).start()
            try:
                for n, line in enumerate(proc.stdout, 1):
                    count[0] = n
                    if cancel.is_set():
                        proc.kill()
                        return
                    if tick and n % 100 == 0:
                        tick(n)
                    p = parse_zfs_diff_line(line)
                    if not p:
                        continue
                    change, ftype, path, newpath = p
                    rel = rel_to(mp, path)
                    if rel is None:
                        continue
                    newrel = (rel_to(mp, newpath) or newpath) if newpath else None
                    if is_xattr_path(rel) or is_xattr_path(newrel):
                        continue  # extended attributes, not files
                    yield change, ftype, rel, newrel
                read_all = True
            finally:
                # After its last line zfs diff still has cleanup to do (e.g. releasing the temporary
                # snapshot it takes when diffing against live), so let it exit by itself. Only kill
                # it when we stop reading early: cancel, an exception, or the stall watchdog.
                if not read_all and proc.poll() is None:
                    proc.kill()
                proc.wait()
            if stalled.is_set():
                raise DiffStalled(f"zfs diff slowed to under 100 changes in {stall // 60} min "
                                  f"after {count[0]:,} changes")
            if proc.returncode not in (0, None) and not cancel.is_set():
                errf.seek(0)
                msg = errf.read().decode(errors="replace").strip()
                if not msg and proc.returncode < 0:
                    msg = (f"zfs diff was killed by signal {-proc.returncode}"
                           + (" (SIGKILL: possibly the kernel's out-of-memory killer; check `dmesg`)"
                              if proc.returncode == -9 else ""))
                raise RuntimeError(msg or f"zfs diff exited with code {proc.returncode}")

    def estimate(self, ds, snap_spec):
        args = ["destroy", "-n", "-v", "-p", f"{ds['name']}@{snap_spec}"]
        assert args[1] == "-n"  # never, ever a real destroy
        out = self._zfs(*args)
        m = re.search(r"^reclaim\t(\d+)", out, re.M)
        return int(m.group(1)) if m else None


class WalkBackend(ZfsBackend):
    """No `zfs diff`: walk each .zfs/snapshot/<snap> tree and compare with the live tree.
    Slower, but works without root as long as the files are readable."""
    name = "walk"
    incremental = True  # consecutive snapshots compared via folder listings (tree_diff)

    def diff_lines(self, ds, a, b, cancel, tick=None, stall=None):
        return self.tree_diff(ds, a, b, cancel, tick)

    def diff(self, ds, snap, cancel, tick=None):
        root = self.snap_root(ds, snap)
        live = ds["mountpoint"]
        seen = [0]

        def walk(rel_dir, live_gone):
            if cancel.is_set():
                return
            seen[0] += 1
            if tick:
                tick(seen[0])
            snap_dir = os.path.join(root, *rel_dir.split("/")) if rel_dir else root
            try:
                it = list(os.scandir(snap_dir))
            except OSError:
                return
            for e in it:
                if not rel_dir and e.name == ".zfs":
                    continue
                rel = f"{rel_dir}/{e.name}" if rel_dir else e.name
                try:
                    is_dir = e.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    gone = live_gone or not os.path.isdir(os.path.join(live, *rel.split("/")))
                    yield from walk(rel, gone)
                elif live_gone or not os.path.lexists(os.path.join(live, *rel.split("/"))):
                    try:
                        st = e.stat(follow_symlinks=False)
                        info = (st.st_size, int(st.st_mtime))
                    except OSError:
                        info = None
                    yield KIND_DELETED, "@" if e.is_symlink() else "F", rel, None, info

        yield from walk("", False)


# --------------------------------------------------------------------------- #
# Demo backend: builds a fake pool with real snapshot dirs in a temp folder
# --------------------------------------------------------------------------- #

class DemoBackend(WalkBackend):
    name = "demo"
    incremental = False  # demo snapshots are plain copies, so inode numbers don't match like on ZFS

    def __init__(self):
        import random
        rnd = random.Random(7)
        self.base = os.path.join(tempfile.gettempdir(), "snapsift-demo")
        shutil.rmtree(self.base, ignore_errors=True)
        self.meta = {}  # path -> (size, mtime): fake sizes, real files stay tiny
        now = int(time.time())
        day = 86400
        # 18 monthly + 14 daily snapshots
        times = sorted({now - day * 30 * m for m in range(18, 0, -1)} | {now - day * d for d in range(14, 0, -1)})
        snaps = [{"name": "auto-" + time.strftime("%Y-%m-%d_%H-%M", time.gmtime(t)), "creation": t,
                  "used": 0, "referenced": 0} for t in times]
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
        specs = {
            "tank/photos": [("Photos/{y}/{ev}/IMG_{n:04d}.svg", 300, (2_000_000, 9_000_000)),
                            ("Photos/{y}/{ev}/VID_{n:04d}.mp4", 30, (40_000_000, 900_000_000))],
            "tank/documents": [("Documents/{ev}/{w}_{n}.txt", 160, (2_000, 400_000)),
                               ("Documents/Taxes/{y}/{w}_{n}.pdf", 40, (80_000, 3_000_000)),
                               ("Projects/{w}/src/{w}_{n}.py", 80, (500, 60_000))],
            "tank/media": [("Music/{ev}/{n:02d} - {w}.flac", 120, (20_000_000, 60_000_000)),
                           ("Movies/{w} ({y}).mkv", 25, (1_500_000_000, 9_000_000_000)),
                           ("Backups/{w}-{y}.zip", 15, (100_000_000, 4_000_000_000))],
        }
        words = ["holiday", "wedding", "birthday", "garden", "roadtrip", "concert", "report",
                 "invoice", "notes", "draft", "budget", "resume", "letter", "archive", "beach", "hiking"]
        self.datasets = []
        for dsname, kinds in specs.items():
            mp = os.path.join(self.base, *dsname.split("/"))
            files = []
            for pattern, count, (lo, hi) in kinds:
                for i in range(count):
                    rel = pattern.format(y=rnd.choice([2019, 2021, 2023, 2025]), ev=rnd.choice(words).title(),
                                         w=rnd.choice(words), n=i + 1)
                    born = rnd.randrange(-6, len(times))  # snapshot index it first appears in
                    born = max(born, 0)
                    roll = rnd.random()
                    if roll < 0.6:
                        died = None                                   # still live
                    elif roll < 0.8:
                        died = len(times) - rnd.randrange(1, 3)       # the recent cull
                    else:
                        died = rnd.randrange(born + 1, len(times) + 1)  # lost along the way
                    if died is not None and died <= born:
                        died = born + 1
                    files.append((rel, born, died, rnd.randrange(lo, hi), times[0] - rnd.randrange(0, 800) * day,
                                  rnd.choice(palette), i))
            # a "moved" folder: Documents/Old -> Documents/Archive (walk mode sees it as deleted)
            for si, s in enumerate(snaps):
                root = os.path.join(mp, ".zfs", "snapshot", s["name"])
                for rel, born, died, size, mtime, col, i in files:
                    if born <= si and (died is None or si < died):
                        self._write(os.path.join(root, *rel.split("/")), rel, col, i, size, mtime)
            for rel, born, died, size, mtime, col, i in files:
                if died is None:
                    self._write(os.path.join(mp, *rel.split("/")), rel, col, i, size, mtime)
            ds_snaps = [dict(s) for s in snaps]
            for s in ds_snaps:
                s["used"] = rnd.randrange(0, 4_000_000_000)
                s["referenced"] = rnd.randrange(50_000_000_000, 90_000_000_000)
            self.datasets.append({"name": dsname, "mountpoint": mp, "used": 70_000_000_000, "snaps": ds_snaps})

    def _write(self, path, rel, col, i, size, mtime):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if rel.endswith(".svg"):
            body = (f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420">'
                    f'<rect width="640" height="420" fill="{col}"/>'
                    f'<circle cx="{120 + (i * 37) % 400}" cy="{110 + (i * 53) % 200}" r="70" fill="#fff" opacity=".35"/>'
                    f'<text x="24" y="390" font-family="sans-serif" font-size="28" fill="#fff">{rel}</text></svg>')
        else:
            body = f"Demo file: {rel}\n\nThis stands in for real content.\n" * 3
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        self.meta[os.path.normcase(os.path.abspath(path))] = (size, mtime)

    def list_datasets(self):
        return json.loads(json.dumps(self.datasets))

    def stat(self, path):
        m = self.meta.get(os.path.normcase(os.path.abspath(path)))
        return m if m else super().stat(path)

    def diff(self, ds, snap, cancel, tick=None):
        time.sleep(0.05)  # make progress visible
        for kind, ftype, rel, newrel, info in super().diff(ds, snap, cancel):
            yield kind, ftype, rel, newrel, self.stat(self.snap_path(ds, snap, rel))

    def estimate(self, ds, snap_spec):
        names = set()
        order = [s["name"] for s in ds["snaps"]]
        for part in snap_spec.split(","):
            a, _, b = part.partition("%")
            if b:
                names.update(order[order.index(a):order.index(b) + 1])
            else:
                names.add(a)
        return sum(s["used"] for s in ds["snaps"] if s["name"] in names)


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #

def ranges_of(sorted_ints):
    out = []
    for g in sorted_ints:
        if out and out[-1][1] == g - 1:
            out[-1][1] = g
        else:
            out.append([g, g])
    return [x for r in out for x in r]


def merge_ranges(flat):
    """[lo, hi, lo, hi, ...] in any order -> sorted, with touching/overlapping runs merged."""
    out = []
    for lo, hi in sorted(zip(flat[::2], flat[1::2])):
        if out and lo <= out[-1] + 1:
            out[-1] = max(out[-1], hi)
        else:
            out += [lo, hi]
    return out


def snap_spec(names_in_order, chosen):
    """Compress chosen snapshot names into zfs 'a%b,c' range syntax (ranges are inclusive)."""
    parts, run = [], []
    for n in names_in_order:
        if n in chosen:
            run.append(n)
        elif run:
            parts.append(run)
            run = []
    if run:
        parts.append(run)
    return ",".join(r[0] if len(r) == 1 else f"{r[0]}%{r[-1]}" for r in parts)


class Scanner:
    def __init__(self, backend, jobs, cache_path, engine="auto", stall=300, log_every=60, keep_rows=20000):
        self.backend = backend
        self.keep_rows = keep_rows  # finished diffs bigger than this wait on disk, not in memory
        self.log_every = log_every
        self.jobs = jobs
        self.engine = engine    # auto: zfs diff, falling back to folder listings if it stalls
        self.stall = stall
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread = None
        self.status = {"state": "idle", "phase": "", "done": 0, "total": 0, "tasks": {},
                       "steps": [], "stepLabels": [],
                       "errors": [], "started": None, "finished": None, "found": 0}
        self.result = None       # dict sent to the browser
        self.result_gz = None    # gzip-compressed JSON of result
        self.view = None         # ResultView of the current result
        self.datasets = []       # datasets of the current result (with snaps)
        self.snaps = []          # global snapshot list: (ds_index, snap dict)
        self.entries = []        # [ds_i, rel, ranges, kind, newrel]
        self.after_scan = None   # called after a new result is installed (auto-save a record)
        self._load_cache()

    # -- cache --------------------------------------------------------------
    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "rb") as f:
                gz = f.read()
            res = json.loads(gzip.decompress(gz))
            if res.get("method") != self.backend.name or res.get("v") != VERSION:
                return
            before = len(res["entries"])
            if clean_result(res):  # saved before extended-attribute entries were filtered out
                log(f"Removed {before - len(res['entries']):,} extended-attribute entries (<xattrdir>) "
                    "from the saved results")
                self._install(res)
            else:
                self._install(res, gz=gz, write=False)  # already on disk, already compressed
            self.status.update(state="done", finished=res["scannedAt"], found=len(res["entries"]))
            log(f"Loaded the last scan's results ({time.ctime(res['scannedAt'])}, {len(res['entries']):,} files)")
        except Exception as e:  # noqa: BLE001
            log(f"WARNING ignoring unreadable cache {self.cache_path}: {e}")

    def _install(self, res, gz=None, write=True):
        if gz is None:
            gz = gzip.compress(json.dumps(res, separators=(",", ":")).encode(), 5)
        self.view = ResultView(self.backend, res, gz)
        self.result, self.result_gz = res, gz
        self.datasets, self.snaps, self.entries = self.view.datasets, self.view.snaps, self.view.entries
        if self.cache_path and write:
            try:
                with open(self.cache_path + ".tmp", "wb") as f:
                    f.write(gz)
                os.replace(self.cache_path + ".tmp", self.cache_path)
            except OSError as e:
                log(f"WARNING could not write cache: {e}")
        if self.after_scan and write:
            try:
                self.after_scan()
            except Exception as e:  # noqa: BLE001
                log(f"WARNING after-scan hook failed: {e}")

    # -- scan ---------------------------------------------------------------
    def start(self, wanted):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return False
            self.cancel.clear()
            self.status = {"state": "scanning", "phase": "listing", "done": 0, "total": 0, "tasks": {},
                           "steps": [], "stepLabels": [],
                           "errors": [], "started": time.time(), "finished": None, "found": 0}
            self.thread = threading.Thread(target=self._run, args=(wanted,), daemon=True)
            self.thread.start()
            threading.Thread(target=self._reporter, args=(self.thread,), daemon=True).start()
            return True

    def _reporter(self, scan_thread):
        """Every `log_every` seconds, log a status line for each task that has been running
        at least that long, so the console shows whether long steps speed up or slow down."""
        if not self.log_every:
            return
        primed = set()
        while scan_thread.is_alive():
            time.sleep(1)
            now = time.time()
            with self.lock:
                tasks = list(self.status["tasks"].values())
            for t in tasks:
                if (t.id, t.pid) not in primed:  # CPU/disk baseline (again once zfs's pid is known)
                    primed.add((t.id, t.pid))
                    t.activity("report")
                if now - t.rep_t >= self.log_every:
                    t.report()

    def _run(self, wanted):
        try:
            self._scan(wanted)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log(f"=== Scan FAILED: {e}")
            self.status.update(state="error", errors=self.status["errors"] + [str(e)], finished=time.time())

    # -- cache of snapshot-to-snapshot diffs: two snapshots never change, so neither does their diff
    def _pair_path(self, key):
        if not key or not self.cache_path:
            return None
        return os.path.join(os.path.dirname(self.cache_path), "snapsift-pairs", f"v1-{key}.json.gz")

    def _pair_load(self, key):
        p = self._pair_path(key)
        if not p or not os.path.exists(p):
            return None
        try:
            with gzip.open(p, "rt", encoding="ascii") as f:
                return [tuple(r) for r in json.load(f)]
        except (OSError, ValueError):
            return None

    def _pair_save(self, key, rows):
        p = self._pair_path(key)
        if not p:
            return False
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with gzip.open(p + ".tmp", "wt", encoding="ascii") as f:
                json.dump(rows, f, separators=(",", ":"))  # \u-escapes keep odd filenames exact
            os.replace(p + ".tmp", p)
            return True
        except OSError as e:
            log(f"WARNING could not cache diff {key}: {e}")
            return False

    def _running(self, label, unit="", kind="process", total=None, step=None):
        """Context manager: shows a Task in the progress panel while the block runs, and
        marks `step` (an index into status["steps"]) with this kind of work."""
        st, lock = self.status, self.lock

        class Running:
            def __enter__(self_):
                self_.task = Task(kind, label, unit, total)
                with lock:
                    st["tasks"][self_.task.id] = self_.task
                    if step is not None:
                        st["steps"][step] = STEP_CODE[kind]
                return self_.task

            def __exit__(self_, exc_type, *exc):
                with lock:
                    st["tasks"].pop(self_.task.id, None)
                self_.task.finish(failed=exc_type is not None)
        return Running()

    def _set_step(self, i, code):
        with self.lock:
            self.status["steps"][i] = code

    def _error(self, msg):
        self.status["errors"].append(msg)
        log("WARNING " + msg)

    def _phase(self, phase, text):
        self.status["phase"] = phase
        log(f"=== {text}")

    def _diff_each(self, datasets, gsnaps):
        """Compare every snapshot with the live filesystem. Simple, but anything deleted
        before snapshot N is re-discovered by every snapshot older than N. Used by walk mode."""
        st = self.status
        order = list(reversed(gsnaps))  # newest first
        with self.lock:
            st.update(phase="diff", total=len(order), steps=["q"] * len(order),
                      stepLabels=[f"{datasets[di]['name']}@{s['name']} vs live" for di, s in order])
        task_kind = "zfs" if self.backend.name == "zfs-diff" else "tree"
        found = {}
        merge_lock = threading.Lock()
        index = {id(s): i for i, (_, s) in enumerate(order)}

        def work(item):
            di, s = item
            i = index[id(s)]
            with self._running(st["stepLabels"][i], "changes read" if task_kind == "zfs" else "folders checked",
                               task_kind, step=i) as tick:
                local = list(self.backend.diff(datasets[di], s, self.cancel, tick))
            self._set_step(i, "k")
            with merge_lock:
                g = s["gid"]
                for kind, ftype, rel, newrel, info in local:
                    e = found.get((di, rel))
                    if e is None:
                        found[(di, rel)] = [{g}, ftype, kind, newrel, info, g]
                    else:
                        e[0].add(g)
                        if g > e[5]:  # newer snapshot wins for type/kind/stat
                            e[1], e[2], e[3], e[5] = ftype, kind, newrel, g
                            if info:
                                e[4] = info
                st["done"] += 1
                st["found"] = len(found)

        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            futs = {ex.submit(work, it): it for it in order}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    di, s = futs[f]
                    self._error(f"{datasets[di]['name']}@{s['name']}: {e}")
                    self._set_step(index[id(s)], "f")
                    with merge_lock:
                        st["done"] += 1
        for e in found.values():
            e[0] = ranges_of(sorted(e[0]))
        return found

    def _diff_chained(self, datasets):
        """Diff consecutive snapshots (newest vs live, then each snapshot vs the next one)
        and walk backwards in time, tracking which paths are missing from live.

        Every change is read from zfs exactly once, so a big cull costs one slow diff
        instead of one per older snapshot. G holds the paths of the snapshot being
        processed that are missing from live: path -> [ftype, kind, newrel, hi_gid],
        where hi_gid is the newest snapshot of the contiguous run that has the path.
        """
        st = self.status
        jobs = []
        for di, d in enumerate(datasets):
            snaps = d["snaps"]
            jobs.append((di, snaps[-1], None))
            for k in range(len(snaps) - 2, -1, -1):
                jobs.append((di, snaps[k], snaps[k + 1]))
        labels = [f"{datasets[di]['name']}@{a['name']} → {'live' if b is None else '@' + b['name']}"
                  for di, a, b in jobs]
        with self.lock:
            st.update(phase="diff", total=len(jobs), steps=["q"] * len(jobs), stepLabels=labels)
        found = {}
        from_cache = set()

        def collect(i, di, a, b, use_cache=True, in_memory=False):
            d, label = datasets[di], labels[i]
            key = f"{a['guid']}-{b['guid']}" if b is not None and a.get("guid") and b.get("guid") else None
            p = self._pair_path(key)
            if use_cache and p and os.path.exists(p):  # diffed in an earlier run; read it on its turn
                log(f"cached {'':<18} {label}  (saved by an earlier scan)")
                from_cache.add(i)
                self._set_step(i, "w")
                return ("on-disk", key, None)
            res = None
            if self.engine != "tree":
                try:
                    with self._running(label, "changes read", "zfs", step=i) as tick:
                        res = list(self.backend.diff_lines(d, a, b, self.cancel, tick,
                                                           stall=self.stall if self.engine == "auto" else None))
                except DiffStalled as e:
                    self._error(f"{label}: {e}; switched to comparing folder listings")
            if res is None:
                with self._running(label, "folders compared", "tree", step=i) as tick:
                    res = self.backend.tree_diff(d, a, b, self.cancel, tick)
            saved = False
            if key and not self.cancel.is_set():
                with self._running(label, "changes", "save", step=i) as t:
                    t(len(res))
                    saved = self._pair_save(key, res)
            self._set_step(i, "w")
            if saved and len(res) > self.keep_rows and not in_memory:
                # It may wait a long time for its turn (processing goes strictly newest to oldest).
                # It's on disk now, so drop it from memory and reload it when its turn comes.
                return ("on-disk", key, len(res))
            return res

        def full_resync(i, di, a):
            """Rebuild G for snapshot a straight from `zfs diff a` (vs live)."""
            G = {}
            with self._running(f"{datasets[di]['name']}@{a['name']} → live (fallback)",
                               "changes read", "resync", step=i) as tick:
                for kind, ftype, rel, newrel, _ in self.backend.diff(datasets[di], a, self.cancel, tick):
                    G[rel] = [ftype, kind, newrel, a["gid"]]
            return G

        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            # Submitted newest-first and consumed strictly in that order. Only a few diffs run
            # ahead of the consumer: finished diffs wait in memory, and several multi-million-line
            # diffs waiting at once can push the NAS into swap.
            futs, pending = [], iter(enumerate(jobs))

            def top_up(n):
                while len(futs) < n:
                    nxt = next(pending, None)
                    if nxt is None:
                        return
                    i, (di, a, b) = nxt
                    futs.append((di, a, b, ex.submit(collect, i, di, a, b)))

            # With a diff cache, finished diffs wait on disk rather than in memory, so every diff
            # can run as soon as a worker is free. Without one, cap how far ahead diffs run.
            window = len(jobs) if self.cache_path else self.jobs * 2
            cur_ds, G, k, broken = None, {}, 0, False
            while True:
                top_up(k + window)
                if k >= len(futs):
                    break
                di, a, b, fut = futs[k]
                futs[k] = None
                i = k
                k += 1
                if self.cancel.is_set():
                    for x in futs:
                        if x:
                            x[3].cancel()
                    return found
                if di != cur_ds:
                    if cur_ds is not None:
                        self._close_all(found, cur_ds, G, datasets[cur_ds]["first"])
                    cur_ds, G, broken = di, {}, False
                d, g = datasets[di], a["gid"]
                st["waitingOn"] = i  # shown in the UI: what processing is waiting for
                try:
                    changes = fut.result()
                    if isinstance(changes, tuple) and changes and changes[0] == "on-disk":
                        key = changes[1]
                        with self._running(labels[i], "changes", "load", step=i) as t:
                            changes = self._pair_load(key)
                            t(len(changes or ()))
                        if changes is None:  # unreadable cache file: diff it again now
                            log(f"WARNING cached diff for {labels[i]} could not be read; diffing it again")
                            from_cache.discard(i)
                            changes = collect(i, di, a, b, use_cache=False, in_memory=True)
                    err = None
                except Exception as e:  # noqa: BLE001
                    if self.cancel.is_set():
                        return found
                    err = e
                st["waitingOn"] = None
                if err is not None or broken:
                    # Can't chain from this step: rebuild snapshot a's state directly against live.
                    if err is not None:
                        self._error(f"{labels[i]}: {type(err).__name__}: {err} "
                                    f"(fell back to comparing @{a['name']} with live)")
                    else:
                        log(f"rebuild {labels[i]}: previous snapshot was skipped, comparing @{a['name']} with live")
                    try:
                        newG = full_resync(i, di, a)
                        broken = False
                    except Exception as e2:  # noqa: BLE001
                        if self.cancel.is_set():
                            return found
                        # Leave G as-is is wrong and an empty G is wrong; mark the chain broken so
                        # the next older snapshot is rebuilt against live instead of from this.
                        self._error(f"{d['name']}@{a['name']}: {type(e2).__name__}: {e2} "
                                    "(snapshot skipped; results for it may be incomplete)")
                        newG, broken = {}, True
                    for rel in list(G):
                        self._close(found, di, rel, G.pop(rel), g + 1)
                    G = newG
                    self._set_step(i, "f")
                    st["done"] += 1
                    continue
                with self._running(labels[i], "items", "process",
                                   total=self._step_total(changes, G), step=i) as tick:
                    self._step(found, di, d, G, g, changes, tick)
                    tick(tick.total)
                changes = None
                self._set_step(i, "c" if i in from_cache else "k")
                st["done"] += 1
                st["found"] = len(found) + len(G)
            if cur_ds is not None:
                self._close_all(found, cur_ds, G, datasets[cur_ds]["first"])
        for e in found.values():
            e[0] = merge_ranges(e[0])
        return found

    @staticmethod
    def _step_total(changes, G):
        return 2 * len(changes) or 1  # refined inside _step once the changes are classified

    def _step(self, found, di, d, G, g, changes, tick=None):
        """Turn G (missing paths of the next-newer snapshot, or empty for live) into G for snapshot g."""
        added, removed, file_ren, dir_ren = [], [], [], []
        tick = tick or (lambda n: None)
        done = 0
        for n, (change, ftype, rel, newrel) in enumerate(changes):
            if not n % 50000:
                tick(n)
            if is_xattr_path(rel) or is_xattr_path(newrel):
                continue  # extended attributes (also in diffs cached before they were filtered)
            if ftype == "/":
                if change == "R" and newrel:
                    dir_ren.append((rel, newrel))
            elif change == "+":
                added.append(rel)
            elif change == "-":
                removed.append((ftype, rel))
            elif change == "R" and newrel:
                file_ren.append((ftype, rel, newrel))

        done = len(changes)
        if isinstance(tick, Task):
            tick.total = (done + len(added) + 2 * len(file_ren) + 2 * len(removed)
                          + (len(G) if dir_ren else 0)) or 1
        tick(done)

        lo = g + 1  # anything leaving G was present from snapshot g+1 up to its hi_gid
        incoming = []  # (rel in snapshot g, ftype, kind, newrel)
        for rel in added:  # created after g: not in snapshot g
            if rel in G:
                self._close(found, di, rel, G.pop(rel), lo)
        done += len(added)
        tick(done)
        for ftype, old, new in file_ren:  # same file, different path in g
            s = G.pop(new, None)
            if s is not None:
                self._close(found, di, new, s, lo)
                incoming.append((old, s[0], s[1], s[2]))
            else:  # still exists, at `new` or wherever later renames took it
                incoming.append((old, ftype, KIND_MOVED, new))
        done += len(file_ren)
        tick(done)
        if dir_ren:
            # Map each tracked path through its deepest renamed ancestor folder. Walking up a
            # path's own parents costs its depth, so one pass over G handles every renamed
            # folder; scanning G once per renamed folder was quadratic and could take hours.
            ren = {new: old for old, new in dir_ren}
            moves = []
            for n, rel in enumerate(G):
                if not n % 50000:
                    tick(done + n)
                p = rel
                while True:
                    cut = p.rfind("/")
                    if cut < 0:
                        break
                    p = p[:cut]
                    old_dir = ren.get(p)
                    if old_dir is not None:
                        moves.append((rel, old_dir + rel[cut:]))
                        break
            done += len(G)
            for rel, old_rel in moves:
                s = G.pop(rel)
                self._close(found, di, rel, s, lo)
                incoming.append((old_rel, s[0], s[1], s[2]))
            if isinstance(tick, Task):
                tick.total += len(moves)
        for ftype, rel in removed:  # in g, freed before g+1: gone from live for good
            incoming.append((rel, ftype, KIND_DELETED, None))
        done += len(removed)
        tick(done)
        for n, (rel, ftype, kind, newrel) in enumerate(incoming):
            if not n % 50000:
                tick(done + n)
            if rel not in G:
                G[rel] = [ftype, kind, newrel, g]

    @staticmethod
    def _close(found, di, rel, s, lo):
        ftype, kind, newrel, hi = s
        e = found.get((di, rel))
        if e is None:  # first close is the newest run: it decides type/kind/stat snapshot
            found[(di, rel)] = [[lo, hi], ftype, kind, newrel, None, hi]
        else:
            e[0] += [lo, hi]

    def _close_all(self, found, di, G, lo):
        for rel, s in G.items():
            self._close(found, di, rel, s, lo)
        G.clear()

    def _scan(self, wanted):
        st = self.status
        self._phase("listing", "Scan started: listing datasets and snapshots")
        with self._running("datasets and snapshots", "", "list"):
            all_ds = self.backend.list_datasets()
        datasets = [d for d in all_ds if not wanted or d["name"] in wanted]
        gsnaps = []
        for di, d in enumerate(datasets):
            d["first"] = len(gsnaps)
            for s in d["snaps"]:
                s["ds"] = di
                s["gid"] = len(gsnaps)
                gsnaps.append((di, s))
            d["last"] = len(gsnaps) - 1
        # found: (ds_i, rel) -> [ranges, ftype, kind, newrel, statinfo, stat_gid]
        # ranges is a flat ascending [lo, hi, lo, hi, ...] list of snapshot gids.
        self._phase("diff", f"Comparing snapshots: {len(datasets)} dataset(s), {len(gsnaps)} snapshot(s)")
        if self.backend.incremental:
            found = self._diff_chained(datasets)
        else:
            found = self._diff_each(datasets, gsnaps)
        if self.cancel.is_set():
            st.update(state="cancelled", finished=time.time())
            log("=== Scan cancelled")
            return

        # classify + stat in the newest snapshot that has the file
        st.update(done=0, total=len(found))
        self._phase("stat", f"Reading file details for {len(found):,} deleted files")
        items = list(found.items())
        stat_task = self._running(f"{len(found):,} files", "files", "stat", total=len(found) or 1)
        task = stat_task.__enter__()

        def finish(item):
            (di, rel), e = item
            d = datasets[di]
            if e[2] == KIND_DELETED and self.backend.live_exists(d, rel):
                e[2] = KIND_REPLACED  # zfs diff says deleted, but something lives at that path again
            if e[4] is None:
                try:
                    e[4] = self.backend.stat(self.backend.snap_path(d, gsnaps[e[5]][1], rel))
                except OSError:
                    e[4] = (0, 0)
            st["done"] += 1
            task(st["done"])

        try:
            with ThreadPoolExecutor(max_workers=max(8, self.jobs * 2)) as ex:
                for f in as_completed([ex.submit(finish, it) for it in items]):
                    if self.cancel.is_set():
                        ex.shutdown(cancel_futures=True)
                        st.update(state="cancelled", finished=time.time())
                        log("=== Scan cancelled")
                        return
                    f.result()
        finally:
            stat_task.__exit__(None, None, None)

        self._phase("build", "Building results")
        build = self._running(f"{len(items):,} files", "", "process")
        build.__enter__()
        items.sort(key=lambda kv: (kv[0][0], kv[0][1]))
        # Folder paths go in a table and rows reference them by index: a big cull
        # puts thousands of files in each folder, so this shrinks the payload and browser memory.
        dirs, dir_index, entries = [], {}, []
        for i, ((di, rel), e) in enumerate(items):
            folder, _, name = rel.rpartition("/")
            dk = dir_index.get((di, folder))
            if dk is None:
                dk = dir_index[(di, folder)] = len(dirs)
                dirs.append([di, folder])
            row = [i, di, dk, name, e[4][0], e[4][1], e[1], e[0], e[2]]
            if e[3]:
                row.append(e[3])
            entries.append(row)
        res = {
            "v": VERSION, "method": self.backend.name, "scannedAt": time.time(),
            "datasets": [{"name": d["name"], "mountpoint": d["mountpoint"], "used": d["used"],
                          "first": d["first"], "last": d["last"]} for d in datasets],
            "snaps": [{"gid": s["gid"], "ds": di, "name": s["name"], "creation": s["creation"], "guid": s.get("guid"),
                       "used": s["used"], "referenced": s["referenced"]} for di, s in gsnaps],
            "dirs": dirs,
            "entries": entries,
            "errors": st["errors"],
        }
        self._install(res)
        build.__exit__(None, None, None)
        st.update(state="done", phase="", finished=time.time(), found=len(entries))
        log(f"=== Scan finished in {fmt_dur(time.time() - st['started'])}: {len(entries):,} files found"
            + (f", {len(st['errors'])} warning(s)" if st["errors"] else ""))

    # -- helpers used by the HTTP layer ---------------------------------------
    def entry_paths(self, eid, gid=None):
        return self.view.entry_paths(eid, gid)


class ResultView:
    """One set of scan results (the current scan, or a saved record) plus how to find its files."""

    def __init__(self, backend, res, gz=None):
        self.backend, self.result, self.result_gz = backend, res, gz
        self.datasets = res["datasets"]
        self.snaps = [(s["ds"], s) for s in res["snaps"]]
        dirs = res["dirs"]
        self.entries = [(e[1], f"{dirs[e[2]][1]}/{e[3]}" if dirs[e[2]][1] else e[3], e[7], e[8],
                         e[9] if len(e) > 9 else None) for e in res["entries"]]

    def entry_paths(self, eid, gid=None):
        di, rel, ranges, kind, newrel = self.entries[eid]
        gids = [g for a, b in zip(ranges[::2], ranges[1::2]) for g in range(a, b + 1)]
        if gid is None:
            gid = gids[-1]
        if gid not in gids:
            raise KeyError("snapshot does not contain this file")
        d = self.datasets[di]
        snap = self.snaps[gid][1]
        return d, snap, self.backend.snap_path(d, snap, rel), self.backend.live_path(d, rel), gids


# --------------------------------------------------------------------------- #
# Records: frozen copies of scan results, browsable after the snapshots are gone
# --------------------------------------------------------------------------- #

class Records:
    """Saved scan results. A record needs no zfs diff, processing or file lookups to browse;
    it's the catalog of what the snapshots held at that moment."""
    AUTO_KEEP = 5  # automatic per-scan records kept; named and pre-destroy records are kept forever
    _ID = re.compile(r"^[0-9A-Za-z-]{1,80}$")

    def __init__(self, scanner, directory):
        self.scanner, self.dir = scanner, directory
        self._loaded = (None, None)  # (id, ResultView): the record currently being browsed
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _path(self, rid, ext):
        if not self.dir or not self._ID.match(rid or ""):
            raise KeyError("unknown record")
        return os.path.join(self.dir, f"{rid}{ext}")

    def list(self):
        out = []
        if self.dir and os.path.isdir(self.dir):
            for name in os.listdir(self.dir):
                if name.endswith(".meta.json"):
                    try:
                        with open(os.path.join(self.dir, name), encoding="utf-8") as f:
                            out.append(json.load(f))
                    except (OSError, ValueError):
                        pass
        return sorted(out, key=lambda m: m.get("savedAt", 0), reverse=True)

    def save(self, name, note="", kind="manual"):
        sc = self.scanner
        if not self.dir:
            raise ValueError("No folder for records (run with a cache path)")
        if not sc.result_gz:
            raise ValueError("No scan results to save yet")
        res = sc.result
        rid = time.strftime("%Y%m%d-%H%M%S") + f"-{kind}-{secrets.token_hex(3)}"
        deleted = [e for e in res["entries"] if e[8] == KIND_DELETED]
        snaps = res["snaps"]
        meta = {"id": rid, "name": (name or "").strip() or f"Scan of {time.strftime('%Y-%m-%d %H:%M', time.localtime(res['scannedAt']))}",
                "note": note or "", "kind": kind, "savedAt": time.time(), "scannedAt": res["scannedAt"],
                "method": res.get("method"), "datasets": [d["name"] for d in res["datasets"]],
                "files": len(res["entries"]), "deleted": len(deleted), "deletedBytes": sum(e[4] for e in deleted),
                "snapshots": len(snaps), "oldest": min((s["creation"] for s in snaps), default=None),
                "newest": max((s["creation"] for s in snaps), default=None), "bytesOnDisk": len(sc.result_gz)}
        data = self._path(rid, ".json.gz")
        with open(data + ".tmp", "wb") as f:
            f.write(sc.result_gz)
        os.replace(data + ".tmp", data)
        with open(self._path(rid, ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        log(f"Saved record '{meta['name']}' ({human(len(sc.result_gz))}) to {data}")
        if kind == "auto":
            autos = [m for m in self.list() if m.get("kind") == "auto"]
            for old in autos[self.AUTO_KEEP:]:
                self.delete(old["id"], quiet=True)
        return meta

    def delete(self, rid, quiet=False):
        for ext in (".json.gz", ".meta.json"):
            p = self._path(rid, ext)
            if os.path.exists(p):
                os.remove(p)
        if self._loaded[0] == rid:
            self._loaded = (None, None)
        if not quiet:
            log(f"Deleted record {rid}")

    def data_path(self, rid):
        p = self._path(rid, ".json.gz")
        if not os.path.exists(p):
            raise KeyError("unknown record")
        return p

    def load(self, rid):
        """(result, gzip bytes) of a record, cleaning out extended-attribute pseudo-entries
        from records saved before those were filtered (rewriting the record once)."""
        p = self.data_path(rid)
        with open(p, "rb") as f:
            gz = f.read()
        res = json.loads(gzip.decompress(gz))
        if clean_result(res):
            gz = gzip.compress(json.dumps(res, separators=(",", ":")).encode(), 5)
            with open(p + ".tmp", "wb") as f:
                f.write(gz)
            os.replace(p + ".tmp", p)
            mp = self._path(rid, ".meta.json")
            try:
                with open(mp, encoding="utf-8") as f:
                    meta = json.load(f)
                deleted = [e for e in res["entries"] if e[8] == KIND_DELETED]
                meta.update(files=len(res["entries"]), deleted=len(deleted),
                            deletedBytes=sum(e[4] for e in deleted), bytesOnDisk=len(gz))
                with open(mp, "w", encoding="utf-8") as f:
                    json.dump(meta, f)
            except (OSError, ValueError):
                pass
            log(f"Cleaned extended-attribute entries out of record {rid}")
        return res, gz

    def view(self, rid):
        if self._loaded[0] != rid:
            res, gz = self.load(rid)
            self._loaded = (rid, ResultView(self.scanner.backend, res, gz))
        return self._loaded[1]

    def alive(self, rid):
        """Which of the record's snapshots still exist: one `zfs list`, no per-file lookups."""
        v = self.view(rid)
        by_guid, by_name = set(), set()
        for d in self.scanner.backend.list_datasets():
            for s in d["snaps"]:
                if s.get("guid"):
                    by_guid.add(str(s["guid"]))
                by_name.add((d["name"], s["name"]))
        alive = []
        for di, s in v.snaps:
            if (s.get("guid") and str(s["guid"]) in by_guid) or \
                    (not s.get("guid") and (v.datasets[di]["name"], s["name"]) in by_name):
                alive.append(s["gid"])
        return alive


# --------------------------------------------------------------------------- #
# Recovery: the one thing SnapSift writes. Copies files out of snapshots into a folder you choose.
# --------------------------------------------------------------------------- #

class Recovery:
    """Copies snapshot files to <dest>/[<dataset>/]<original path>.

    Safety rules: never overwrites (existing targets are skipped); copies to a temporary name and
    links it into place only when complete, so cancels and crashes never leave a partial file
    under a real name; refuses destinations inside .zfs or without enough free space; keeps
    contents, times, permissions, owner/group, xattrs and symlinks; writes a manifest (TSV) of
    everything it did into the destination folder."""

    def __init__(self, scanner):
        self.scanner = scanner
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread = None
        self.status = {"state": "idle"}

    # -- planning -----------------------------------------------------------
    def plan(self, items, dest, layout, view=None):
        """items: [[entry_id, snapshot_gid or None], ...] -> (dest, [(src, target, size, label)])."""
        dest = (dest or "").strip()
        if not dest or not os.path.isabs(dest):
            raise ValueError("Enter an absolute destination path, e.g. /mnt/tank/recovered")
        dest = os.path.normpath(dest)
        if f"{os.sep}.zfs{os.sep}" in dest + os.sep:
            raise ValueError("The destination can't be inside a .zfs snapshot directory (snapshots are read-only)")
        sc = view or self.scanner.view  # current scan, or a saved record being browsed
        if not sc or not sc.entries:
            raise ValueError("No scan results to recover from")
        out, seen = [], set()
        for item in items:
            eid, gid = (item[0], item[1] if len(item) > 1 else None) if isinstance(item, list) else (item, None)
            d, snap, src, _live, _ = sc.entry_paths(int(eid), None if gid is None else int(gid))
            rel = sc.entries[int(eid)][1]
            parts = ([*d["name"].split("/")] if layout == "dataset" else []) + rel.split("/")
            target = os.path.join(dest, *parts)
            if target in seen:
                continue  # same file selected twice (e.g. two versions with the dataset layout off)
            seen.add(target)
            try:
                size = os.lstat(src).st_size
            except OSError:
                size = 0
            out.append((src, target, size, f"{d['name']}/{rel} @{snap['name']}", sc.backend.snap_root(d, snap)))
        return dest, out

    @staticmethod
    def _existing_ancestor(path):
        while path and not os.path.exists(path):
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        return path

    def check(self, items, dest, layout, view=None):
        info = {"ok": False}
        try:
            dest, plan = self.plan(items, dest, layout, view)
        except (ValueError, KeyError) as e:
            info["error"] = str(e)
            return info
        anc = self._existing_ancestor(dest)
        conflicts = sum(1 for _, t, _, _, _ in plan if os.path.lexists(t))
        needed = sum(s for _, t, s, _, _ in plan if not os.path.lexists(t))
        try:
            free = shutil.disk_usage(anc).free
        except OSError:
            free = None
        info.update(dest=dest, destExists=os.path.isdir(dest), files=len(plan), conflicts=conflicts,
                    needed=needed, free=free, writable=os.access(anc, os.W_OK),
                    examples=[[label, t] for _, t, _, label, _ in plan[:3]])
        if not info["writable"]:
            info["error"] = f"No permission to write in {anc}"
        elif free is not None and needed > free * 0.98:
            info["error"] = f"Not enough free space: needs {human(needed)}, {human(free)} free"
        elif not plan:
            info["error"] = "Nothing selected"
        else:
            info["ok"] = True
        return info

    # -- running ------------------------------------------------------------
    def start(self, items, dest, layout, view=None):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("A recovery is already running")
            info = self.check(items, dest, layout, view)
            if not info["ok"]:
                raise ValueError(info.get("error") or "Can't recover to that destination")
            dest, plan = self.plan(items, dest, layout, view)  # resolved now: a new scan can't change it
            self.cancel.clear()
            self.status = {"state": "running", "dest": dest, "total": len(plan), "done": 0,
                           "bytes": 0, "bytesTotal": sum(s for _, _, s, _, _ in plan), "copied": 0,
                           "skipped": 0, "failed": 0, "errors": [], "started": time.time(),
                           "finished": None, "manifest": None, "current": None}
            self.thread = threading.Thread(target=self._run, args=(dest, plan), daemon=True)
            self.thread.start()

    def _run(self, dest, plan):
        st = self.status
        log(f"=== Recovery started: {len(plan):,} files ({human(st['bytesTotal'])}) into {dest}")
        os.makedirs(dest, exist_ok=True)
        real_dest = os.path.realpath(dest)
        manifest = os.path.join(dest, time.strftime("snapsift-recovery-%Y%m%d-%H%M%S.tsv"))
        st["manifest"] = manifest
        made_dirs = {}  # created target dir -> snapshot dir it mirrors (to copy its timestamps)
        try:
            mf = open(manifest, "w", encoding="utf-8", errors="backslashreplace")
            mf.write("result\tsource\ttarget\tbytes\tnote\n")
        except OSError as e:
            mf = None
            log(f"WARNING could not write manifest {manifest}: {e}")

        def record(result, src, target, size, note=""):
            if mf:
                mf.write(f"{result}\t{src}\t{target}\t{size}\t{note}\n")

        last_log = time.time()
        for src, target, size, label, src_root in plan:
            if self.cancel.is_set():
                break
            st["current"] = label
            try:
                if os.path.lexists(target):
                    st["skipped"] += 1
                    record("skipped", src, target, size, "already exists")
                else:
                    parent = os.path.dirname(target)
                    self._makedirs(parent, real_dest, os.path.dirname(src), src_root, made_dirs)
                    self._copy(src, target)
                    st["copied"] += 1
                    record("copied", src, target, size)
            except Exception as e:  # noqa: BLE001
                if self.cancel.is_set():
                    break
                st["failed"] += 1
                msg = f"{label}: {e}"
                if len(st["errors"]) < 200:
                    st["errors"].append(msg)
                record("failed", src, target, size, str(e))
                log(f"WARNING recover failed: {msg}")
            st["done"] += 1
            st["bytes"] += size
            if time.time() - last_log >= 60:
                last_log = time.time()
                log(f"status RECOVERING        {st['done']:,} / {st['total']:,} files, "
                    f"{human(st['bytes'])} / {human(st['bytesTotal'])}")
        for tdir, sdir in sorted(made_dirs.items(), key=lambda kv: -len(kv[0])):  # deepest first
            try:
                shutil.copystat(sdir, tdir, follow_symlinks=False)
            except OSError:
                pass
        if mf:
            mf.close()
        st.update(state="cancelled" if self.cancel.is_set() else "done", finished=time.time(), current=None)
        log(f"=== Recovery {st['state']}: {st['copied']:,} copied, {st['skipped']:,} skipped (already existed), "
            f"{st['failed']:,} failed; manifest {manifest}")

    @staticmethod
    def _makedirs(parent, real_dest, src_dir, src_root, made):
        if os.path.isdir(parent):
            if not (os.path.realpath(parent) + os.sep).startswith(real_dest + os.sep):
                raise OSError(f"{parent} resolves outside the destination (symlinked folder?)")
            return
        missing, p, s = [], parent, src_dir
        while not os.path.isdir(p):
            missing.append((p, s))
            p, s = os.path.dirname(p), os.path.dirname(s)
        if not (os.path.realpath(p) + os.sep).startswith(real_dest + os.sep) and \
                os.path.realpath(p) != real_dest:
            raise OSError(f"{p} resolves outside the destination (symlinked folder?)")
        for d, sd in reversed(missing):
            os.mkdir(d)
            if sd.startswith(src_root + os.sep):  # only folders inside the snapshot
                made[d] = sd

    def _copy(self, src, target):
        """Copy to a temporary name, then hard-link into place (fails instead of overwriting)."""
        st = os.lstat(src)
        tmp = os.path.join(os.path.dirname(target), f".{os.path.basename(target)}.snapsift-{os.getpid()}.partial")
        try:
            if stat.S_ISLNK(st.st_mode):
                os.symlink(os.readlink(src), tmp)
            elif stat.S_ISREG(st.st_mode):
                with open(src, "rb") as fi, open(tmp, "xb") as fo:
                    self._copy_data(fi, fo, st.st_size)
            else:
                raise OSError("not a regular file or symlink (device/fifo/socket); skipped")
            if hasattr(os, "chown"):
                try:
                    os.chown(tmp, st.st_uid, st.st_gid, follow_symlinks=False)
                except (OSError, NotImplementedError):
                    pass
            try:
                shutil.copystat(src, tmp, follow_symlinks=False)  # mode, times, xattrs where possible
            except (OSError, NotImplementedError):
                pass
            os.link(tmp, target, follow_symlinks=False)  # atomic, and raises if target exists
        finally:
            if os.path.lexists(tmp):
                os.unlink(tmp)

    def _copy_data(self, fi, fo, size):
        chunk = 64 << 20
        try:  # kernel-side copy on Linux: fast, no userspace buffers
            off = 0
            while off < size:
                if self.cancel.is_set():
                    raise OSError("cancelled")
                n = os.sendfile(fo.fileno(), fi.fileno(), off, min(chunk, size - off))
                if n == 0:
                    break
                off += n
            return
        except (AttributeError, OSError) as e:
            if str(e) == "cancelled":
                raise
            fo.seek(0)
            fo.truncate()
            fi.seek(0)
        while True:
            if self.cancel.is_set():
                raise OSError("cancelled")
            buf = fi.read(8 << 20)
            if not buf:
                break
            fo.write(buf)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024 or unit == "PB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "SnapSift/" + VERSION
    scanner: Scanner = None
    recovery: Recovery = None
    records: Records = None
    token = None
    html = b""

    def log_message(self, fmt, *args):  # quieter
        if os.environ.get("SNAPSIFT_DEBUG"):
            super().log_message(fmt, *args)

    # -- plumbing -----------------------------------------------------------
    def _authed(self, q):
        if not self.token:
            return True
        cookie = self.headers.get("Cookie", "")
        m = re.search(r"(?:^|;\s*)snapsift=([^;]+)", cookie)
        supplied = (m.group(1) if m else "") or self.headers.get("X-Token", "") or q.get("t", [""])[0]
        return secrets.compare_digest(supplied, self.token)

    def _send(self, code, body=b"", ctype="application/json", headers=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/" and self.token and q.get("t", [""])[0] == self.token:
            return self._send(302, b"", "text/plain", {
                "Location": "/", "Set-Cookie": f"snapsift={self.token}; Path=/; HttpOnly; SameSite=Strict"})
        if not self._authed(q):
            return self._send(401, "<h1>401</h1><p>Open the URL with the <code>?t=</code> token that SnapSift "
                                   "printed when it started.</p>", "text/html")
        try:
            route = {
                "/": self.get_index,
                "/api/datasets": self.get_datasets,
                "/api/status": self.get_status,
                "/api/data": self.get_data,
                "/api/file": self.get_file,
                "/api/versions": self.get_versions,
                "/api/log": self.get_log,
                "/api/recover/status": self.get_recover_status,
                "/api/records": self.get_records,
                "/api/records/data": self.get_record_data,
                "/api/records/alive": self.get_record_alive,
            }.get(u.path)
            if not route:
                return self._send(404, "not found", "text/plain")
            route(q)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        if not self._authed(parse_qs(u.query)):
            return self._json({"error": "unauthorized"}, 401)
        try:
            route = {"/api/scan": self.post_scan, "/api/cancel": self.post_cancel,
                     "/api/estimate": self.post_estimate, "/api/script": self.post_script,
                     "/api/recover/check": self.post_recover_check, "/api/recover/start": self.post_recover_start,
                     "/api/recover/cancel": self.post_recover_cancel,
                     "/api/records/save": self.post_record_save,
                     "/api/records/delete": self.post_record_delete}.get(u.path)
            if not route:
                return self._send(404, "not found", "text/plain")
            route(self._body())
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": str(e)}, 500)

    # -- GET ----------------------------------------------------------------
    def get_index(self, q):
        self._send(200, self.html, "text/html; charset=utf-8")

    def get_datasets(self, q):
        ds = self.scanner.backend.list_datasets()
        self._json([{"name": d["name"], "mountpoint": d["mountpoint"], "used": d["used"],
                     "snapshots": len(d["snaps"]),
                     "oldest": d["snaps"][0]["creation"], "newest": d["snaps"][-1]["creation"]} for d in ds])

    def get_status(self, q):
        with self.scanner.lock:
            s = dict(self.scanner.status)
            tasks = sorted(s["tasks"].values(), key=lambda t: t.started)
            s["steps"] = "".join(s["steps"])
        if not q.get("labels"):
            s.pop("stepLabels", None)  # sent once per scan; the browser asks again when `started` changes
        out = []
        for t in tasks:
            act = t.activity()
            out.append({"id": t.id, "kind": t.kind, "label": t.label, "unit": t.unit, "n": t.n,
                        "total": t.total, "started": t.started, "last": t.last, **(act or {})})
        s["tasks"] = out
        s["serverTime"] = time.time()
        s["startupWarnings"] = STARTUP_WARNINGS
        s["method"] = self.scanner.backend.name
        s["hasResult"] = self.scanner.result is not None
        s["resultAt"] = self.scanner.result["scannedAt"] if self.scanner.result else None
        self._json(s)

    def get_recover_status(self, q):
        s = dict(self.recovery.status)
        s["serverTime"] = time.time()
        self._json(s)

    def post_recover_check(self, body):
        self._json(self.recovery.check(body.get("items", []), body.get("dest"), body.get("layout", "dataset"),
                                       self._view(body)))

    def post_recover_start(self, body):
        try:
            self.recovery.start(body.get("items", []), body.get("dest"), body.get("layout", "dataset"),
                                self._view(body))
        except (ValueError, RuntimeError, KeyError) as e:
            return self._json({"error": str(e)}, 409)
        self._json({"started": True})

    # -- records --------------------------------------------------------------
    def get_records(self, q):
        self._json({"records": self.records.list(), "dir": self.records.dir})

    def get_record_data(self, q):
        gz = self.records.view(q.get("id", [""])[0]).result_gz  # loads (and cleans) once, then cached
        if "gzip" in self.headers.get("Accept-Encoding", ""):
            self._send(200, gz, "application/json", {"Content-Encoding": "gzip"})
        else:
            self._send(200, gzip.decompress(gz))

    def get_record_alive(self, q):
        self._json({"alive": self.records.alive(q.get("id", [""])[0]), "checked": time.time()})

    def post_record_save(self, body):
        try:
            self._json(self.records.save(body.get("name", ""), body.get("note", ""), "manual"))
        except ValueError as e:
            self._json({"error": str(e)}, 409)

    def post_record_delete(self, body):
        self.records.delete(body.get("id", ""))
        self._json({"ok": True})

    def post_recover_cancel(self, body):
        self.recovery.cancel.set()
        self._json({"ok": True})

    def get_log(self, q):
        after = int(q.get("after", ["0"])[0])
        with _log_lock:
            lines = [[n, t] for n, t in _log_lines if n > after]
            last = _log_seq
        self._json({"lines": lines, "last": last, "file": LOG_FILE})

    def get_data(self, q):
        if not self.scanner.result_gz:
            return self._json({"error": "no scan yet"}, 404)
        if "gzip" in self.headers.get("Accept-Encoding", ""):
            self._send(200, self.scanner.result_gz, "application/json", {"Content-Encoding": "gzip"})
        else:
            self._send(200, gzip.decompress(self.scanner.result_gz))

    def get_file(self, q):
        eid = int(q["id"][0])
        gid = int(q["snap"][0]) if q.get("snap") else None
        _, _, path, _, _ = self._view(q).entry_paths(eid, gid)
        name = os.path.basename(path)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if q.get("text"):
            ctype = "text/plain; charset=utf-8"
        if os.path.islink(path):
            return self._send(200, f"symlink -> {os.readlink(path)}", "text/plain; charset=utf-8")
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                code = 206
        if q.get("text"):
            end = min(end, start + 256 * 1024 - 1)
        length = max(0, end - start + 1)
        safe = name.encode("ascii", "replace").decode().replace('"', "'")
        disp = "attachment" if q.get("dl") else "inline"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Disposition", f'{disp}; filename="{safe}"')
        if ctype.startswith(("text/html", "image/svg", "application/xhtml", "text/xml", "application/xml",
                             "application/octet-stream")):
            self.send_header("Content-Security-Policy", "sandbox")  # snapshot HTML/SVG can't run script
        self.send_header("X-Content-Type-Options", "nosniff")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _view(self, src):
        """The results a request is about: a saved record (`rec`), or the current scan."""
        rec = src.get("rec")
        if isinstance(rec, list):
            rec = rec[0]
        if rec:
            return self.records.view(rec)
        if not self.scanner.view:
            raise KeyError("no scan results yet")
        return self.scanner.view

    def get_versions(self, q):
        eid = int(q["id"][0])
        v = self._view(q)
        d, _, _, _, gids = v.entry_paths(eid)
        rel = v.entries[eid][1]
        versions = []
        for g in gids:
            snap = v.snaps[g][1]
            try:
                size, mtime = v.backend.stat(v.backend.snap_path(d, snap, rel))
            except OSError:
                continue
            if versions and versions[-1]["size"] == size and versions[-1]["mtime"] == mtime:
                versions[-1]["to"] = g
                versions[-1]["count"] += 1
            else:
                versions.append({"from": g, "to": g, "count": 1, "size": size, "mtime": mtime})
        self._json(versions)

    # -- POST ---------------------------------------------------------------
    def post_scan(self, body):
        ok = self.scanner.start(set(body.get("datasets") or []))
        self._json({"started": ok})

    def post_cancel(self, body):
        self.scanner.cancel.set()
        self._json({"ok": True})

    def _group_marked(self, gids):
        by_ds = {}
        for g in gids:
            di, s = self.scanner.snaps[int(g)]
            by_ds.setdefault(di, set()).add(s["name"])
        for di, names in sorted(by_ds.items()):
            d = self.scanner.datasets[di]
            order = [s["name"] for _, s in self.scanner.snaps[d["first"]:d["last"] + 1]]
            yield di, d, snap_spec(order, names), len(names)

    def post_estimate(self, body):
        out, total = [], 0
        for di, d, spec, n in self._group_marked(body.get("snaps", [])):
            dd = {"name": d["name"], "snaps": [s for _, s in self.scanner.snaps[d["first"]:d["last"] + 1]]}
            try:
                b = self.scanner.backend.estimate(dd, spec)
                err = None
            except Exception as e:  # noqa: BLE001
                b, err = None, str(e)
            total += b or 0
            out.append({"dataset": d["name"], "count": n, "bytes": b, "error": err})
        self._json({"datasets": out, "total": total})

    def post_script(self, body):
        kind = body.get("kind")
        lines = ["#!/bin/sh", f"# Generated by SnapSift {VERSION} on {time.ctime()}"]
        if kind == "destroy":
            lines += ["# REVIEW BEFORE RUNNING. Destroyed snapshots cannot be recovered."]
            n_snaps = len(body.get("snaps", []))
            try:  # keep a browsable record of what these snapshots held, in case you need it later
                meta = self.records.save(f"Before destroying {n_snaps} snapshot{'s' if n_snaps != 1 else ''}",
                                         note="Saved automatically when a destroy script was generated.",
                                         kind="destroy")
                lines.append(f"# A record of this scan was saved as '{meta['name']}' (Records tab), so you can")
                lines.append("# still browse what these snapshots contained after they're gone.")
            except Exception as e:  # noqa: BLE001
                lines.append(f"# WARNING: could not save a record of this scan first: {e}")
            lines += ["# Each line is preceded by a dry run (-n) so you can see what it would do.", ""]
            for di, d, spec, n in self._group_marked(body.get("snaps", [])):
                target = shlex.quote(f"{d['name']}@{spec}")
                lines += [f"# {d['name']}: {n} snapshot(s)", f"zfs destroy -nv {target}",
                          f"# zfs destroy -v {target}", ""]
            lines.append("# Uncomment the real destroy lines once the dry runs look right.")
        else:
            dest = (body.get("dest") or "").strip()
            lines += ["# Copies files out of snapshots. Never overwrites (cp -n).",
                      "set -u", ""]
            v = self._view(body)
            for eid in body.get("ids", []):
                d, snap, src, live, _ = v.entry_paths(int(eid))
                rel = v.entries[int(eid)][1]
                target = os.path.join(dest, d["name"].replace("/", "_"), *rel.split("/")) if dest else live
                target = target.replace("\\", "/")
                src = src.replace("\\", "/")
                lines.append(f"mkdir -p {shlex.quote(os.path.dirname(target))} && "
                             f"cp -an {shlex.quote(src)} {shlex.quote(target)}")
        text = "\n".join(lines) + "\n"
        self._send(200, text.encode("utf-8", "surrogateescape"), "text/plain; charset=utf-8")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Browse files that only exist in ZFS snapshots.")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--jobs", type=int, default=4, help="parallel zfs diff processes (default 4)")
    ap.add_argument("--method", choices=["diff", "walk"], default="diff",
                    help="diff = zfs diff (fast, needs root); walk = compare directory trees (slow, no root)")
    ap.add_argument("--engine", choices=["auto", "zfs", "tree"], default="auto",
                    help="how consecutive snapshots are compared: auto = zfs diff, switching to folder "
                         "listings for any diff that stalls; zfs = never switch; tree = always folder listings")
    ap.add_argument("--stall-minutes", type=float, default=5,
                    help="auto engine: switch when zfs diff makes <100 changes of progress in this long (default 5)")
    ap.add_argument("--log-every", type=float, default=60,
                    help="seconds between console status lines for long-running steps (default 60, 0 = off)")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="regex of dataset names to skip")
    ap.add_argument("--cache", default=None, help="scan cache file (default ~/.cache/snapsift-<method>.json.gz)")
    ap.add_argument("--no-token", action="store_true", help="disable the access token (only with --host 127.0.0.1!)")
    ap.add_argument("--demo", action="store_true", help="run against a generated fake pool")
    args = ap.parse_args()

    if hasattr(signal, "SIGUSR1"):  # `kill -USR1 <pid>` prints every thread's stack, for diagnosing hangs
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    if args.demo:
        print("Building demo pool...")
        backend = DemoBackend()
    elif args.method == "walk":
        backend = WalkBackend(args.exclude)
    else:
        backend = ZfsBackend(args.exclude)
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            print("WARNING: not running as root; `zfs diff` will probably be denied. Use sudo, "
                  "`zfs allow <user> diff,mount <dataset>`, or --method walk.")

    cache = args.cache
    if cache is None and not args.demo:
        cache = os.path.join(os.path.expanduser("~"), ".cache", f"snapsift-{backend.name}.json.gz")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
    global LOG_FILE
    if cache:
        LOG_FILE = os.path.join(os.path.dirname(cache), "snapsift.log")
    log(f"SnapSift {VERSION} started ({backend.name})" + (f", log file {LOG_FILE}" if LOG_FILE else ""))

    engine = "tree" if backend.name == "walk" else args.engine
    Handler.scanner = Scanner(backend, args.jobs, cache, engine, int(args.stall_minutes * 60), args.log_every)
    Handler.recovery = Recovery(Handler.scanner)
    rec_dir = os.path.join(os.path.dirname(cache), "snapsift-records") if cache else \
        os.path.join(tempfile.gettempdir(), "snapsift-demo-records")
    Handler.records = Records(Handler.scanner, rec_dir)
    Handler.scanner.after_scan = lambda: Handler.records.save("", "Saved automatically after a scan.", "auto")
    warn_if_boot_pool(rec_dir)
    Handler.token = None if args.no_token else secrets.token_urlsafe(18)
    Handler.html = INDEX_HTML.encode("utf-8")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    host = args.host
    if host in ("0.0.0.0", "::"):
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            host = s.getsockname()[0]
            s.close()
        except OSError:
            host = "localhost"
    suffix = f"/?t={Handler.token}" if Handler.token else "/"
    print(f"\nSnapSift {VERSION}  ({backend.name})  read-only")
    print(f"  open:  http://{host}:{args.port}{suffix}\n  Ctrl+C to stop\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        Handler.scanner.cancel.set()
        print("bye")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SnapSift</title>
<style>
:root{
  color-scheme:light;
  --bg:#f4f4f2;--surface:#ffffff;--surface-2:#f0efec;--surface-3:#e8e7e3;--border:#e2e1dd;
  --text:#0b0b0b;--text-2:#52514e;--muted:#8a8984;
  --accent:#2a78d6;--accent-soft:rgba(42,120,214,.12);--accent-ink:#ffffff;
  --danger:#d23b3a;--danger-soft:rgba(210,59,58,.11);--warn:#a86e00;--warn-soft:rgba(237,161,0,.14);--ok:#13865a;
  --c-image:#2a78d6;--c-video:#eb6834;--c-audio:#1baf7a;--c-doc:#eda100;--c-archive:#e87ba4;--c-code:#4a3aa7;--c-other:#a3a29c;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 4px 16px rgba(0,0,0,.05);
  --row-h:34px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --bg:#111110;--surface:#1a1a19;--surface-2:#222220;--surface-3:#2b2b29;--border:#2f2f2c;
  --text:#ffffff;--text-2:#c3c2b7;--muted:#8d8c84;
  --accent:#3987e5;--accent-soft:rgba(57,135,229,.18);
  --danger:#e66767;--danger-soft:rgba(230,103,103,.15);--warn:#e0a52a;--warn-soft:rgba(201,133,0,.2);--ok:#3cc68f;
  --c-image:#3987e5;--c-video:#d95926;--c-audio:#199e70;--c-doc:#c98500;--c-archive:#d55181;--c-code:#9085e9;--c-other:#6f6e68;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 4px 16px rgba(0,0,0,.25);
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#111110;--surface:#1a1a19;--surface-2:#222220;--surface-3:#2b2b29;--border:#2f2f2c;
  --text:#ffffff;--text-2:#c3c2b7;--muted:#8d8c84;
  --accent:#3987e5;--accent-soft:rgba(57,135,229,.18);
  --danger:#e66767;--danger-soft:rgba(230,103,103,.15);--warn:#e0a52a;--warn-soft:rgba(201,133,0,.2);--ok:#3cc68f;
  --c-image:#3987e5;--c-video:#d95926;--c-audio:#199e70;--c-doc:#c98500;--c-archive:#d55181;--c-code:#9085e9;--c-other:#6f6e68;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 4px 16px rgba(0,0,0,.25);
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:13.5px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;display:flex;flex-direction:column;overflow:hidden}
button,input,select,textarea{font:inherit;color:inherit}
button{cursor:pointer;border:1px solid var(--border);background:var(--surface);border-radius:8px;padding:6px 12px;transition:background .12s,border-color .12s}
button:hover{background:var(--surface-2)}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);font-weight:600}
button.primary:hover{filter:brightness(1.08)}
button.danger{color:var(--danger);border-color:color-mix(in srgb,var(--danger) 40%,var(--border))}
button.ghost{border-color:transparent;background:transparent}
button.ghost:hover{background:var(--surface-2)}
button:disabled{opacity:.5;cursor:default}
input[type=text],input[type=search],input[type=date],select{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:6px 10px;outline:none}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
.muted{color:var(--muted)}
.sec{color:var(--text-2)}

/* header */
header{display:flex;align-items:center;gap:16px;padding:10px 16px;background:var(--surface);border-bottom:1px solid var(--border);flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:15px;letter-spacing:-.01em;white-space:nowrap}
.brand svg{flex:none}
.brand small{font-weight:400;color:var(--muted);font-size:12px}
.search{flex:1 1 300px;max-width:620px;min-width:200px;position:relative}
.search input{width:100%;padding:8px 12px 8px 34px;border-radius:10px;background:var(--surface-2);border-color:transparent}
.search input:focus{background:var(--surface)}
.search svg{position:absolute;left:11px;top:50%;transform:translateY(-50%);color:var(--muted)}
.search kbd{position:absolute;right:10px;top:50%;transform:translateY(-50%)}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--border);border-bottom-width:2px;border-radius:4px;padding:0 5px;color:var(--muted);background:var(--surface)}
.hdr-right{display:flex;align-items:center;gap:8px}
#scanInfo{font-size:12px;color:var(--muted);white-space:nowrap}

/* progress */
/* tabs & views */
.tabs{display:flex;gap:2px;background:var(--surface-2);border-radius:10px;padding:3px}
.tabs button{border:0;background:transparent;padding:5px 14px;border-radius:8px;color:var(--text-2);font-weight:550}
.tabs button:hover{color:var(--text);background:transparent}
.tabs button.on{background:var(--surface);color:var(--text);box-shadow:0 1px 2px rgba(0,0,0,.15)}
.hsp{flex:1}
[hidden]{display:none!important}
.pill{display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600}
.view{display:none;flex:1;min-height:0;flex-direction:column}
.view.on{display:flex}
#view-files .layout{flex:1}
.small{font-size:12px}
.scanpage{flex:1;overflow:auto;padding:16px;display:grid;grid-template-columns:minmax(300px,400px) minmax(0,1fr);gap:16px;align-content:start;align-items:start}
.card.pad{padding:16px 18px}
.card.pad h2{margin:0 0 6px}
.card.pad>p{margin:0 0 12px;font-size:12.5px}
.dsbtns{display:flex;gap:6px;align-items:center;margin-top:12px}
.dsbtns .sp{flex:1}
#dsList{max-height:52vh;overflow:auto;border-top:1px solid var(--border)}
#dsList .dsrow{grid-template-columns:20px minmax(0,1fr);align-items:start}
#dsList .dsn{display:flex;flex-direction:column;gap:2px;min-width:0}
#dsList .dsn>*{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#dsList .m{font-size:12px;color:var(--text-2)}
.empty a,.idle a{color:var(--accent)}
.idle{padding:30px 10px;text-align:center}
.idle h3{margin:0 0 6px;font-weight:600}
.idle a{color:var(--accent)}
.conpage{flex:1;min-height:0;display:flex;flex-direction:column;gap:10px;padding:14px 16px}
.contools{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.contools .sp{flex:1}
#conText{flex:1;min-height:0;margin:0;overflow:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px;font-size:12px;line-height:1.5;white-space:pre;user-select:text}
#progress{display:none;flex-direction:column;gap:10px;font-size:12.5px}
#progress.on{display:flex}
#progress.on+.idle,#progCard:has(#progress.on) .idle{display:none}
#progCard.lost{background:linear-gradient(var(--danger-soft),var(--danger-soft)),var(--surface)}
@media (max-width:900px){.scanpage{grid-template-columns:1fr}}
#progress.lost .tasks,#progress.lost .steps{opacity:.5}
.prow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.prow .sp{flex:1}
.live{display:inline-flex;align-items:center;gap:8px;font-weight:650;font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted);flex:none;position:relative;display:inline-block}
.dot.ok{background:var(--ok)}
.dot.ok::after{content:"";position:absolute;inset:-4px;border-radius:50%;border:2px solid var(--ok);animation:ping 1.6s ease-out infinite}
.dot.quiet{background:var(--warn)}
.dot.bad{background:var(--danger)}
@keyframes ping{0%{transform:scale(.55);opacity:.9}100%{transform:scale(1.5);opacity:0}}
.steps{display:flex;gap:2px;height:16px}
.steps i{flex:1 1 0;min-width:1px;border-radius:3px;background:var(--surface-3)}
.steps i.d{background:var(--c-image);animation:breathe 1.2s ease-in-out infinite}
.steps i.t{background:var(--c-code);animation:breathe 1.2s ease-in-out infinite}
.steps i.s{background:var(--c-audio);animation:breathe 1.2s ease-in-out infinite}
.steps i.w{background:color-mix(in srgb,var(--c-image) 35%,var(--surface-3))}
.steps i.p{background:repeating-linear-gradient(45deg,var(--c-doc) 0 5px,color-mix(in srgb,var(--c-doc) 45%,var(--surface-3)) 5px 10px);background-size:14.14px 14.14px;animation:march .7s linear infinite}
.steps i.k{background:var(--ok)}
.steps i.c{background:color-mix(in srgb,var(--ok) 50%,var(--surface-3))}
.steps i.f{background:var(--danger)}
@keyframes breathe{50%{opacity:.4}}
@keyframes march{to{background-position:14.14px 0}}
.legend{display:flex;gap:4px 14px;flex-wrap:wrap;color:var(--muted);font-size:11.5px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block}
.tasks{display:flex;flex-direction:column;gap:6px}
#conText .w{color:var(--danger)}
#conText .ph{color:var(--accent);font-weight:600}
#conText .st{color:var(--text-2)}
#pWarn:empty,#pWait:empty{display:none}
#pWait{background:var(--surface-2);border-radius:8px;padding:6px 10px}
#pWarn details{background:var(--warn-soft);border-radius:8px;padding:6px 10px}
#pWarn summary{cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#pWarn .mono{white-space:pre-wrap;margin-top:6px;max-height:200px;overflow:auto}
.task{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:4px 12px;align-items:center;padding:8px 10px;border:1px solid var(--border);border-radius:10px;background:var(--surface-2)}
.kchip{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;font-weight:700;letter-spacing:.05em;padding:2px 8px;border-radius:6px;white-space:nowrap;color:var(--text);background:color-mix(in srgb,var(--kc) 20%,transparent)}
.kchip i{width:8px;height:8px;border-radius:2px;background:var(--kc)}
.task .tl{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tstate{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--text-2);white-space:nowrap}
.tmeta{grid-column:2/4;display:flex;gap:4px 14px;flex-wrap:wrap;color:var(--text-2);font-size:12px;font-variant-numeric:tabular-nums}
.tmeta b{color:var(--text);font-weight:600}
.tbar{grid-column:2/4;height:4px;border-radius:3px;background:var(--surface-3);overflow:hidden}
.tbar b{display:block;height:100%;background:var(--kc);border-radius:3px;transition:width .4s}
@media (prefers-reduced-motion:reduce){.steps i,.dot.ok::after{animation:none!important}}
#errbar{display:none;padding:7px 16px;background:var(--warn-soft);font-size:12.5px;border-bottom:1px solid var(--border)}
#errbar.on{display:block}
#errbar details{display:inline}

/* layout */
.layout{flex:1;display:grid;grid-template-columns:330px minmax(0,1fr) auto;min-height:0}
aside.side{border-right:1px solid var(--border);background:var(--surface);display:flex;flex-direction:column;min-height:0}
.side-h{padding:14px 14px 10px;border-bottom:1px solid var(--border)}
.side-h h2,.card h2{margin:0;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
.side-h p{margin:6px 0 10px;color:var(--text-2);font-size:12.5px}
.side-tools{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.side-tools input[type=date]{padding:4px 8px;font-size:12px}
.side-tools button{padding:4px 10px;font-size:12px}
#snaps{flex:1;overflow:auto;padding:4px 0 8px}
.dsg>summary{list-style:none;cursor:pointer;padding:9px 14px;display:flex;align-items:center;gap:8px;font-weight:600;position:sticky;top:0;background:var(--surface);z-index:1;border-bottom:1px solid var(--border)}
.dsg>summary::-webkit-details-marker{display:none}
.dsg>summary .chev{transition:transform .15s;color:var(--muted)}
.dsg[open]>summary .chev{transform:rotate(90deg)}
.dsg>summary .n{margin-left:auto;font-weight:400;color:var(--muted);font-size:12px}
.dsg>summary button{padding:1px 8px;font-size:11px;font-weight:400}
.snap{display:grid;grid-template-columns:18px 1fr auto;gap:2px 8px;padding:6px 14px 6px 14px;cursor:pointer;border-left:3px solid transparent;user-select:none}
.snap:hover{background:var(--surface-2)}
.snap input{margin:2px 0 0;accent-color:var(--danger);pointer-events:none}  /* the row handles clicks; see #snaps onclick */
.snap .sn{font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.snap .sd{font-size:11.5px;color:var(--muted);text-align:right;white-space:nowrap}
.snap .sh{grid-column:2/4;display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--text-2)}
.snap .hb{flex:1;height:4px;background:var(--surface-3);border-radius:3px;overflow:hidden}
.snap .hb b{display:block;height:100%;background:var(--accent);border-radius:3px}
.snap.m{background:var(--danger-soft);border-left-color:var(--danger)}
.snap.m .sn{text-decoration:line-through;text-decoration-color:var(--danger)}
.snap.m .hb b{background:var(--danger)}
.snap .only{color:var(--warn);font-weight:600}
.sumcard{border-top:1px solid var(--border);padding:12px 14px;background:var(--surface);display:flex;flex-direction:column;gap:8px}
.sumcard .big{font-size:13px}
.sumcard .big b{font-size:18px;letter-spacing:-.01em}
.sumcard .lostline{padding:8px 10px;border-radius:8px;background:var(--surface-2);font-size:12.5px}
.sumcard .lostline.bad{background:var(--danger-soft);color:var(--text)}
.sumcard .lostline.good{color:var(--ok)}
.sumcard .btns{display:flex;gap:6px;flex-wrap:wrap}
.sumcard .btns button{font-size:12px;padding:5px 10px}
#estOut{font-size:12px;color:var(--text-2)}

/* content */
.content{display:flex;flex-direction:column;min-width:0;min-height:0;padding:14px 16px 0;gap:12px}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:10px 14px;box-shadow:var(--shadow);min-width:0}
.tile .l{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:600}
.tile .v{font-size:22px;font-weight:650;letter-spacing:-.02em;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .s{font-size:12px;color:var(--text-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile.bad{border-color:color-mix(in srgb,var(--danger) 45%,var(--border));background:linear-gradient(var(--danger-soft),var(--danger-soft)),var(--surface)}
.tile.bad .v{color:var(--danger)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow)}
.chartcard{padding:10px 14px 6px}
.chartcard .hd{display:flex;align-items:center;gap:10px}
.chartcard .hd .sub{color:var(--muted);font-size:12px}
.chartcard .hd .sp{flex:1}
.seg{display:inline-flex;background:var(--surface-2);border-radius:8px;padding:2px}
.seg button{border:0;background:transparent;padding:3px 10px;font-size:12px;border-radius:6px;color:var(--text-2)}
.seg button.on{background:var(--surface);color:var(--text);box-shadow:0 1px 2px rgba(0,0,0,.12)}
#chart{display:block;width:100%;height:150px;margin-top:4px}
#chart .bar{fill:var(--accent)}
#chart .bar.dim{opacity:.28}
#chart .hit{fill:transparent;cursor:pointer}
#chart .hit:hover+.bar,#chart .bar.hov{filter:brightness(1.15)}
#chart .grid{stroke:var(--border);stroke-width:1}
#chart .axis{fill:var(--muted);font-size:10.5px}
#chart .snapt{stroke:var(--muted);stroke-width:1;opacity:.5}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);box-shadow:var(--shadow);border-radius:8px;padding:7px 10px;font-size:12px;z-index:50;display:none;max-width:320px}
#tip b{font-weight:650}

.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.chips{display:flex;gap:4px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer;user-select:none;color:var(--text-2)}
.chip i{width:8px;height:8px;border-radius:2px;display:inline-block}
.chip .n{color:var(--muted);font-size:11px}
.chip.on{background:var(--accent-soft);border-color:var(--accent);color:var(--text)}
.chip.warnchip.on{background:var(--danger-soft);border-color:var(--danger)}
.toolbar .sp{flex:1}
.toolbar select{padding:4px 8px;font-size:12px}
.filterpill{display:inline-flex;align-items:center;gap:6px;background:var(--accent-soft);border-radius:999px;padding:3px 6px 3px 10px;font-size:12px}
.filterpill button{border:0;background:transparent;padding:0 4px;color:var(--text-2)}

.listcard{flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden;border-bottom-left-radius:0;border-bottom-right-radius:0;border-bottom:0}
.lhead,.row{display:grid;grid-template-columns:30px minmax(0,1fr) 128px 118px 150px;align-items:center;column-gap:10px;padding:0 12px 0 8px}
.lhead{height:32px;border-bottom:1px solid var(--border);font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:600}
.lhead span[data-sort]{cursor:pointer}
.lhead span[data-sort]:hover{color:var(--text)}
.lhead .on{color:var(--text)}
#list{flex:1;overflow:auto;position:relative;outline:none}
#spacer{position:relative}
#rows{position:absolute;left:0;right:0;top:0}
.row{height:var(--row-h);border-bottom:1px solid color-mix(in srgb,var(--border) 55%,transparent);cursor:default;white-space:nowrap}
.row:hover{background:var(--surface-2)}
.row.active{background:var(--accent-soft)}
.row.sel{background:color-mix(in srgb,var(--accent) 7%,transparent)}
.row.lostrow .nm{color:var(--danger)}
.row .name{display:flex;align-items:center;gap:7px;min-width:0;padding-left:calc(var(--d,0) * 18px)}
.row .nm{overflow:hidden;text-overflow:ellipsis;min-width:0}
.row.dir .nm{font-weight:550}
.row.ds .nm{font-weight:700}
.row .dirp{color:var(--muted);overflow:hidden;text-overflow:ellipsis;min-width:0;font-size:12px;flex:1 1 0}
.row .cnt{color:var(--muted);font-size:11.5px;flex:none}
.chev{display:inline-block;width:14px;text-align:center;color:var(--muted);transition:transform .12s;flex:none;font-size:11px}
.chev.open{transform:rotate(90deg)}
.fold{width:16px;height:13px;flex:none;color:var(--muted)}
.ext{flex:none;display:inline-flex;align-items:center;gap:5px;font-size:10.5px;font-weight:600;color:var(--text-2);background:var(--surface-2);border-radius:5px;padding:1px 6px;min-width:44px;text-transform:uppercase;letter-spacing:.02em}
.ext i{width:6px;height:6px;border-radius:2px;flex:none}
.tag{flex:none;font-size:10.5px;border-radius:5px;padding:1px 6px;font-weight:600}
.tag.lost{background:var(--danger-soft);color:var(--danger)}
.tag.moved{background:var(--surface-3);color:var(--text-2)}
.tag.repl{background:var(--warn-soft);color:var(--warn)}
.sz{display:flex;align-items:center;gap:8px;justify-content:flex-end;font-variant-numeric:tabular-nums;font-size:12.5px}
.sz .bar{width:40px;height:4px;border-radius:3px;background:var(--surface-3);overflow:hidden;flex:none}
.sz .bar b{display:block;height:100%;background:var(--accent);border-radius:3px}
.when{font-size:12px;color:var(--text-2);font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis}
.pres{position:relative;height:8px;border-radius:4px;background:var(--surface-3)}
.pres b{position:absolute;top:0;bottom:0;background:var(--accent);border-radius:4px;min-width:3px}
.pres b.gone{background:transparent;border:1px dashed var(--muted);border-left:0;border-radius:0 4px 4px 0}
.lostrow .pres b{background:var(--danger)}
.ck{width:16px;height:16px;border:1.5px solid var(--muted);border-radius:4px;display:inline-block;cursor:pointer;position:relative;flex:none}
.ck.checked,.ck.mixed{background:var(--accent);border-color:var(--accent)}
.ck.checked::after{content:"";position:absolute;left:4px;top:1px;width:4px;height:8px;border:solid #fff;border-width:0 2px 2px 0;transform:rotate(45deg)}
.ck.mixed::after{content:"";position:absolute;left:3px;right:3px;top:6px;height:2px;background:#fff}
.empty{padding:60px 20px;text-align:center;color:var(--muted)}
.empty h3{color:var(--text);margin:0 0 6px;font-weight:600}

.selbar{display:none;align-items:center;gap:10px;padding:8px 12px;border-top:1px solid var(--border);background:var(--surface-2)}
.selbar.on{display:flex}
.selbar .sp{flex:1}

/* drawer */
.drawer{width:0;overflow:hidden;border-left:1px solid var(--border);background:var(--surface);transition:width .18s ease;display:flex;flex-direction:column;min-height:0}
.drawer.on{width:440px}
.drawer .inner{width:440px;display:flex;flex-direction:column;min-height:0;height:100%}
.dh{display:flex;align-items:flex-start;gap:8px;padding:14px 16px 10px;border-bottom:1px solid var(--border)}
.dh h3{margin:0;font-size:15px;word-break:break-all;flex:1;line-height:1.35}
.dbody{overflow:auto;padding:14px 16px;display:flex;flex-direction:column;gap:14px}
.preview{border-radius:10px;background:var(--surface-2);min-height:120px;display:flex;align-items:center;justify-content:center;overflow:hidden;border:1px solid var(--border)}
.preview img,.preview video{max-width:100%;max-height:340px;display:block}
.preview audio{width:100%;margin:16px}
.preview iframe{width:100%;height:420px;border:0;background:#fff}
.preview pre{margin:0;padding:12px;width:100%;max-height:340px;overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-word;align-self:stretch}
.preview .np{color:var(--muted);padding:30px;text-align:center}
.kv{display:grid;grid-template-columns:118px 1fr;gap:6px 12px;font-size:12.5px}
.kv dt{color:var(--muted)}
.kv dd{margin:0;word-break:break-all}
.dacts{display:flex;gap:6px;flex-wrap:wrap}
.dacts button,.dacts a{font-size:12px;padding:5px 10px}
a.btn{display:inline-block;text-decoration:none;color:inherit;border:1px solid var(--border);border-radius:8px;background:var(--surface)}
a.btn.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600}
.strip{display:flex;gap:2px;flex-wrap:wrap}
.strip i{width:8px;height:14px;border-radius:2px;background:var(--surface-3)}
.strip i.in{background:var(--accent)}
.strip i.mk{outline:1.5px solid var(--danger);outline-offset:-1.5px}
.strip i.in.mk{background:var(--danger)}
.strip i.gn{opacity:.3}
.vers{display:flex;flex-direction:column;gap:6px}
.ver{display:flex;align-items:center;gap:8px;padding:7px 10px;border:1px solid var(--border);border-radius:8px;font-size:12px}
.ver .vi{flex:1;min-width:0}
.ver .vi div{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ver button,.ver a{font-size:11.5px;padding:3px 8px}
.ver.cur{border-color:var(--accent)}
h4{margin:0 0 6px;font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}

/* dialogs */
dialog{border:1px solid var(--border);border-radius:14px;background:var(--surface);color:var(--text);padding:0;box-shadow:0 20px 60px rgba(0,0,0,.3);width:min(720px,calc(100vw - 32px))}
dialog::backdrop{background:rgba(0,0,0,.4);backdrop-filter:blur(2px)}
.dlg-h{padding:16px 20px 6px}
.dlg-h h2{margin:0;font-size:17px}
.dlg-h p{margin:6px 0 0;color:var(--text-2)}
.dlg-b{padding:10px 20px;max-height:60vh;overflow:auto}
.dlg-f{padding:12px 20px 16px;display:flex;gap:8px;justify-content:flex-end;align-items:center}
.dlg-f .sp{flex:1}
.dsrow{display:grid;grid-template-columns:20px 1fr auto auto;gap:12px;align-items:center;padding:8px 6px;border-bottom:1px solid var(--border);cursor:pointer}
.dsrow:hover{background:var(--surface-2)}
.dsrow .m{font-size:12px;color:var(--muted);white-space:nowrap}
textarea{width:100%;height:46vh;background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:10px;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;resize:vertical;white-space:pre}
.opts{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.opts input[type=text]{flex:1;min-width:200px}
#recBanner{display:flex;align-items:center;gap:12px;padding:9px 16px;background:color-mix(in srgb,var(--c-code) 16%,var(--surface));border-bottom:1px solid var(--border);font-size:12.5px}
#recBanner .sp{flex:1}
#recBanner b{font-weight:650}
#startWarn:empty{display:none}
#startWarn{background:var(--warn-soft);border-radius:8px;padding:8px 10px;font-size:12.5px;margin-bottom:12px}
#rsNote{height:auto;resize:vertical;font:inherit;white-space:pre-wrap}
.recrow{border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin-top:8px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 12px;align-items:start}
.recrow.cur{border-color:var(--c-code);box-shadow:0 0 0 1px var(--c-code)}
.recrow .t{font-weight:650;overflow:hidden;text-overflow:ellipsis}
.recrow .m{font-size:12px;color:var(--text-2);grid-column:1}
.recrow .note{font-size:12px;color:var(--muted);grid-column:1/3;white-space:pre-wrap}
.recrow .acts{display:flex;gap:6px;grid-row:1/3;grid-column:2;align-self:center}
.recrow .acts button,.recrow .acts a{font-size:12px;padding:4px 10px}
.kbadge{display:inline-block;font-size:10.5px;font-weight:600;padding:1px 7px;border-radius:5px;margin-left:6px;background:var(--surface-3);color:var(--text-2);vertical-align:1px}
.kbadge.destroy{background:var(--danger-soft);color:var(--danger)}
.kbadge.manual{background:color-mix(in srgb,var(--c-code) 18%,transparent);color:var(--text)}
.tag.gone{background:var(--surface-3);color:var(--danger)}
.snap.gone{opacity:.55}
.snap.gone .sn{text-decoration:line-through}
.fld{display:flex;flex-direction:column;gap:5px;margin-bottom:10px;font-size:12.5px;color:var(--text-2)}
.fld input{font-size:13px;padding:8px 10px}
.recopts{gap:6px 18px;font-size:12.5px}
.recinfo{border:1px solid var(--border);border-radius:10px;padding:10px 12px;font-size:12.5px;display:flex;flex-direction:column;gap:6px;min-height:52px}
.recinfo .err{color:var(--danger);font-weight:600}
.recinfo .warn{color:var(--warn)}
.recinfo .ex{font-size:11.5px;display:grid;grid-template-columns:minmax(0,1fr);gap:2px;color:var(--text-2)}
.recinfo .ex div{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#recPanel{position:fixed;right:16px;bottom:16px;width:min(400px,calc(100vw - 32px));z-index:60;padding:12px 14px;display:flex;flex-direction:column;gap:8px}
#recPanel .rh{display:flex;align-items:center;gap:8px}
#recPanel .rh .sp{flex:1}
#recPanel .rh button{padding:2px 8px;font-size:12px}
#recPanel .tbar{height:6px}
#recText{word-break:break-all}
#recErr details{font-size:12px}
#recErr .mono{white-space:pre-wrap;max-height:160px;overflow:auto;margin-top:4px}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--text);color:var(--bg);padding:8px 16px;border-radius:10px;font-size:12.5px;z-index:99;opacity:0;transition:opacity .2s;pointer-events:none}
.toast.on{opacity:1}

@media (max-width:1100px){
  .layout{grid-template-columns:280px minmax(0,1fr)}
  .drawer.on{position:fixed;right:0;top:0;bottom:0;z-index:40;width:min(440px,100vw);box-shadow:var(--shadow)}
  .drawer .inner{width:min(440px,100vw)}
  .lhead,.row{grid-template-columns:30px minmax(0,1fr) 110px 100px}
  .lhead>:nth-child(5),.row>:nth-child(5){display:none}
}
@media (max-width:760px){
  body{overflow:auto}
  .layout{display:flex;flex-direction:column}
  aside.side{max-height:45vh;border-right:0;border-bottom:1px solid var(--border)}
  .content{padding:12px 16px 0;min-height:80vh}
  .tiles{grid-template-columns:repeat(2,minmax(0,1fr))}
  .lhead,.row{grid-template-columns:30px minmax(0,1fr) 92px}
  .lhead>:nth-child(4),.row>:nth-child(4){display:none}
  .brand small{display:none}
}
</style>
</head>
<body>
<header>
  <div class="brand">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7l9-4 9 4-9 4-9-4z" stroke="var(--accent)"/><path d="M3 12l9 4 9-4" opacity=".6"/><path d="M3 17l9 4 9-4" opacity=".35"/></svg>
    SnapSift
  </div>
  <nav class="tabs" id="tabs" role="tablist">
    <button role="tab" data-tab="files">Files</button>
    <button role="tab" data-tab="scan">Scan</button>
    <button role="tab" data-tab="records">Records</button>
    <button role="tab" data-tab="console">Console</button>
  </nav>
  <div class="search" id="searchBox">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
    <input id="q" type="search" placeholder="Search deleted files - e.g. taxes 2023 pdf" autocomplete="off" spellcheck="false">
    <kbd>/</kbd>
  </div>
  <span class="hsp"></span>
  <div class="hdr-right">
    <button id="scanPill" class="pill" hidden title="Open the Scan tab"><span class="dot ok"></span><span id="scanPillText">Scanning</span></button>
    <span id="scanInfo"></span>
    <button id="themeBtn" class="ghost" title="Theme" aria-label="Toggle theme">&#9680;</button>
  </div>
</header>

<!-- ===== Files ===== -->
<div class="view" id="view-files">
<div id="recBanner" hidden></div>
<div id="errbar"></div>
<div class="layout">
  <aside class="side">
    <div class="side-h">
      <h2>Snapshots</h2>
      <p>Tick the snapshots you're thinking of destroying. Every deleted file that exists <i>only</i> in ticked snapshots turns red &mdash; that's what you'd lose for good.</p>
      <div class="side-tools">
        <span class="muted" style="font-size:12px">Older than</span>
        <input type="date" id="olderThan">
        <button id="markOlder">Tick</button>
        <button id="clearMarks" class="ghost">Clear</button>
      </div>
    </div>
    <div id="snaps"></div>
    <div class="sumcard" id="sumcard"></div>
  </aside>

  <section class="content">
    <div class="tiles" id="tiles"></div>
    <div class="card chartcard">
      <div class="hd">
        <h2>When files disappeared</h2>
        <span class="sub" id="chartSub"></span>
        <span class="sp"></span>
        <span id="timePill"></span>
        <div class="seg" id="metricSeg"><button data-m="size" class="on">Size</button><button data-m="count">Files</button></div>
      </div>
      <svg id="chart" role="img" aria-label="Deleted data over time"></svg>
    </div>
    <div class="toolbar">
      <div class="seg" id="viewSeg"><button data-v="tree" class="on">Folders</button><button data-v="list">Flat list</button></div>
      <div class="chips" id="catChips"></div>
      <span class="sp"></span>
      <div class="chips" id="kindChips"></div>
      <select id="minSize" title="Minimum size">
        <option value="0">Any size</option><option value="1048576">&ge; 1 MB</option><option value="10485760">&ge; 10 MB</option>
        <option value="104857600">&ge; 100 MB</option><option value="1073741824">&ge; 1 GB</option>
      </select>
    </div>
    <div class="card listcard">
      <div class="lhead"><span></span><span data-sort="name">Name</span><span data-sort="size" style="text-align:right">Size</span><span data-sort="when">Gone since</span><span>In snapshots</span></div>
      <div id="list" tabindex="0"><div id="spacer"><div id="rows"></div></div></div>
      <div class="selbar" id="selbar"><span id="selText"></span><span class="sp"></span><button class="ghost" id="selClear">Clear</button><button id="selRestore">Restore script&hellip;</button><button class="primary" id="selRecover">Recover&hellip;</button></div>
    </div>
  </section>

  <aside class="drawer" id="drawer"><div class="inner" id="drawerInner"></div></aside>
</div>
</div>

<!-- ===== Scan ===== -->
<div class="view" id="view-scan">
  <div class="scanpage">
    <section class="card pad" id="dsCard">
      <div id="startWarn"></div>
      <h2>Datasets</h2>
      <p class="sec">Each snapshot is compared with the next one using <code>zfs diff</code>, newest to oldest. Nothing is modified. Finished steps are saved, so a restart picks up where it left off.</p>
      <div id="dsList"><div class="muted">Loading datasets&hellip;</div></div>
      <div class="dsbtns"><button class="ghost" id="dsAll">All</button><button class="ghost" id="dsNone">None</button><span class="sp"></span><button class="primary" id="dsGo">Start scan</button></div>
      <p class="muted small" id="lastScan"></p>
    </section>
    <section class="card pad" id="progCard">
      <div id="scanIdle" class="idle"><h3>No scan running</h3><p class="sec">Pick datasets and press <b>Start scan</b>. Live progress appears here, and the full log is on the <a href="#console" data-goto="console">Console</a> tab.</p></div>
      <div id="progress" aria-live="polite">
        <div class="prow"><span class="live" id="pLive"><span class="dot"></span>Starting</span><b id="pPhase"></b><span class="sec" id="pText"></span><span class="sp"></span><button id="cancelBtn" class="danger">Cancel scan</button></div>
        <div class="steps" id="pSteps"></div>
        <div class="legend" id="pLegend"></div>
        <div id="pWait" class="sec small"></div>
        <div id="pWarn"></div>
        <div class="tasks" id="pTasks"></div>
      </div>
    </section>
  </div>
</div>

<!-- ===== Records ===== -->
<div class="view" id="view-records">
  <div class="scanpage">
    <section class="card pad">
      <h2>Save the current scan</h2>
      <p class="sec">A record is a frozen copy of these results: every file, where it was, its size and dates, and which snapshots held it. Opening one needs no scanning or lookups, and it stays browsable after the snapshots are destroyed, so you can always check what they contained.</p>
      <label class="fld"><span>Name</span><input id="rsName" type="text" placeholder="e.g. Before pruning 2024 snapshots" autocomplete="off"></label>
      <label class="fld"><span>Note (optional)</span><textarea id="rsNote" rows="3" placeholder="Why you saved it, what you were about to do&hellip;"></textarea></label>
      <div class="dsbtns"><span class="sp"></span><button class="primary" id="rsSave">Save record</button></div>
      <p class="muted small">A record is saved automatically after every scan (the last 5 are kept) and whenever you generate a destroy script (kept until you delete it). Records you save here are kept until you delete them.</p>
    </section>
    <section class="card pad">
      <h2>Saved records</h2>
      <p class="muted small" id="recDir"></p>
      <div id="recList"><div class="muted">Loading&hellip;</div></div>
    </section>
  </div>
</div>

<!-- ===== Console ===== -->
<div class="view" id="view-console">
  <div class="conpage">
    <div class="contools"><span class="sec" id="conNote">The same output as the terminal SnapSift runs in, updating live.</span><span class="sp"></span>
      <label class="muted small"><input type="checkbox" id="conFollow" checked> Follow new lines</label>
      <button id="conDl">Download .log</button><button id="conCopy">Copy all</button></div>
    <pre id="conText" class="mono" tabindex="0"></pre>
  </div>
</div>

<dialog id="recDlg">
  <div class="dlg-h"><h2 id="recTitle">Recover files</h2><p>Copies the files out of their snapshot into a folder on the NAS, keeping their original folder structure. Nothing is ever overwritten: files that already exist there are skipped.</p></div>
  <div class="dlg-b">
    <label class="fld"><span>Destination folder</span><input type="text" id="recDest" class="mono" placeholder="/mnt/tank/recovered" spellcheck="false" autocomplete="off"></label>
    <div class="opts recopts">
      <label><input type="radio" name="recLayout" value="dataset" checked> <span class="mono">folder/<b>dataset</b>/original/path</span></label>
      <label><input type="radio" name="recLayout" value="paths"> <span class="mono">folder/original/path</span></label>
    </div>
    <div id="recInfo" class="recinfo"></div>
  </div>
  <div class="dlg-f"><span class="sp"></span><button id="recCancelDlg">Cancel</button><button class="primary" id="recGo" disabled>Recover</button></div>
</dialog>

<div id="recPanel" class="card" hidden aria-live="polite">
  <div class="rh"><span class="dot ok" id="recDot"></span><b id="recHead">Recovering</b><span class="sp"></span><button id="recStop" class="ghost">Cancel</button><button id="recClose" class="ghost" title="Dismiss" hidden>&#x2715;</button></div>
  <div class="tbar" style="--kc:var(--ok)"><b id="recBar"></b></div>
  <div id="recText" class="small sec"></div>
  <div id="recErr"></div>
</div>

<dialog id="scriptDlg">
  <div class="dlg-h"><h2 id="scTitle">Script</h2><p id="scDesc"></p></div>
  <div class="dlg-b">
    <div class="opts" id="scOpts">
      <label><input type="radio" name="dest" value="orig" checked> Original location</label>
      <label><input type="radio" name="dest" value="folder"> Into folder:</label>
      <input type="text" id="scDest" placeholder="/mnt/tank/restored" class="mono">
    </div>
    <textarea id="scText" readonly spellcheck="false"></textarea>
  </div>
  <div class="dlg-f"><span class="muted" id="scNote" style="font-size:12px"></span><span class="sp"></span><button id="scDl">Download .sh</button><button id="scCopy">Copy</button><button class="primary" id="scClose">Done</button></div>
</dialog>

<div id="tip"></div>
<div class="toast" id="toast"></div>

<script>
"use strict";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const nf=new Intl.NumberFormat();
const fmtNum=n=>nf.format(n);
function fmtBytes(n){if(n==null)return"–";const u=["B","KB","MB","GB","TB","PB"];let i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return(i===0?n:n<10?n.toFixed(1):n.toFixed(0))+" "+u[i]}
const fmtDate=t=>t?new Date(t*1000).toLocaleDateString(undefined,{year:"numeric",month:"short",day:"numeric"}):"–";
const fmtDT=t=>t?new Date(t*1000).toLocaleString(undefined,{year:"numeric",month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"}):"–";
function fmtAgo(t){const s=Date.now()/1000-t;if(s<90)return"just now";if(s<5400)return Math.round(s/60)+" min ago";if(s<129600)return Math.round(s/3600)+" h ago";return Math.round(s/86400)+" days ago"}
async function api(p,opt){const r=await fetch(p,opt);if(!r.ok){let m=await r.text();try{m=JSON.parse(m).error||m}catch(_){}throw new Error(m.slice(0,400))}return r}
const post=(p,b)=>api(p,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});
function toast(m){const t=$("#toast");t.textContent=m;t.classList.add("on");clearTimeout(toast.h);toast.h=setTimeout(()=>t.classList.remove("on"),1800)}
async function copy(text){try{await navigator.clipboard.writeText(text)}catch(_){const ta=document.createElement("textarea");ta.value=text;document.body.appendChild(ta);ta.select();document.execCommand("copy");ta.remove()}toast("Copied")}
const LS={get(k,d){try{const v=localStorage.getItem("snapsift."+k);return v==null?d:JSON.parse(v)}catch(_){return d}},set(k,v){try{localStorage.setItem("snapsift."+k,JSON.stringify(v))}catch(_){}}};

/* ---------- theme ---------- */
function applyTheme(){const t=LS.get("theme","auto");if(t==="auto")delete document.documentElement.dataset.theme;else document.documentElement.dataset.theme=t;$("#themeBtn").title="Theme: "+t}
$("#themeBtn").onclick=()=>{const o=["auto","light","dark"];LS.set("theme",o[(o.indexOf(LS.get("theme","auto"))+1)%3]);applyTheme();renderChart()};
applyTheme();

/* ---------- categories ---------- */
const CATS=[
 ["image","Images","jpg jpeg png gif webp heic heif tif tiff bmp raw cr2 cr3 nef arw dng orf rw2 raf svg psd avif jxl"],
 ["video","Video","mp4 mkv mov avi wmv m4v mts m2ts webm mpg mpeg 3gp flv vob ts"],
 ["audio","Audio","mp3 flac wav m4a aac ogg opus wma aiff aif alac mid"],
 ["doc","Docs","pdf doc docx xls xlsx ppt pptx odt ods odp txt md rtf csv epub mobi pages numbers key tex log eml msg"],
 ["archive","Archives","zip rar 7z tar gz tgz bz2 xz zst iso img dmg vmdk qcow2 vhd vhdx bak tbz"],
 ["code","Code","py js mjs ts tsx jsx c h cpp hpp cs java go rs rb php sh ps1 bat json yaml yml toml ini cfg conf xml html htm css scss sql ipynb kt swift lua r"],
 ["other","Other",""]];
const EXT2CAT={};for(const[k,,ex]of CATS)for(const x of ex.split(" "))if(x)EXT2CAT[x]=k;
const TEXTEXT=new Set("txt md csv log json yaml yml toml ini cfg conf xml html htm css scss sql py js mjs ts tsx jsx c h cpp hpp cs java go rs rb php sh ps1 bat srt nfo kt swift lua r tex eml".split(" "));
const KINDS=[[0,"Deleted","Gone from the live filesystem"],[2,"Replaced","Path exists again, but it's a different file now"],[1,"Moved","Renamed/moved; content still exists at the new path"]];

/* ---------- state ---------- */
let D=null,S=[],DS=[],E=[];
let VIEWREC=null,ALIVE=null;  // the saved record being viewed (null = current scan); which of its snapshots still exist
const recQ=()=>VIEWREC?`&rec=${encodeURIComponent(VIEWREC.id)}`:"";
const recB=o=>VIEWREC?{...o,rec:VIEWREC.id}:o;
let marked=new Set(),sel=new Set(),expanded=new Set(),autoCollapsed=new Set();
let F={q:"",cats:new Set(),kinds:new Set([0]),minSize:0,t0:null,t1:null,lostOnly:false,goneOnly:false};
let view=LS.get("view","tree"),sortKey=LS.get("sort","size"),metric="size";
let FL=[],CB=[],ROWS=[],TREE=null,P=null,activeId=null,lastSnapClick=null,status=null;

/* Entries are kept lean so a million of them fit comfortably in memory:
   each folder path is stored once (DIRS) and paths/extensions are derived on demand. */
let DIRS=[],DIRLC=[];
class Entry{
  constructor(r,now){
    const[id,ds,di,name,size,mtime,ftype,ranges,kind,to]=r;
    this.id=id;this.ds=ds;this.di=di;this.name=name;this.size=size;this.mtime=mtime;this.ftype=ftype;
    this.ranges=ranges;this.kind=kind;this.to=to;
    this.first=ranges[0];this.last=ranges[ranges.length-1];
    let n=0;for(let i=0;i<ranges.length;i+=2)n+=ranges[i+1]-ranges[i]+1;this.nsnap=n;
    const next=this.last<DS[ds].last?S[this.last+1]:null;
    this.when=next?next.creation:now;this.pending=!next;
    this.cat=EXT2CAT[this.ext]||"other";
  }
  get dir(){return DIRS[this.di]}
  get path(){const d=DIRS[this.di];return d?d+"/"+this.name:this.name}
  get ext(){const n=this.name,i=n.lastIndexOf(".");return i>0?n.slice(i+1).toLowerCase():""}
}
function prep(res){
  D=res;S=res.snaps;DS=res.datasets;
  DIRS=res.dirs.map(d=>d[1]);DIRLC=res.dirs.map(([ds,d])=>(DS[ds].name+"/"+d).toLowerCase());
  E=res.entries.map(r=>new Entry(r,res.scannedAt));
  res.entries=null;res.dirs=null;  // let the raw JSON be garbage-collected
  marked=new Set((LS.get("marked:"+res.scannedAt,[])).filter(g=>g<S.length));
  sel=new Set();expanded=new Set(DS.map((_,i)=>"d"+i));autoCollapsed=new Set();
  computeP();
}
// P: prefix count of snapshots that will still exist (not ticked, and not already destroyed when viewing a record)
function computeP(){const n=S.length;P=new Int32Array(n+1);for(let i=0;i<n;i++)P[i+1]=P[i]+(marked.has(i)||(ALIVE&&!ALIVE[i])?0:1);LS.set("marked:"+(D&&D.scannedAt),[...marked])}
/* Viewing a record: mark files whose snapshots have all been destroyed since it was saved. */
function computeGone(){
  if(!ALIVE){for(const e of E)e.gone=false;return}
  const A=new Int32Array(S.length+1);for(let i=0;i<S.length;i++)A[i+1]=A[i]+ALIVE[i];
  for(const e of E){let g=true;const r=e.ranges;for(let i=0;i<r.length;i+=2)if(A[r[i+1]+1]-A[r[i]]>0){g=false;break}e.gone=g}
}
function isLost(e){if(!marked.size)return false;const r=e.ranges;for(let i=0;i<r.length;i+=2)if(P[r[i+1]+1]-P[r[i]]>0)return false;return true}

/* ---------- filtering ---------- */
/* Every search term must appear in the dataset/folder path or the file name.
   Folder matches are computed once per folder, not once per file. */
function makeMatcher(q){
  const terms=q.toLowerCase().split(/\s+/).filter(Boolean);
  if(!terms.length)return null;
  const dm=terms.map(t=>{const a=new Uint8Array(DIRLC.length);for(let i=0;i<DIRLC.length;i++)if(DIRLC[i].includes(t))a[i]=1;return a});
  return e=>{let nl=null;
    for(let k=0;k<terms.length;k++){
      if(dm[k][e.di])continue;
      const t=terms[k];if(nl===null)nl=e.name.toLowerCase();
      if(nl.includes(t))continue;
      if(t.includes("/")&&(DIRLC[e.di]+"/"+nl).includes(t))continue;
      return false}
    return true};
}
function applyFilters(){
  const match=makeMatcher(F.q);
  CB=[];FL=[];
  for(const e of E){
    if(!F.kinds.has(e.kind))continue;
    if(F.cats.size&&!F.cats.has(e.cat))continue;
    if(e.size<F.minSize)continue;
    if(match&&!match(e))continue;
    if(F.lostOnly&&!isLost(e))continue;
    if(F.goneOnly&&!e.gone)continue;
    CB.push(e);
    if(F.t0!=null&&(e.when<F.t0||e.when>=F.t1))continue;
    FL.push(e);
  }
}
function refresh(keepScroll){
  applyFilters();renderTiles();renderChart();renderChips();buildRows();
  if(!keepScroll)$("#list").scrollTop=0;
  renderList();renderSel();
}

/* ---------- tiles ---------- */
function renderTiles(){
  let size=0;for(const e of FL)size+=e.size;
  let lostN=0,lostB=0;if(marked.size)for(const e of E)if(e.kind===0&&isLost(e)){lostN++;lostB+=e.size}
  const ds=new Set(FL.map(e=>e.ds)).size;
  const oldest=S.length?Math.min(...S.map(s=>s.creation)):0;
  const filtered=FL.length!==E.filter(e=>e.kind===0).length||F.kinds.size!==1;
  $("#tiles").innerHTML=
   `<div class="tile"><div class="l">${filtered?"Matching files":"Deleted files"}</div><div class="v">${fmtNum(FL.length)}</div><div class="s">across ${ds} dataset${ds===1?"":"s"}</div></div>`+
   `<div class="tile"><div class="l">Total size</div><div class="v">${fmtBytes(size)}</div><div class="s">newest copy of each file</div></div>`+
   `<div class="tile"><div class="l">Snapshots scanned</div><div class="v">${fmtNum(S.length)}</div><div class="s">back to ${fmtDate(oldest)}</div></div>`+
   (marked.size?`<div class="tile ${lostN?"bad":""}"><div class="l">Lost if ticked are destroyed</div><div class="v">${fmtNum(lostN)} files</div><div class="s">${fmtBytes(lostB)} of deleted data</div></div>`
               :`<div class="tile"><div class="l">At risk</div><div class="v">–</div><div class="s">tick snapshots on the left to see</div></div>`);
}

/* ---------- chart ---------- */
function renderChart(){
  const svg=$("#chart"),W=svg.clientWidth||800,H=150,L=48,R=8,T=10,B=22;
  const list=CB;
  if(!list.length){svg.innerHTML=`<text x="${W/2}" y="${H/2}" text-anchor="middle" class="axis">Nothing to chart</text>`;$("#chartSub").textContent="";return}
  let tmin=Infinity,tmax=-Infinity;for(const e of list){if(e.when<tmin)tmin=e.when;if(e.when>tmax)tmax=e.when}
  const span=tmax-tmin,unit=span<=120*86400?"day":span<=800*86400?"week":"month";
  const bs=t=>{const d=new Date(t*1000);if(unit==="month")return new Date(d.getFullYear(),d.getMonth(),1)/1000;d.setHours(0,0,0,0);if(unit==="week")d.setDate(d.getDate()-((d.getDay()+6)%7));return d/1000};
  const nx=t=>{const d=new Date(t*1000);if(unit==="month")d.setMonth(d.getMonth()+1);else d.setDate(d.getDate()+(unit==="week"?7:1));return d/1000};
  const bk=[],idx=new Map();for(let t=bs(tmin);t<=tmax;t=nx(t)){idx.set(t,bk.length);bk.push({t0:t,t1:nx(t),b:0,n:0})}
  for(const e of list){const b=bk[idx.get(bs(e.when))];b.b+=e.size;b.n++}
  const val=b=>metric==="size"?b.b:b.n;
  const mx=Math.max(1,...bk.map(val));
  const pw=W-L-R,ph=H-T-B,bw=pw/bk.length,gap=bw>6?2:bw>3?1:0;
  const y=v=>T+ph-(v/mx)*ph;
  const fmtV=v=>metric==="size"?fmtBytes(v):fmtNum(Math.round(v));
  let s="";
  for(const f of [0.5,1]){const yy=y(mx*f);s+=`<line class="grid" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axis" x="${L-6}" y="${yy+3.5}" text-anchor="end">${fmtV(mx*f)}</text>`}
  s+=`<line class="grid" x1="${L}" x2="${W-R}" y1="${T+ph}" y2="${T+ph}"/>`;
  const sel=F.t0!=null;
  bk.forEach((b,i)=>{
    const w=Math.max(1,Math.min(bw-gap,36)),x=L+i*bw+(bw-w)/2,v=val(b);  // few buckets: keep bars bar-shaped
    s+=`<rect class="hit" data-i="${i}" x="${L+i*bw}" y="${T}" width="${bw}" height="${ph}"/>`;
    if(v>0){const h=Math.max(2,(v/mx)*ph),yy=T+ph-h,r=Math.min(4,w/2,h);
      s+=`<path class="bar${sel&&(b.t0<F.t0||b.t0>=F.t1)?" dim":""}" data-i="${i}" d="M${x},${T+ph}V${yy+r}Q${x},${yy} ${x+r},${yy}H${x+w-r}Q${x+w},${yy} ${x+w},${yy+r}V${T+ph}Z"/>`}
  });
  const fmtB=t=>unit==="month"?new Date(t*1000).toLocaleDateString(undefined,{month:"short",year:"2-digit"}):new Date(t*1000).toLocaleDateString(undefined,{month:"short",day:"numeric",year:span>300*86400?"2-digit":undefined});
  const nl=Math.max(2,Math.floor(pw/90)),step=Math.max(1,Math.ceil(bk.length/nl));
  for(let i=0;i<bk.length;i+=step)s+=`<text class="axis" x="${L+i*bw+bw/2}" y="${H-6}" text-anchor="middle">${fmtB(bk[i].t0)}</text>`;
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.innerHTML=s;
  $("#chartSub").textContent=`per ${unit} · dated by the first snapshot that no longer has the file`;
  svg.onmousemove=ev=>{const i=ev.target.dataset&&ev.target.dataset.i;const tip=$("#tip");if(i==null){tip.style.display="none";return}
    const b=bk[+i],pend=b.t1>D.scannedAt-1&&list.some(e=>e.pending&&e.when>=b.t0&&e.when<b.t1);
    tip.innerHTML=`<b>${unit==="month"?new Date(b.t0*1000).toLocaleDateString(undefined,{month:"long",year:"numeric"}):(unit==="week"?"Week of ":"")+fmtDate(b.t0)}</b><br>${fmtNum(b.n)} files · ${fmtBytes(b.b)}${pend?'<br><span class="muted">includes files deleted after the newest snapshot</span>':""}<br><span class="muted">click to filter · shift-click to extend</span>`;
    tip.style.display="block";const tx=Math.min(ev.clientX+14,innerWidth-330);tip.style.left=tx+"px";tip.style.top=(ev.clientY+14)+"px";
    $$("#chart .bar.hov").forEach(x=>x.classList.remove("hov"));const bar=svg.querySelector(`.bar[data-i="${i}"]`);if(bar)bar.classList.add("hov")};
  svg.onmouseleave=()=>{$("#tip").style.display="none"};
  svg.onclick=ev=>{const i=ev.target.dataset&&ev.target.dataset.i;if(i==null)return;const b=bk[+i];
    if(ev.shiftKey&&F.t0!=null){F.t0=Math.min(F.t0,b.t0);F.t1=Math.max(F.t1,b.t1)}
    else if(F.t0===b.t0&&F.t1===b.t1){F.t0=F.t1=null}else{F.t0=b.t0;F.t1=b.t1}
    refresh()};
  $("#timePill").innerHTML=F.t0!=null?`<span class="filterpill">${fmtDate(F.t0)} – ${fmtDate(F.t1-1)}<button title="Clear" id="clrTime">✕</button></span>`:"";
  const c=$("#clrTime");if(c)c.onclick=()=>{F.t0=F.t1=null;refresh()};
}
$("#metricSeg").onclick=e=>{const m=e.target.dataset.m;if(!m)return;metric=m;$$("#metricSeg button").forEach(b=>b.classList.toggle("on",b.dataset.m===m));renderChart()};
new ResizeObserver(()=>{if(D)renderChart()}).observe($("#chart"));

/* ---------- chips ---------- */
function renderChips(){
  const cc={};for(const[k]of CATS)cc[k]=0;
  const match=makeMatcher(F.q);
  for(const e of E){if(!F.kinds.has(e.kind)||e.size<F.minSize)continue;if(match&&!match(e))continue;cc[e.cat]++}
  const present=CATS.filter(([k])=>cc[k]||F.cats.has(k));
  const total=present.reduce((a,[k])=>a+cc[k],0);
  $("#catChips").innerHTML=`<span class="chip ${F.cats.size?"":"on"}" data-c="" title="Show every file type">All<span class="n">${fmtNum(total)}</span></span>`+
    present.map(([k,l])=>`<span class="chip ${F.cats.has(k)?"on":""}" data-c="${k}" title="Click to add or remove ${l.toLowerCase()} from the filter"><i style="background:var(--c-${k})"></i>${l}<span class="n">${fmtNum(cc[k])}</span></span>`).join("");
  const kc={0:0,1:0,2:0};for(const e of E)kc[e.kind]++;
  $("#kindChips").innerHTML=KINDS.filter(([k])=>kc[k]||k===0).map(([k,l,t])=>`<span class="chip ${F.kinds.has(k)?"on":""}" data-k="${k}" title="${esc(t)}">${l}<span class="n">${fmtNum(kc[k])}</span></span>`).join("")
    +(marked.size?`<span class="chip warnchip ${F.lostOnly?"on":""}" data-lost="1" title="Only files that exist solely in ticked snapshots">Would be lost</span>`:"")
    +(ALIVE?`<span class="chip warnchip ${F.goneOnly?"on":""}" data-gone="1" title="Files whose snapshots have all been destroyed since this record was saved">Gone for good<span class="n">${fmtNum(E.reduce((a,e)=>a+(e.gone&&F.kinds.has(e.kind)?1:0),0))}</span></span>`:"");
}
// Type chips toggle independently (Images + Audio = both); "All" clears the type filter.
// Alt-click a type to show only that type.
$("#catChips").onclick=e=>{const c=e.target.closest(".chip");if(!c)return;const k=c.dataset.c;
  if(!k)F.cats=new Set();
  else if(e.altKey)F.cats=new Set([k]);
  else{F.cats.has(k)?F.cats.delete(k):F.cats.add(k);
    if(F.cats.size&&CATS.every(([c])=>F.cats.has(c)||!E.some(x=>x.cat===c)))F.cats=new Set()}  // every type on = All
  refresh()};
$("#kindChips").onclick=e=>{const c=e.target.closest(".chip");if(!c)return;
  if(c.dataset.lost){F.lostOnly=!F.lostOnly;return refresh()}
  if(c.dataset.gone){F.goneOnly=!F.goneOnly;return refresh()}
  const k=+c.dataset.k;F.kinds.has(k)?F.kinds.delete(k):F.kinds.add(k);if(!F.kinds.size)F.kinds.add(0);refresh()};
$("#minSize").onchange=e=>{F.minSize=+e.target.value;refresh()};
let qT;$("#q").oninput=e=>{clearTimeout(qT);qT=setTimeout(()=>{F.q=e.target.value.trim();autoCollapsed.clear();refresh()},160)};

/* ---------- tree / rows ---------- */
function buildTree(list){
  const root={kids:new Map(),files:[],key:"",name:""},byDir=new Array(DIRS.length);
  for(const e of list){
    let n=byDir[e.di];
    if(!n){
      n=root.kids.get(e.ds);
      if(!n){n={kids:new Map(),files:[],key:"d"+e.ds,name:DS[e.ds].name,isDs:true,ds:e.ds,path:""};root.kids.set(e.ds,n)}
      if(e.dir)for(const seg of e.dir.split("/")){let c=n.kids.get(seg);if(!c){c={kids:new Map(),files:[],key:n.key+"/"+seg,name:seg,ds:e.ds,path:n.path?n.path+"/"+seg:seg};n.kids.set(seg,c)}n=c}
      byDir[e.di]=n;
    }
    n.files.push(e);
  }
  const agg=n=>{let s=0,c=0,w=0,w0=Infinity;for(const k of n.kids.values()){agg(k);s+=k.size;c+=k.count;w=Math.max(w,k.when);w0=Math.min(w0,k.when0)}
    for(const f of n.files){s+=f.size;c++;w=Math.max(w,f.when);w0=Math.min(w0,f.when)}n.size=s;n.count=c;n.when=w;n.when0=w0};
  agg(root);recount(root);return root;
}
function recount(n){let s=0,l=0;for(const k of n.kids.values()){recount(k);s+=k.sel;l+=k.lost}for(const f of n.files){if(sel.has(f.id))s++;if(f.kind===0&&isLost(f))l++}n.sel=s;n.lost=l}
const autoAll=()=>(F.q||F.t0!=null||F.lostOnly||F.cats.size)&&FL.length<=2500;
const isOpen=k=>autoAll()?!autoCollapsed.has(k):expanded.has(k);
function sorter(){
  if(sortKey==="name")return[(a,b)=>a.name.localeCompare(b.name),(a,b)=>a.name.localeCompare(b.name)];
  if(sortKey==="when")return[(a,b)=>b.when-a.when,(a,b)=>b.when-a.when||b.size-a.size];
  return[(a,b)=>b.size-a.size,(a,b)=>b.size-a.size];
}
function buildRows(){
  ROWS=[];const[cd,cf]=sorter();
  if(view==="list"){TREE=null;ROWS=FL.slice().sort(cf).map(e=>({e,depth:0,flat:true}));return}
  TREE=buildTree(FL);
  const walk=(n,depth)=>{
    for(const k of [...n.kids.values()].sort(cd)){
      let node=k,label=k.name;
      if(!k.isDs)while(node.kids.size===1&&node.files.length===0){node=node.kids.values().next().value;label+="/"+node.name}
      ROWS.push({dir:node,label,depth,parent:n});
      if(isOpen(node.key))walk(node,depth+1);
    }
    for(const f of n.files.slice().sort(cf))ROWS.push({e:f,depth});
  };
  walk(TREE,0);
}
const RH=34;
function presHTML(e){
  const d=DS[e.ds],t0=S[d.first].creation,t1=D.scannedAt,sp=Math.max(1,t1-t0);
  const pos=t=>Math.max(0,Math.min(100,(t-t0)/sp*100));
  let h="";const r=e.ranges;
  for(let i=0;i<r.length;i+=2){const a=pos(S[r[i]].creation),b=pos(S[r[i+1]].creation);h+=`<b style="left:${a}%;width:${Math.max(b-a,0)}%"></b>`}
  const lb=pos(S[e.last].creation),w=pos(e.when);if(w>lb)h+=`<b class="gone" style="left:${lb}%;width:${w-lb}%"></b>`;
  return h;
}
function rowHTML(r,i){
  if(r.dir){
    const n=r.dir,open=isOpen(n.key),ps=r.parent&&r.parent.size?n.size/r.parent.size*100:100;
    const ck=n.sel===0?"":n.sel===n.count?"checked":"mixed";
    const when=n.when0===n.when?fmtDate(n.when):`${fmtDate(n.when0).replace(/,? \d{4}$/,"")} – ${fmtDate(n.when)}`;
    return `<div class="row dir${n.isDs?" ds":""}" data-i="${i}" style="--d:${r.depth}"><span class="ck ${ck}" data-a="ck"></span>`+
      `<span class="name"><span class="chev${open?" open":""}">▶</span>`+
      (n.isDs?`<svg class="fold" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>`
             :`<svg class="fold" viewBox="0 0 24 20" fill="currentColor" opacity=".55"><path d="M2 3a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2z"/></svg>`)+
      `<span class="nm" title="${esc(n.isDs?DS[n.ds].mountpoint:n.path)}">${esc(r.label)}</span><span class="cnt">${fmtNum(n.count)} file${n.count===1?"":"s"}</span>${n.lost?`<span class="tag lost">${fmtNum(n.lost)} would be lost</span>`:""}</span>`+
      `<span class="sz">${fmtBytes(n.size)}<i class="bar"><b style="width:${ps}%"></b></i></span><span class="when">${when}</span><span></span></div>`;
  }
  const e=r.e,lost=e.kind===0&&isLost(e);
  const tag=(e.kind===1?`<span class="tag moved" title="now at ${esc(e.to||"")}">moved</span>`:e.kind===2?`<span class="tag repl">replaced</span>`:"")
    +(e.gone?`<span class="tag gone" title="Every snapshot holding this file has been destroyed">gone for good</span>`:"");
  return `<div class="row file${sel.has(e.id)?" sel":""}${activeId===e.id?" active":""}${lost?" lostrow":""}" data-i="${i}" style="--d:${r.depth}"><span class="ck${sel.has(e.id)?" checked":""}" data-a="ck"></span>`+
    `<span class="name">${r.flat?"":`<span class="chev"></span>`}<span class="ext"><i style="background:var(--c-${e.cat})"></i>${esc(e.ext.slice(0,5)||"file")}</span><span class="nm" title="${esc(e.path)}">${esc(e.name)}</span>`+
    (r.flat?`<span class="dirp">${esc(DS[e.ds].name+"/"+e.dir)}</span>`:"")+tag+`</span>`+
    `<span class="sz">${fmtBytes(e.size)}</span><span class="when" title="${e.pending?"Deleted after the newest snapshot":"Last seen "+fmtDT(S[e.last].creation)}">${e.pending?"after last snap":fmtDate(e.when)}</span>`+
    `<span class="pres" title="in ${e.nsnap} snapshot${e.nsnap===1?"":"s"}: ${esc(S[e.first].name)} → ${esc(S[e.last].name)}">${presHTML(e)}</span></div>`;
}
function renderList(){
  const list=$("#list"),sp=$("#spacer"),rows=$("#rows");
  if(!ROWS.length){sp.style.height="auto";rows.style.top="0";
    rows.innerHTML=E.length?`<div class="empty"><h3>Nothing matches</h3>Try clearing filters or the search box.</div>`:`<div class="empty"><h3>No deleted files found</h3>Every file in your snapshots still exists in the live filesystem.</div>`;return}
  sp.style.height=ROWS.length*RH+"px";
  const st=list.scrollTop,vh=list.clientHeight||600;
  const a=Math.max(0,Math.floor(st/RH)-10),b=Math.min(ROWS.length,Math.ceil((st+vh)/RH)+10);
  let h="";for(let i=a;i<b;i++)h+=rowHTML(ROWS[i],i);
  rows.style.top=a*RH+"px";rows.innerHTML=h;
  $$(".lhead [data-sort]").forEach(x=>x.classList.toggle("on",x.dataset.sort===sortKey));
}
let rafL=0;$("#list").addEventListener("scroll",()=>{if(!rafL)rafL=requestAnimationFrame(()=>{rafL=0;renderList()})});
new ResizeObserver(()=>{if(D)renderList()}).observe($("#list"));

function filesUnder(n,out=[]){for(const k of n.kids.values())filesUnder(k,out);for(const f of n.files)out.push(f.id);return out}
$("#rows").onclick=ev=>{
  const rowEl=ev.target.closest(".row");if(!rowEl)return;const r=ROWS[+rowEl.dataset.i];
  if(ev.target.dataset.a==="ck"){
    if(r.dir){const ids=filesUnder(r.dir),all=r.dir.sel===r.dir.count;for(const id of ids)all?sel.delete(id):sel.add(id)}
    else sel.has(r.e.id)?sel.delete(r.e.id):sel.add(r.e.id);
    if(TREE)recount(TREE);renderList();renderSel();return}
  if(r.dir){toggleDir(r.dir.key);return}
  openDrawer(r.e);
};
function toggleDir(k){if(autoAll()){autoCollapsed.has(k)?autoCollapsed.delete(k):autoCollapsed.add(k)}else{expanded.has(k)?expanded.delete(k):expanded.add(k)}buildRows();renderList()}
$("#viewSeg").onclick=e=>{const v=e.target.dataset.v;if(!v)return;view=v;LS.set("view",v);syncView();refresh()};
function syncView(){$$("#viewSeg button").forEach(b=>b.classList.toggle("on",b.dataset.v===view))}
$(".lhead").onclick=e=>{const k=e.target.dataset.sort;if(!k)return;sortKey=k;LS.set("sort",k);buildRows();renderList()};

/* keyboard */
document.addEventListener("keydown",ev=>{
  if(tab!=="files")return;
  const inField=/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
  if(ev.key==="/"&&!inField){ev.preventDefault();$("#q").focus();return}
  if(ev.key==="Escape"){if(inField)document.activeElement.blur();else closeDrawer();return}
  if(inField||document.querySelector("dialog[open]"))return;
  const cur=ROWS.findIndex(r=>r.e&&r.e.id===activeId);
  if(ev.key==="ArrowDown"||ev.key==="j"||ev.key==="ArrowUp"||ev.key==="k"){
    ev.preventDefault();const d=(ev.key==="ArrowDown"||ev.key==="j")?1:-1;
    for(let i=cur+d;i>=0&&i<ROWS.length;i+=d)if(ROWS[i].e){openDrawer(ROWS[i].e);ensureVisible(i);break}
  }else if(ev.key===" "&&cur>=0){ev.preventDefault();const id=ROWS[cur].e.id;sel.has(id)?sel.delete(id):sel.add(id);if(TREE)recount(TREE);renderList();renderSel()}
});
function ensureVisible(i){const l=$("#list"),y=i*RH;if(y<l.scrollTop)l.scrollTop=y;else if(y+RH>l.scrollTop+l.clientHeight)l.scrollTop=y+RH-l.clientHeight}

/* ---------- selection ---------- */
function renderSel(){
  let b=0;for(const e of E)if(sel.has(e.id))b+=e.size;
  $("#selbar").classList.toggle("on",sel.size>0);
  $("#selText").innerHTML=`<b>${fmtNum(sel.size)}</b> file${sel.size===1?"":"s"} selected · ${fmtBytes(b)}`;
}
$("#selClear").onclick=()=>{sel.clear();if(TREE)recount(TREE);renderList();renderSel()};
$("#selRestore").onclick=()=>openScript("restore");
$("#selRecover").onclick=()=>openRecover([...sel].map(id=>[id]));

/* ---------- drawer ---------- */
function closeDrawer(){activeId=null;$("#drawer").classList.remove("on");renderList()}
function fileURL(e,g,extra=""){return`/api/file?id=${e.id}${g!=null?"&snap="+g:""}${extra}${recQ()}`}
function previewHTML(e,g){
  const u=fileURL(e,g);
  if(e.ftype==="@")return`<pre class="mono" data-src="${u}&text=1">loading…</pre>`;
  if(e.cat==="image"&&!["raw","cr2","cr3","nef","arw","dng","orf","rw2","raf","psd","heic","heif","tif","tiff"].includes(e.ext))return`<img src="${u}" alt="">`;
  if(e.cat==="video"&&["mp4","m4v","webm","mov","mkv"].includes(e.ext))return`<video src="${u}" controls preload="metadata"></video>`;
  if(e.cat==="audio"&&["mp3","m4a","aac","ogg","opus","wav","flac"].includes(e.ext))return`<audio src="${u}" controls preload="metadata"></audio>`;
  if(e.ext==="pdf")return`<iframe src="${u}"></iframe>`;
  if(TEXTEXT.has(e.ext)||(!e.ext&&e.size<200000))return`<pre class="mono" data-src="${u}&text=1">loading…</pre>`;
  return`<div class="np">No preview for .${esc(e.ext||"?")} files<br><small>${fmtBytes(e.size)}</small></div>`;
}
function loadPreviewText(root){const pre=root.querySelector("pre[data-src]");if(!pre)return;
  api(pre.dataset.src).then(r=>r.text()).then(t=>{pre.textContent=t+(t.length>=262000?"\n\n… (truncated)":"")}).catch(err=>{pre.textContent="Could not load: "+err.message})}
async function openDrawer(e){
  activeId=e.id;const d=DS[e.ds];renderList();
  const live=(d.mountpoint+"/"+e.path);
  const strip=[];for(let g=d.first;g<=d.last;g++){let inn=false;for(let i=0;i<e.ranges.length;i+=2)if(g>=e.ranges[i]&&g<=e.ranges[i+1]){inn=true;break}
    strip.push(`<i class="${inn?"in":""}${marked.has(g)?" mk":""}${ALIVE&&!ALIVE[g]?" gn":""}" title="${esc(S[g].name)} · ${fmtDT(S[g].creation)}${ALIVE&&!ALIVE[g]?" (destroyed)":""}"></i>`)}
  const lost=e.kind===0&&isLost(e);
  const status=e.kind===1?`Moved → <span class="mono">${esc(e.to)}</span>`:e.kind===2?`Replaced — something else lives at this path now`:lost?`<b style="color:var(--danger)">Deleted — would be lost</b> if the ticked snapshots are destroyed`:"Deleted";
  $("#drawerInner").innerHTML=`
   <div class="dh"><h3>${esc(e.name)}</h3><button class="ghost" id="dClose" title="Close (Esc)">✕</button></div>
   <div class="dbody">
     <div class="preview" id="dPrev">${e.gone?"":previewHTML(e)}</div>
     <div class="dacts">
       <button class="primary" id="dRecover">Recover&hellip;</button>
       <a class="btn" style="padding:5px 12px" href="${fileURL(e,null,"&dl=1")}">Download</a>
       <button id="dCopyRestore">Copy restore command</button>
       <button id="dCopySnap">Copy snapshot path</button>
       <button id="dPick">${sel.has(e.id)?"Unselect":"Select"}</button>
     </div>
     <dl class="kv">
       <dt>Status</dt><dd>${status}</dd>
       <dt>Was at</dt><dd class="mono">${esc(live)}</dd>
       <dt>Dataset</dt><dd>${esc(d.name)}</dd>
       <dt>Size</dt><dd>${fmtBytes(e.size)} <span class="muted">(${fmtNum(e.size)} bytes)</span></dd>
       <dt>Modified</dt><dd>${fmtDT(e.mtime)}</dd>
       <dt>Disappeared</dt><dd>${e.pending?`after ${fmtDT(S[e.last].creation)} <span class="muted">(newest snapshot)</span>`:`between ${fmtDT(S[e.last].creation)} and ${fmtDT(e.when)}`}</dd>
       <dt>Kept by</dt><dd>${e.nsnap} of ${d.last-d.first+1} snapshots <span class="muted">(${esc(S[e.first].name)} → ${esc(S[e.last].name)})</span></dd>
     </dl>
     <div><h4>Snapshot timeline</h4><div class="strip">${strip.join("")}</div></div>
     <div><h4>Versions</h4><div class="vers" id="dVers"><span class="muted">checking each snapshot…</span></div></div>
   </div>`;
  $("#drawer").classList.add("on");
  if(e.gone){  // only the record of it remains: no file to preview, download or recover
    $("#dPrev").innerHTML=`<div class="np"><b>Gone for good</b><br>Every snapshot that held this file has been destroyed.<br>This record is all that's left of it.</div>`;
    ["#dRecover","#dCopyRestore","#dCopySnap"].forEach(s=>{const b=$(s);if(b)b.hidden=true});
    $$("#drawerInner .dacts a").forEach(a=>a.hidden=true);
    $("#dVers").innerHTML=`<span class="muted">No snapshots left to read versions from.</span>`;
    $("#dClose").onclick=closeDrawer;
    $("#dPick").onclick=ev=>{sel.has(e.id)?sel.delete(e.id):sel.add(e.id);ev.target.textContent=sel.has(e.id)?"Unselect":"Select";if(TREE)recount(TREE);renderList();renderSel()};
    return;
  }
  loadPreviewText($("#dPrev"));
  $("#dClose").onclick=closeDrawer;
  $("#dCopySnap").onclick=()=>copy(snapPath(e,e.last));
  $("#dRecover").onclick=()=>openRecover([[e.id]]);
  $("#dCopyRestore").onclick=async()=>{const r=await post("/api/script",recB({kind:"restore",ids:[e.id]}));const t=(await r.text()).split("\n").filter(l=>l&&!l.startsWith("#")&&!l.startsWith("set "));copy(t.join("\n"))};
  $("#dPick").onclick=ev=>{sel.has(e.id)?sel.delete(e.id):sel.add(e.id);ev.target.textContent=sel.has(e.id)?"Unselect":"Select";if(TREE)recount(TREE);renderList();renderSel()};
  try{
    const vs=await(await api(`/api/versions?id=${e.id}${recQ()}`)).json();
    if(activeId!==e.id)return;
    $("#dVers").innerHTML=vs.length?vs.map((v,i)=>`<div class="ver${i===vs.length-1?" cur":""}"><div class="vi"><div><b>${fmtBytes(v.size)}</b> · modified ${fmtDT(v.mtime)}</div><div class="muted">${v.count} snapshot${v.count===1?"":"s"}: ${esc(S[v.from].name)}${v.count>1?" → "+esc(S[v.to].name):""}</div></div>`+
      `<button data-g="${v.to}">Preview</button><button data-rg="${v.to}">Recover</button><a class="btn" style="padding:3px 8px" href="${fileURL(e,v.to,"&dl=1")}">Download</a></div>`).join("")
      +(vs.length>1?`<div class="muted" style="font-size:12px">${vs.length} different versions across snapshots. Highlighted = newest.</div>`:""):`<span class="muted">none readable</span>`;
    $("#dVers").onclick=ev=>{const rg=ev.target.dataset.rg;if(rg!=null)return openRecover([[e.id,+rg]]);
      const g=ev.target.dataset.g;if(g==null)return;$("#dPrev").innerHTML=previewHTML(e,+g);loadPreviewText($("#dPrev"))};
  }catch(err){$("#dVers").innerHTML=`<span class="muted">${esc(err.message)}</span>`}
}
function snapPath(e,g){const d=DS[e.ds];return`${d.mountpoint}/.zfs/snapshot/${S[g].name}/${e.path}`}

/* ---------- snapshots panel ---------- */
let HOLD=null;
function computeHolds(){
  const n=S.length,c=new Float64Array(n+1),b=new Float64Array(n+1),only=new Int32Array(n);
  for(const e of E){if(e.kind!==0)continue;const r=e.ranges;for(let i=0;i<r.length;i+=2){c[r[i]]++;c[r[i+1]+1]--;b[r[i]]+=e.size;b[r[i+1]+1]-=e.size}if(r.length===2&&r[0]===r[1])only[r[0]]++}
  for(let i=1;i<=n;i++){c[i]+=c[i-1];b[i]+=b[i-1]}
  HOLD={c,b,only};
}
function renderSnaps(){
  computeHolds();
  const open=LS.get("openDs",null);
  let h="";
  DS.forEach((d,di)=>{
    let mx=1;for(let g=d.first;g<=d.last;g++)mx=Math.max(mx,HOLD.b[g]);
    const cnt=d.last-d.first+1,isOpen=open?open.includes(d.name):(S.length<400||di===0);
    h+=`<details class="dsg" data-ds="${di}"${isOpen?" open":""}><summary><span class="chev">▶</span>${esc(d.name)}<span class="n">${cnt}</span><button class="ghost" data-all="${di}" title="Tick / untick all in this dataset">all</button></summary>`;
    for(let g=d.last;g>=d.first;g--){const s=S[g],hb=HOLD.b[g],hc=HOLD.c[g];
      const gone=ALIVE&&!ALIVE[g];
      h+=`<label class="snap${marked.has(g)?" m":""}${gone?" gone":""}" data-g="${g}" title="${esc(d.name+"@"+s.name)}${gone?" (destroyed since this record was saved)":""}\nreferenced ${fmtBytes(s.referenced)} · used ${fmtBytes(s.used)}"><input type="checkbox"${marked.has(g)?" checked":""}${gone?" disabled":""}><span class="sn">${esc(s.name)}</span><span class="sd">${gone?"destroyed":fmtDate(s.creation)}</span>`+
        `<span class="sh"><span>${fmtNum(hc)} deleted · ${fmtBytes(hb)}${HOLD.only[g]?` · <span class="only" title="files found in no other snapshot">${fmtNum(HOLD.only[g])} only here</span>`:""}</span><span class="hb"><b style="width:${hb/mx*100}%"></b></span><span class="muted" title="space unique to this snapshot (zfs used)">${fmtBytes(s.used)}</span></span></label>`}
    h+=`</details>`;
  });
  $("#snaps").innerHTML=h;
  renderSum();
}
$("#snaps").addEventListener("toggle",()=>LS.set("openDs",$$(".dsg[open]").map(x=>DS[+x.dataset.ds].name)),true);
$("#snaps").onclick=ev=>{
  const all=ev.target.dataset.all;
  if(all!=null){ev.preventDefault();const d=DS[+all];let every=true;for(let g=d.first;g<=d.last;g++)if(!marked.has(g))every=false;
    for(let g=d.first;g<=d.last;g++)every?marked.delete(g):marked.add(g);return marksChanged()}
  const l=ev.target.closest(".snap");if(!l)return;ev.preventDefault();
  if(l.classList.contains("gone"))return;  // already destroyed
  const g=+l.dataset.g,on=!marked.has(g);
  if(ev.shiftKey&&lastSnapClick!=null&&S[lastSnapClick].ds===S[g].ds){const[a,b]=[Math.min(g,lastSnapClick),Math.max(g,lastSnapClick)];for(let i=a;i<=b;i++)on?marked.add(i):marked.delete(i)}
  else on?marked.add(g):marked.delete(g);
  lastSnapClick=g;marksChanged();
};
function marksChanged(){
  computeP();
  $$(".snap").forEach(l=>{const m=marked.has(+l.dataset.g);l.classList.toggle("m",m);l.querySelector("input").checked=m});
  if(!marked.size)F.lostOnly=false;
  $("#estOut")&&($("#estOut").textContent="");
  renderSum();
  if(F.lostOnly)refresh(true);else{renderTiles();renderChips();if(TREE)recount(TREE);renderList();if(activeId!=null){const e=E[activeId];if(e)openDrawer(e)}}
}
function renderSum(){
  if(!marked.size){$("#sumcard").innerHTML=`<div class="lostline">No snapshots ticked. Tick some to see what destroying them would cost you.</div>`;return}
  let n=0,b=0;for(const e of E)if(e.kind===0&&isLost(e)){n++;b+=e.size}
  let used=0;for(const g of marked)used+=S[g].used;
  $("#sumcard").innerHTML=`<div class="big"><b>${fmtNum(marked.size)}</b> snapshot${marked.size===1?"":"s"} ticked</div>`+
    (n?`<div class="lostline bad"><b>${fmtNum(n)}</b> deleted file${n===1?"":"s"} (${fmtBytes(b)}) exist only in these and would be gone for good.</div>`
      :`<div class="lostline good">✔ Every deleted file in these snapshots is also kept by another snapshot you're keeping.</div>`)+
    `<div class="btns">${n?`<button id="showLost" class="danger">Show them</button>`:""}`+
    (VIEWREC?`</div><div class="muted small">Viewing a saved record: space estimates and destroy scripts work on the current scan.</div>`
      :`<button id="estBtn">Estimate space freed</button><button id="destroyBtn">Destroy script…</button></div><div id="estOut"></div>`);
  const sl=$("#showLost");if(sl)sl.onclick=()=>{F.lostOnly=true;F.kinds=new Set([0]);F.t0=F.t1=null;refresh()};
  if(VIEWREC)return;
  $("#estBtn").onclick=async()=>{const o=$("#estOut");o.textContent="Asking zfs (dry run)…";
    try{const r=await(await post("/api/estimate",{snaps:[...marked]})).json();
      o.innerHTML=`<b>≈ ${fmtBytes(r.total)}</b> would be freed`+(r.datasets.length>1?"<br>"+r.datasets.map(x=>`${esc(x.dataset)}: ${x.error?`<span style="color:var(--danger)">${esc(x.error)}</span>`:fmtBytes(x.bytes)}`).join("<br>"):"")+
        (r.datasets.some(x=>x.error)&&r.datasets.length===1?`<br><span style="color:var(--danger)">${esc(r.datasets[0].error)}</span>`:"")+`<br><span class="muted">from <code>zfs destroy -nv</code>; nothing was destroyed</span>`}
    catch(err){o.textContent=err.message}};
  $("#destroyBtn").onclick=()=>openScript("destroy");
}
$("#markOlder").onclick=()=>{const v=$("#olderThan").value;if(!v)return toast("Pick a date first");const t=new Date(v+"T00:00:00")/1000;let n=0;S.forEach((s,g)=>{if(s.creation<t&&!marked.has(g)){marked.add(g);n++}});toast(`Ticked ${n} snapshot${n===1?"":"s"}`);marksChanged()};
$("#clearMarks").onclick=()=>{marked.clear();marksChanged()};

/* ---------- script dialog ---------- */
let scKind="restore";
async function openScript(kind){
  scKind=kind;const dlg=$("#scriptDlg");
  $("#scOpts").style.display=kind==="restore"?"flex":"none";
  $("#scTitle").textContent=kind==="restore"?`Restore ${fmtNum(sel.size)} file${sel.size===1?"":"s"}`:`Destroy ${fmtNum(marked.size)} snapshot${marked.size===1?"":"s"}`;
  $("#scDesc").innerHTML=kind==="restore"?"Run this on the NAS as root. It copies files out of the newest snapshot that has them, keeping timestamps and never overwriting anything."
    :"Each command runs as a <b>dry run</b> first. The real destroy lines are commented out — review the output, then uncomment them. This cannot be undone.";
  $("#scNote").textContent=kind==="destroy"&&HOLD?(()=>{let n=0;for(const e of E)if(e.kind===0&&isLost(e))n++;return n?`⚠ ${fmtNum(n)} deleted files would be lost`:"No unique deleted files would be lost"})():"";
  await genScript();if(!dlg.open)dlg.showModal();
}
async function genScript(){
  const body=scKind==="restore"?recB({kind:"restore",ids:[...sel],dest:document.querySelector('input[name=dest]:checked').value==="folder"?$("#scDest").value:""}):{kind:"destroy",snaps:[...marked]};
  $("#scText").value="generating…";
  try{$("#scText").value=await(await post("/api/script",body)).text()}catch(err){$("#scText").value="# "+err.message}
}
$$("input[name=dest]").forEach(r=>r.onchange=genScript);
let dT;$("#scDest").oninput=()=>{document.querySelector('input[name=dest][value=folder]').checked=true;clearTimeout(dT);dT=setTimeout(genScript,300)};
$("#scCopy").onclick=()=>copy($("#scText").value);
$("#scClose").onclick=()=>$("#scriptDlg").close();
$("#scDl").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([$("#scText").value],{type:"text/x-sh"}));a.download=scKind==="restore"?"snapsift-restore.sh":"snapsift-destroy.sh";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)};

/* ---------- recovery ---------- */
let recItems=[],recCheckT=null,recPollT=null,recPrev=null,recDismissed=null;
const recLayout=()=>document.querySelector('input[name=recLayout]:checked').value;
function openRecover(items){
  if(!items.length)return toast("Nothing selected");
  recItems=items;
  const bytes=items.reduce((a,[id])=>a+(E[id]?E[id].size:0),0);
  $("#recTitle").textContent=`Recover ${fmtNum(items.length)} file${items.length===1?"":"s"} (${fmtBytes(bytes)})`;
  $("#recDest").value=LS.get("recDest","");  // no guessed default: where to write is a deliberate choice
  setTimeout(()=>$("#recDest").focus(),50);
  const lay=LS.get("recLayout","dataset");document.querySelector(`input[name=recLayout][value=${lay}]`).checked=true;
  $("#recGo").disabled=true;$("#recDlg").showModal();recCheck();
}
async function recCheck(){
  clearTimeout(recCheckT);
  $("#recInfo").innerHTML=`<span class="muted">Checking destination…</span>`;$("#recGo").disabled=true;
  let r;
  try{r=await(await post("/api/recover/check",recB({items:recItems,dest:$("#recDest").value,layout:recLayout()}))).json()}
  catch(err){$("#recInfo").innerHTML=`<span class="err">${esc(err.message)}</span>`;return}
  const lines=[];
  if(r.error)lines.push(`<span class="err">⚠ ${esc(r.error)}</span>`);
  if(r.files!=null){
    lines.push(`<span><b>${fmtNum(r.files-r.conflicts)}</b> file${r.files-r.conflicts===1?"":"s"} to copy · needs <b>${fmtBytes(r.needed)}</b>${r.free!=null?` · ${fmtBytes(r.free)} free there`:""}</span>`);
    if(r.conflicts)lines.push(`<span class="warn">${fmtNum(r.conflicts)} already exist${r.conflicts===1?"s":""} at the destination and will be skipped (never overwritten).</span>`);
    if(!r.destExists)lines.push(`<span class="muted">The folder doesn't exist yet; it will be created.</span>`);
    const short=to=>r.dest&&to.startsWith(r.dest)?`<span class="muted">&lt;folder&gt;</span>${esc(to.slice(r.dest.length))}`:esc(to);
    if(r.examples&&r.examples.length)lines.push(`<div class="ex">${r.examples.map(([from,to])=>`<div title="${esc(from)} → ${esc(to)}"><span class="mono">${short(to)}</span></div>`).join("")}${r.files>r.examples.length?`<div class="muted">…and ${fmtNum(r.files-r.examples.length)} more</div>`:""}</div>`);
  }
  $("#recInfo").innerHTML=lines.join("");
  $("#recGo").disabled=!r.ok;
}
$("#recDest").oninput=()=>{clearTimeout(recCheckT);$("#recGo").disabled=true;recCheckT=setTimeout(recCheck,500)};
$$("input[name=recLayout]").forEach(x=>x.onchange=recCheck);
$("#recCancelDlg").onclick=()=>$("#recDlg").close();
$("#recGo").onclick=async()=>{
  $("#recGo").disabled=true;
  try{await post("/api/recover/start",recB({items:recItems,dest:$("#recDest").value,layout:recLayout()}))}
  catch(err){$("#recInfo").insertAdjacentHTML("afterbegin",`<span class="err">⚠ ${esc(err.message)}</span>`);return}
  LS.set("recDest",$("#recDest").value);LS.set("recLayout",recLayout());
  $("#recDlg").close();recDismissed=null;recPrev=null;recPoll();
};
async function recPoll(){
  clearTimeout(recPollT);
  let s;try{s=await(await api("/api/recover/status")).json()}catch(err){recPollT=setTimeout(recPoll,2000);return}
  renderRec(s);
  if(s.state==="running")recPollT=setTimeout(recPoll,1000);
}
function renderRec(s){
  const p=$("#recPanel");
  if(s.state==="idle"||recDismissed===s.started){p.hidden=true;return}
  p.hidden=false;
  const run=s.state==="running",pct=s.bytesTotal?s.bytes/s.bytesTotal*100:(s.total?s.done/s.total*100:100);
  let rate=0;if(recPrev&&s.serverTime>recPrev.t)rate=(s.bytes-recPrev.b)/(s.serverTime-recPrev.t);recPrev={b:s.bytes,t:s.serverTime};
  $("#recDot").className="dot "+(run?"ok":s.failed?"bad":s.state==="cancelled"?"quiet":"ok");
  $("#recDot").style.animation=run?"":"none";
  $("#recHead").textContent=run?"Recovering…":s.state==="cancelled"?"Recovery cancelled":"Recovery finished";
  $("#recBar").style.width=pct+"%";
  $("#recText").innerHTML=run
    ?`${fmtNum(s.done)} / ${fmtNum(s.total)} files · ${fmtBytes(s.bytes)} / ${fmtBytes(s.bytesTotal)}${rate>0?` · ${fmtBytes(rate)}/s`:""}<br><span class="mono muted">${esc(s.current||"")}</span>`
    :`<b>${fmtNum(s.copied)}</b> copied · ${fmtNum(s.skipped)} skipped (already existed) · ${s.failed?`<span style="color:var(--danger)">${fmtNum(s.failed)} failed</span>`:"0 failed"}<br>into <span class="mono">${esc(s.dest)}</span><br><span class="muted">List of everything: <span class="mono">${esc(s.manifest||"")}</span></span>`;
  $("#recErr").innerHTML=s.errors&&s.errors.length?`<details><summary>${s.errors.length} error${s.errors.length===1?"":"s"}</summary><div class="mono">${esc(s.errors.join("\n"))}</div></details>`:"";
  $("#recStop").hidden=!run;$("#recClose").hidden=run;
  if(!run&&!renderRec.notified?.[s.started]){(renderRec.notified??={})[s.started]=1;if(s.state==="done")toast(`Recovered ${fmtNum(s.copied)} files`)}
}
$("#recStop").onclick=()=>post("/api/recover/cancel",{});
$("#recClose").onclick=async()=>{const s=await(await api("/api/recover/status")).json();recDismissed=s.started;$("#recPanel").hidden=true};

/* ---------- tabs ---------- */
let tab=null,dsLoaded=false;
function showTab(name,save=true){
  if(!["files","scan","records","console"].includes(name))name="files";
  tab=name;
  $$(".view").forEach(v=>v.classList.toggle("on",v.id==="view-"+name));
  $$("#tabs button").forEach(b=>{const on=b.dataset.tab===name;b.classList.toggle("on",on);b.setAttribute("aria-selected",on)});
  $("#searchBox").style.visibility=name==="files"?"":"hidden";
  if(save){LS.set("tab",name);try{history.replaceState(null,"","#"+name)}catch(_){}}
  if(name==="console")conPoll();
  if(name==="scan"&&!dsLoaded)loadDatasets();
  if(name==="records")loadRecords();
  if(name==="files"&&D){renderChart();renderList()}
}
$("#tabs").onclick=e=>{const t=e.target.closest("button");if(t)showTab(t.dataset.tab)};
document.addEventListener("click",e=>{const g=e.target.closest("[data-goto]");if(g){e.preventDefault();showTab(g.dataset.goto)}});
$("#scanPill").onclick=()=>showTab("scan");
window.addEventListener("hashchange",()=>showTab(location.hash.slice(1),false));
function setPill(s,cls,lost){
  const pill=$("#scanPill");
  if(!s||s.state!=="scanning"){pill.hidden=true;return}
  const steps=s.steps||"",done=[...steps].filter(c=>"kcf".includes(c)).length;
  pill.hidden=false;
  pill.querySelector(".dot").className="dot "+cls;
  $("#scanPillText").textContent=lost?"Lost contact":
    s.phase==="diff"&&steps?`Scanning ${done}/${steps.length}`:(PHASES[s.phase]||"Scanning");
}

/* ---------- scanning ---------- */
async function loadDatasets(){
  dsLoaded=true;
  try{
    const ds=await(await api("/api/datasets")).json();
    const prev=D?new Set(D.datasets.map(d=>d.name)):null;
    $("#dsList").innerHTML=ds.length?ds.map(d=>`<label class="dsrow"><input type="checkbox" value="${esc(d.name)}"${!prev||prev.has(d.name)?" checked":""}><span class="dsn"><b>${esc(d.name)}</b><span class="muted mono">${esc(d.mountpoint)}</span><span class="m">${fmtNum(d.snapshots)} snapshots · ${fmtDate(d.oldest)} → ${fmtDate(d.newest)} · ${fmtBytes(d.used)}</span></span></label>`).join("")
      :`<div class="empty"><h3>No datasets with snapshots</h3>Check that datasets are mounted and that you have snapshots.</div>`;
  }catch(err){dsLoaded=false;$("#dsList").innerHTML=`<div class="empty"><h3>Couldn't list datasets</h3><span class="mono">${esc(err.message)}</span></div>`}
}
$("#dsAll").onclick=()=>$$("#dsList input").forEach(i=>i.checked=true);
$("#dsNone").onclick=()=>$$("#dsList input").forEach(i=>i.checked=false);
$("#dsGo").onclick=async()=>{const names=$$("#dsList input:checked").map(i=>i.value);if(!names.length)return toast("Pick at least one dataset");
  $("#dsGo").disabled=true;await post("/api/scan",{datasets:names});poll()};
$("#cancelBtn").onclick=()=>post("/api/cancel",{});

/* ---------- live progress panel ---------- */
const PHASES={listing:"Listing snapshots",diff:"Comparing snapshots",stat:"Reading file details",build:"Building results"};
const KINDS_UI={zfs:["ZFS DIFF","--c-image"],resync:["ZFS DIFF (vs live)","--c-image"],tree:["FOLDER LISTING","--c-code"],
  save:["SAVING","--c-audio"],process:["PROCESSING","--c-doc"],load:["LOADING FROM CACHE","--c-doc"],
  stat:["FILE DETAILS","--c-archive"],list:["LISTING","--c-other"]};
const STEP_UI={q:"queued",d:"zfs diff running",t:"listing folders",s:"saving to cache",w:"diffed, waiting its turn (processed newest to oldest)",
  p:"processing",k:"done",c:"done (loaded from cache)",f:"fell back / failed"};
const STEP_COLOR={q:"var(--surface-3)",d:"var(--c-image)",t:"var(--c-code)",s:"var(--c-audio)",
  w:"color-mix(in srgb,var(--c-image) 35%,var(--surface-3))",p:"var(--c-doc)",k:"var(--ok)",
  c:"color-mix(in srgb,var(--ok) 50%,var(--surface-3))",f:"var(--danger)"};
const fmtDur=s=>{s=Math.max(0,Math.floor(s));return s<60?s+"s":s<3600?`${Math.floor(s/60)}m ${String(s%60).padStart(2,"0")}s`:`${Math.floor(s/3600)}h ${String(Math.floor(s%3600/60)).padStart(2,"0")}m`};
let stepLabels=[],labelsFor=null,stepsDrawn="",rates={},pollFails=0,lastStatus=null;

/* Is this task demonstrably doing something? Counter moved recently, or its process/thread
   is using CPU or reading disk (Linux /proc), or it's blocked in a disk read (state D). */
function liveness(t,now){
  const age=now-t.last,cpu=t.cpu??null,rd=t.read??null;
  if(age<20)return["ok","working"];
  if(t.state==="D")return["ok","reading disk"];
  if(cpu!==null&&cpu>=3)return["ok",`working (CPU ${cpu}%)`];
  if(rd!==null&&rd>0)return["ok","reading disk"];
  if(age<300)return["quiet",`quiet for ${fmtDur(age)}`];
  return["bad",`no progress for ${fmtDur(age)}`];
}
function renderProgress(s,lost){
  const p=$("#progress");p.classList.add("on");p.classList.toggle("lost",!!lost);
  $("#progCard").classList.toggle("lost",!!lost);$("#scanIdle").style.display="none";
  const now=s.serverTime;
  const states=s.tasks.map(t=>liveness(t,now)[0]);
  const [cls,txt]=lost?["bad","Lost contact with SnapSift"]:
    !s.tasks.length||states.includes("ok")?["ok","Working"]:states.includes("quiet")?["quiet","Quiet"]:["bad","No progress"];
  $("#pLive").innerHTML=`<span class="dot ${cls}"></span>${txt}`;
  setPill(s,cls,lost);
  $("#pPhase").textContent=lost?"":PHASES[s.phase]||"Scanning";
  const steps=s.steps||"";
  const doneN=[...steps].filter(c=>c==="k"||c==="c"||c==="f").length;
  $("#pText").textContent=lost
    ?`No reply for ${fmtDur((Date.now()-lost)/1000)}. It may have been stopped (for example, the shell it ran in closed). Showing the last known state; retrying.`
    :[s.phase==="diff"&&steps?`${doneN} / ${steps.length} steps processed`:s.total?`${fmtNum(s.done)} / ${fmtNum(s.total)}`:"",
      `${fmtNum(s.found)} found`,`elapsed ${fmtDur(now-s.started)}`].filter(Boolean).join(" · ");
  // Processing is strictly in order; say what it's waiting for, so a still counter isn't a mystery.
  const wo=s.waitingOn,waitsOnDiff=!lost&&wo!=null&&"dts".includes(steps[wo]);
  const waiting=[...steps].filter(c=>c==="w").length;
  $("#pWait").innerHTML=waitsOnDiff?`⏳ Processing is waiting for step ${wo+1} to finish its diff: <span class="mono">${esc(stepLabels[wo]||"")}</span>`+
    (waiting?` · ${waiting} later step${waiting===1?" is":"s are"} already diffed and will be processed right after it.`:""):"";
  const errs=s.errors||[];
  if(steps+errs.length!==stepsDrawn){
    stepsDrawn=steps+errs.length;
    const errFor=lbl=>{const m=lbl&&errs.filter(e=>e.startsWith(lbl)||e.startsWith(lbl.split(" → ")[0]+":"));return m&&m.length?"\n"+m.join("\n"):""};
    $("#pSteps").innerHTML=[...steps].map((c,i)=>`<i class="${c}" title="${esc((stepLabels[i]||"step "+(i+1))+" — "+STEP_UI[c]+(c==="f"?errFor(stepLabels[i]):""))}"></i>`).join("");
    $("#pWarn").innerHTML=errs.length?`<details><summary>⚠ ${errs.length} warning${errs.length===1?"":"s"} so far — latest: ${esc(errs[errs.length-1].slice(0,220))}</summary><div class="mono">${esc(errs.slice(-30).join("\n"))}</div></details>`:"";
    const present=[..."dtswpkcfq"].filter(c=>steps.includes(c));
    $("#pLegend").innerHTML=present.map(c=>`<span><i style="background:${STEP_COLOR[c]}"></i>${STEP_UI[c]}</span>`).join("");
  }
  $("#pSteps").style.display=$("#pLegend").style.display=steps?"":"none";
  $("#pTasks").innerHTML=s.tasks.map(t=>{
    const[name,col]=KINDS_UI[t.kind]||[t.kind.toUpperCase(),"--c-other"];
    const[lc,lt]=lost?["bad","unknown"]:liveness(t,now);
    const prev=rates[t.id];let rate=prev?prev.rate:0;
    if(prev&&now-prev.t>=0.5){const r=(t.n-prev.n)/(now-prev.t);rate=prev.rate?prev.rate*0.6+r*0.4:r}
    if(!prev||now-prev.t>=0.5)rates[t.id]={n:t.n,t:now,rate};
    const meta=[];
    if(t.unit)meta.push(`<b>${fmtNum(t.n)}</b>${t.total?" / "+fmtNum(t.total):""} ${esc(t.unit)}`);
    if(rate>0.5)meta.push(`${fmtNum(Math.round(rate))}/s`);
    if(t.cpu!=null)meta.push(`CPU <b>${t.cpu}%</b>`);
    if(t.read!=null)meta.push(`disk read <b>${fmtBytes(t.read)}/s</b>`);
    meta.push(`running ${fmtDur(now-t.started)}`);
    if(now-t.last>=60&&t.unit)meta.push(`last count change ${fmtDur(now-t.last)} ago`);
    const pct=t.total?Math.min(100,t.n/t.total*100):null;
    return `<div class="task" style="--kc:var(${col})"><span class="kchip"><i></i>${name}</span><span class="tl mono" title="${esc(t.label)}">${esc(t.label)}</span>`+
      `<span class="tstate"><span class="dot ${lc}"></span>${lt}</span><div class="tmeta">${meta.join("<span class=muted>·</span>")}</div>`+
      (pct!=null?`<div class="tbar"><b style="width:${pct}%"></b></div>`:"")+`</div>`;
  }).join("")||`<div class="muted">Between tasks…</div>`;
  for(const id in rates)if(!s.tasks.some(t=>String(t.id)===id))delete rates[id];
}

/* ---------- console view ---------- */
let conLast=0,conT=null,conRaw=[];
async function conPoll(){
  clearTimeout(conT);
  if(tab!=="console")return;
  try{
    const r=await(await api(`/api/log?after=${conLast}`)).json();
    if(r.file)$("#conNote").textContent=`The same output as the terminal SnapSift runs in, updating live. Also saved to ${r.file}`;
    if(r.lines.length){
      conLast=r.lines[r.lines.length-1][0];
      const pre=$("#conText");
      const html=r.lines.map(([,t])=>{conRaw.push(t);
        const c=/WARNING|FAILED/.test(t)?"w":/ === /.test(t)?"ph":/ status /.test(t)?"st":"";
        return c?`<span class="${c}">${esc(t)}</span>`:esc(t)}).join("\n");
      pre.insertAdjacentHTML("beforeend",(pre.childNodes.length?"\n":"")+html);
      if($("#conFollow").checked)pre.scrollTop=pre.scrollHeight;
    }
  }catch(err){}
  conT=setTimeout(conPoll,1500);
}
$("#conCopy").onclick=()=>copy(conRaw.join("\n"));
$("#conDl").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([conRaw.join("\n")+"\n"],{type:"text/plain"}));a.download="snapsift.log";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)};

let pollT=null;
async function poll(){
  clearTimeout(pollT);
  const wantLabels=!lastStatus||labelsFor!==lastStatus.started||stepLabels.length!==(lastStatus.steps||"").length;
  try{status=await(await api("/api/status"+(wantLabels?"?labels=1":""))).json();pollFails=0}
  catch(err){
    pollFails++;
    if(lastStatus&&lastStatus.state==="scanning"&&pollFails>=2){
      if(!poll.lostAt)poll.lostAt=Date.now();
      renderProgress(lastStatus,poll.lostAt);
    }
    pollT=setTimeout(poll,2000);return;
  }
  poll.lostAt=null;
  const p=$("#progress"),s=status,wasScanning=lastStatus&&lastStatus.state==="scanning";lastStatus=s;
  if(s.stepLabels){stepLabels=s.stepLabels;labelsFor=s.started;stepsDrawn=""}
  if(s.state==="scanning"){
    renderProgress(s,null);
    if(!D)$("#rows").innerHTML=`<div class="empty"><h3>Scan in progress</h3>Results appear here when it finishes. Follow it on the <a href="#scan" data-goto="scan">Scan</a> tab.</div>`;
    $("#dsGo").disabled=true;$("#dsGo").textContent="Scan running…";
    pollT=setTimeout(poll,1000);return;
  }
  p.classList.remove("on");$("#progCard").classList.remove("lost");$("#scanIdle").style.display="";
  setPill(null);
  $("#dsGo").disabled=false;$("#dsGo").textContent="Start scan";
  $("#lastScan").textContent=s.resultAt?`Last scan finished ${fmtAgo(s.resultAt)}${s.finished&&s.started?` after ${fmtDur(s.finished-s.started)}`:""}.`:"";
  if(wasScanning){dsLoaded=false;if(s.state==="done"){toast(`Scan finished: ${fmtNum(s.found)} files`);if(tab==="scan")showTab("files")}}
  $("#scanInfo").textContent=s.resultAt?`${s.method} · scanned ${fmtAgo(s.resultAt)}`:"";
  if(s.errors&&s.errors.length){$("#errbar").classList.add("on");
    $("#errbar").innerHTML=`⚠ ${s.errors.length} problem${s.errors.length===1?"":"s"} during the ${s.state==="error"?"failed ":""}scan. <details><summary style="cursor:pointer;display:inline">details</summary><div class="mono" style="margin-top:6px;white-space:pre-wrap">${esc(s.errors.slice(0,50).join("\n"))}</div></details>`}
  else $("#errbar").classList.remove("on");
  if(s.state==="cancelled")toast("Scan cancelled");
  $("#startWarn").innerHTML=(s.startupWarnings||[]).map(w=>`⚠ ${esc(w)}`).join("<br>");
  if(VIEWREC){if(wasScanning&&s.state==="done")toast("New scan results are ready. Use “Back to current scan” to see them.")}
  else if(s.hasResult&&(!D||D.scannedAt!==s.resultAt))await loadData();
  else if(!s.hasResult&&!D){
    $("#rows").innerHTML=`<div class="empty"><h3>No results yet</h3>Run a scan on the <a href="#scan" data-goto="scan">Scan</a> tab.</div>`;
    if(!wasScanning)showTab("scan");
  }
}
/* Load the current scan (rec=null) or a saved record into the Files tab. */
async function loadData(rec=null){
  $("#rows").innerHTML=`<div class="empty">Loading ${rec?"record":"results"}…</div>`;
  const res=await(await api(rec?`/api/records/data?id=${encodeURIComponent(rec.id)}`:"/api/data")).json();
  VIEWREC=rec;ALIVE=null;F.goneOnly=false;closeDrawer();
  prep(res);
  if(rec){
    try{  // one `zfs list`: which of the record's snapshots still exist
      const a=await(await api(`/api/records/alive?id=${encodeURIComponent(rec.id)}`)).json();
      ALIVE=new Uint8Array(S.length);for(const g of a.alive)ALIVE[g]=1;
    }catch(err){toast("Couldn't check which snapshots still exist: "+err.message)}
  }
  computeGone();computeP();
  syncView();renderSnaps();refresh();renderBanner();
  $("#scanInfo").textContent=rec?`record · scanned ${fmtDate(res.scannedAt)}`:`${res.method} · scanned ${fmtAgo(res.scannedAt)}`;
}
function renderBanner(){
  const b=$("#recBanner");
  if(!VIEWREC){b.hidden=true;return}
  const alive=ALIVE?ALIVE.reduce((a,x)=>a+x,0):null,gone=E.reduce((a,e)=>a+(e.gone?1:0),0);
  b.hidden=false;
  b.innerHTML=`<span>📁 Viewing record <b>${esc(VIEWREC.name)}</b>: scan from ${fmtDT(VIEWREC.scannedAt)}, saved ${fmtDT(VIEWREC.savedAt)}.`+
    (alive!=null?` ${fmtNum(alive)} of ${fmtNum(S.length)} snapshots still exist${gone?`; <b>${fmtNum(gone)}</b> files are gone for good (no snapshot left).`:"."}`:"")+`</span>`+
    `<span class="sp"></span><button id="bannerBack">Back to current scan</button>`;
  $("#bannerBack").onclick=()=>backToCurrent();
}
async function backToCurrent(){
  if(!lastStatus||!lastStatus.hasResult){VIEWREC=null;ALIVE=null;renderBanner();return toast("No current scan results yet")}
  await loadData(null);renderRecords();
}

/* ---------- records tab ---------- */
let RECS=[];
async function loadRecords(){
  try{const r=await(await api("/api/records")).json();RECS=r.records;$("#recDir").textContent=r.dir?`Stored in ${r.dir}`:"";renderRecords()}
  catch(err){$("#recList").innerHTML=`<div class="muted">${esc(err.message)}</div>`}
}
function renderRecords(){
  const K={auto:["automatic","auto"],destroy:["before destroy","destroy"],manual:["saved","manual"]};
  $("#recList").innerHTML=RECS.length?RECS.map(m=>{const[kl,kc]=K[m.kind]||[m.kind,""];
    return `<div class="recrow${VIEWREC&&VIEWREC.id===m.id?" cur":""}"><div class="t">${esc(m.name)}<span class="kbadge ${kc}">${kl}</span></div>`+
      `<div class="m">${fmtNum(m.deleted)} deleted files (${fmtBytes(m.deletedBytes)}) · ${fmtNum(m.snapshots)} snapshots, ${fmtDate(m.oldest)} → ${fmtDate(m.newest)} · ${esc((m.datasets||[]).join(", "))}<br>scanned ${fmtDT(m.scannedAt)} · saved ${fmtDT(m.savedAt)} · ${fmtBytes(m.bytesOnDisk)} on disk</div>`+
      `<div class="acts"><button class="primary" data-open="${esc(m.id)}">${VIEWREC&&VIEWREC.id===m.id?"Viewing":"Open"}</button><a class="btn" href="/api/records/data?id=${encodeURIComponent(m.id)}" download="snapsift-record-${esc(m.id)}.json">Download</a><button class="danger" data-del="${esc(m.id)}">Delete</button></div>`+
      (m.note?`<div class="note">${esc(m.note)}</div>`:"")+`</div>`}).join("")
    :`<div class="muted" style="padding:12px 0">No records yet. One is saved automatically after each scan.</div>`;
}
$("#recList").onclick=async ev=>{
  const o=ev.target.dataset.open,d=ev.target.dataset.del;
  if(o){const m=RECS.find(x=>x.id===o);if(m){await loadData(m);showTab("files");renderRecords()}}
  if(d){const m=RECS.find(x=>x.id===d);if(!m||!confirm(`Delete the record "${m.name}"? This can't be undone.`))return;
    await post("/api/records/delete",{id:d});if(VIEWREC&&VIEWREC.id===d)await backToCurrent();loadRecords()}
};
$("#rsSave").onclick=async()=>{
  $("#rsSave").disabled=true;
  try{const m=await(await post("/api/records/save",{name:$("#rsName").value,note:$("#rsNote").value})).json();
    toast(`Saved record "${m.name}"`);$("#rsName").value="";$("#rsNote").value="";loadRecords()}
  catch(err){toast(err.message)}
  $("#rsSave").disabled=false;
};
$("#rows").innerHTML=`<div class="empty">Connecting…</div>`;
showTab(location.hash.slice(1)||LS.get("tab","files"),false);
poll();
recPoll();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
