"""
Phase 2: download, verify and extract LibriSpeech and MUSAN.

Run from the project root, for example:
    python scripts/download_data.py --datasets dev-clean test-clean
    python scripts/download_data.py --datasets train-clean-100 musan
    python scripts/download_data.py --verify-only

Every step is safe to re-run:
  * downloads resume from a .part file if interrupted
  * archives are checked against the official MD5 (LibriSpeech) before use
  * extraction is skipped if it already completed (marker file)
  * the extracted data is verified by counting utterances, speakers and files

Layout produced
    data/download/                 archives (.tar.gz), deletable after verification
    data/raw/LibriSpeech/<subset>/ speaker/chapter/*.flac + *.trans.txt
    data/raw/musan/{noise,music,speech}/
    data/manifests/download_record.json   sizes, MD5s, mirror, counts, timestamps

Sources (both CC BY 4.0)
    LibriSpeech  https://www.openslr.org/12
    MUSAN        https://www.openslr.org/17
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

# Official OpenSLR mirrors, tried in order. Override with --mirror.
MIRRORS = [
    "https://openslr.elda.org/resources",
    "https://openslr.trmal.net/resources",
    "https://openslr.magicdatatech.com/resources",
]

GB = 1024**3

# md5: from https://openslr.elda.org/resources/12/md5sum.txt
# MUSAN has no published checksum, so its MD5 is computed and recorded instead.
# Expected counts are the published LibriSpeech statistics; MUSAN's paper
# reports 929 noise files, so its count check tolerates a small difference.
DATASETS = {
    "dev-clean": dict(
        slr=12, archive="dev-clean.tar.gz", md5="42e2234ba48799c1f50f24a7926300a1",
        size_gb=0.34, kind="librispeech", utterances=2703, speakers=40),
    "test-clean": dict(
        slr=12, archive="test-clean.tar.gz", md5="32fa31d27d2e1cad72775fee3f4849a9",
        size_gb=0.35, kind="librispeech", utterances=2620, speakers=40),
    "test-other": dict(
        slr=12, archive="test-other.tar.gz", md5="fb5a50374b501bb3bac4815ee91d3135",
        size_gb=0.33, kind="librispeech", utterances=2939, speakers=33),
    "train-clean-100": dict(
        slr=12, archive="train-clean-100.tar.gz", md5="2a93770f6d5c6c964bc36631d331a522",
        size_gb=6.3, kind="librispeech", utterances=28539, speakers=251),
    "musan": dict(
        slr=17, archive="musan.tar.gz", md5=None,
        size_gb=11.0, kind="musan", noise_files=929),
}
CORE = ["dev-clean", "test-clean", "train-clean-100", "musan"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def md5_of(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    size = path.stat().st_size
    with open(path, "rb") as f, tqdm(total=size, unit="B", unit_scale=True,
                                     desc=f"MD5 {path.name}", leave=False) as bar:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            bar.update(len(block))
    return h.hexdigest()


def download(url, dest, retries=12, timeout=60):
    """Stream url to dest, resuming from dest.part. Returns True on success.

    Waits up to about 8 minutes per mirror for the network to come back
    (WiFi drops show up as 'getaddrinfo failed') before trying the next mirror.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have:
            print(f"  resuming from {have / GB:.2f} GB")
        req = urllib.request.Request(url, headers={"User-Agent": "noise-robust-asr/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if have and r.status != 206:      # server ignored Range: restart
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                mode = "ab" if have else "wb"
                with open(part, mode) as f, tqdm(total=total, initial=have, unit="B",
                                                 unit_scale=True, desc=dest.name) as bar:
                    while True:
                        block = r.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                        bar.update(len(block))
            if total and part.stat().st_size != total:
                raise IOError(f"incomplete: {part.stat().st_size} of {total} bytes")
            part.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416:                     # already fully downloaded
                part.replace(dest)
                return True
            print(f"  HTTP {e.code} from {url} (attempt {attempt}/{retries})")
            if e.code in (403, 404):
                return False
        except (urllib.error.URLError, OSError, IOError) as e:
            print(f"  {type(e).__name__}: {e} (attempt {attempt}/{retries})")
        time.sleep(min(60, 5 * attempt))
    return False


def safe_extract(archive, out_dir):
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in tqdm(members, desc=f"Extract {archive.name}", unit="file"):
            tar.extract(m, out_dir, filter="data")   # blocks path traversal


def verify_librispeech(raw, name, spec):
    root = raw / "LibriSpeech" / name
    if not root.is_dir():
        return False, f"missing folder {root}"
    speakers = [p for p in root.iterdir() if p.is_dir()]
    flacs = list(root.rglob("*.flac"))
    trans_lines, missing_audio = 0, 0
    for t in root.rglob("*.trans.txt"):
        for line in t.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trans_lines += 1
            utt_id = line.split(" ", 1)[0]
            if not (t.parent / f"{utt_id}.flac").exists():
                missing_audio += 1
    ok = (len(flacs) == spec["utterances"] == trans_lines
          and len(speakers) == spec["speakers"] and missing_audio == 0)
    detail = (f"{len(flacs)} flac / {trans_lines} transcripts "
              f"(expected {spec['utterances']}), {len(speakers)} speakers "
              f"(expected {spec['speakers']}), {missing_audio} transcripts without audio")
    return ok, detail, dict(flac=len(flacs), transcripts=trans_lines,
                            speakers=len(speakers))


def verify_musan(raw, name, spec):
    root = raw / "musan"
    if not root.is_dir():
        return False, f"missing folder {root}", {}
    counts = {c: len(list((root / c).rglob("*.wav"))) for c in ("noise", "music", "speech")}
    noise_ok = abs(counts["noise"] - spec["noise_files"]) <= 5
    ok = noise_ok and counts["music"] > 0 and counts["speech"] > 0
    detail = (f"noise {counts['noise']} wav (paper reports {spec['noise_files']}), "
              f"music {counts['music']}, speech {counts['speech']}")
    return ok, detail, counts


def verify(raw, name):
    spec = DATASETS[name]
    fn = verify_librispeech if spec["kind"] == "librispeech" else verify_musan
    res = fn(raw, name, spec)
    if len(res) == 2:
        return res[0], res[1], {}
    return res


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=CORE, choices=list(DATASETS),
                    help=f"default: {' '.join(CORE)}")
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--mirror", default=None,
                    help="use only this base URL, e.g. https://openslr.elda.org/resources")
    ap.add_argument("--delete-archives", action="store_true",
                    help="delete each .tar.gz after its extraction is verified")
    ap.add_argument("--verify-only", action="store_true",
                    help="only verify already extracted data")
    args = ap.parse_args()

    dl_dir, raw_dir = args.data_dir / "download", args.data_dir / "raw"
    man_dir = args.data_dir / "manifests"
    for d in (dl_dir, raw_dir, man_dir):
        d.mkdir(parents=True, exist_ok=True)
    record_path = man_dir / "download_record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    mirrors = [args.mirror.rstrip("/")] if args.mirror else MIRRORS

    failed = []
    for name in args.datasets:
        spec = DATASETS[name]
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        marker = raw_dir / f".extracted_{name}"
        archive = dl_dir / spec["archive"]
        entry = record.get(name, {})

        if not args.verify_only and not marker.exists():
            # ---- disk space: archive + extracted copy + 10% margin
            if not archive.exists():
                need = spec["size_gb"] * 2.2 * GB
                free = shutil.disk_usage(args.data_dir).free
                if free < need:
                    print(f"[FAIL] need about {need / GB:.1f} GB free, "
                          f"have {free / GB:.1f} GB on this drive")
                    failed.append(name)
                    continue
                # ---- download
                for base in mirrors:
                    url = f"{base}/{spec['slr']}/{spec['archive']}"
                    print(f"Downloading {url}")
                    if download(url, archive):
                        entry["url"] = url
                        break
                else:
                    print(f"[FAIL] could not download {spec['archive']} from any mirror")
                    failed.append(name)
                    continue

            # ---- checksum
            digest = md5_of(archive)
            entry.update(md5=digest, archive_bytes=archive.stat().st_size)
            if spec["md5"] and digest != spec["md5"]:
                print(f"[FAIL] MD5 mismatch for {archive.name}: got {digest}, "
                      f"expected {spec['md5']}. Deleting it; re-run to download again.")
                archive.unlink()
                failed.append(name)
                continue
            print(f"[PASS] MD5 {digest}"
                  + ("  (official)" if spec["md5"] else "  (recorded, none published)"))

            # ---- extract
            try:
                safe_extract(archive, raw_dir)
            except (tarfile.TarError, OSError, EOFError) as e:
                print(f"[FAIL] extraction of {archive.name} failed: {type(e).__name__}: {e}")
                failed.append(name)
                continue
            marker.write_text(datetime.now(timezone.utc).isoformat())

        # ---- verify
        ok, detail, counts = verify(raw_dir, name)
        print(f"[{'PASS' if ok else 'FAIL'}] {detail}")
        if not ok:
            failed.append(name)
            continue
        entry.update(counts=counts, verified_utc=datetime.now(timezone.utc)
                     .isoformat(timespec="seconds"))
        record[name] = entry
        record_path.write_text(json.dumps(record, indent=2))

        if args.delete_archives and archive.exists():
            archive.unlink()
            print(f"Deleted {archive}")

    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    else:
        print(f"All verified: {', '.join(args.datasets)}")
    print(f"Record: {record_path}")
    print("=" * 70)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
