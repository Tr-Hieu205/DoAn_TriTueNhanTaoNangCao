"""
data/download_data.py
Automated acquisition and caching of the official UIT-VSFC dataset splits
(train, validation, test) from the Hugging Face Dataset Parquet endpoint.

CLI Usage:
    python data/download_data.py [--output_dir data/uit_vsfc] [--force] [--save_csv]
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from typing import Dict, List, Optional

# Ensure UTF-8 output on Windows PowerShell / cmd to prevent UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Official Hugging Face Parquet URLs for UIT-VSFC
HF_PARQUET_API_URLS = {
    "train": "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/train/0.parquet",
    "validation": "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/validation/0.parquet",
    "test": "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/test/0.parquet",
}

# Fallback resolve endpoints in case the API endpoint undergoes schema changes
HF_PARQUET_RESOLVE_FALLBACK = {
    "train": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}

# Official expected row counts per split for sanity verification
EXPECTED_ROW_COUNTS = {
    "train": 11426,
    "validation": 1583,
    "test": 3166,
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _download_stream_to_file(url: str, dest_path: str, retries: int = 3, timeout: int = 30) -> int:
    """
    Downloads a binary stream from URL to dest_path with retry backoff and atomic renaming.
    Uses standard library urllib.request with zero external dependencies.
    """
    temp_path = f"{dest_path}.tmp"
    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status != 200:
                    raise urllib.error.HTTPError(
                        url, response.status, f"HTTP Error {response.status}", response.headers, None
                    )

                with open(temp_path, "wb") as f:
                    chunk_size = 64 * 1024  # 64 KB chunks
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)

            # Validate non-empty file
            downloaded_bytes = os.path.getsize(temp_path)
            if downloaded_bytes < 1000:
                raise ValueError(f"Downloaded file suspiciously small ({downloaded_bytes} bytes)")

            # Validate Parquet magic bytes ('PAR1')
            with open(temp_path, "rb") as f:
                magic = f.read(4)
                if magic != b"PAR1":
                    raise ValueError(f"Corrupted download: expected magic header b'PAR1', found {magic}")

            # Atomically rename temporary file to final target
            os.replace(temp_path, dest_path)
            return downloaded_bytes

        except Exception as exc:
            last_exception = exc
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt < retries:
                sleep_sec = attempt * 1.5
                print(f"[WARN] Download attempt {attempt} failed ({exc}). Retrying in {sleep_sec:.1f}s...")
                time.sleep(sleep_sec)
            else:
                break

    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_exception}")


def download_vsfc_dataset(
    output_dir: Optional[str] = None,
    force: bool = False,
    save_csv: bool = False,
    splits: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Acquires the UIT-VSFC dataset splits into local cache.

    Args:
        output_dir: Target cache directory. Defaults to 'data/uit_vsfc'.
        force: If True, re-downloads even if files already exist.
        save_csv: If True, also exports companion CSV copies of the parquet files.
        splits: List of split names to acquire (defaults to ['train', 'validation', 'test']).

    Returns:
        Dict mapping split_name to absolute path of the local parquet file.
    """
    if splits is None:
        splits = ["train", "validation", "test"]

    if output_dir is None:
        # Default to relative folder 'data/uit_vsfc' anchored from script location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "uit_vsfc")

    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, str] = {}

    print("=" * 70)
    print("UIT-VSFC Dataset Acquisition Pipeline")
    print(f"Target directory: {os.path.abspath(output_dir)}")
    print("=" * 70)

    for split in splits:
        target_parquet = os.path.join(output_dir, f"{split}.parquet")

        # Caching check
        if os.path.exists(target_parquet) and os.path.getsize(target_parquet) > 1000 and not force:
            size_bytes = os.path.getsize(target_parquet)
            print(f"[*] {split:10s} : CACHED ({size_bytes:,} bytes) -> {target_parquet}")
            results[split] = target_parquet
        else:
            primary_url = HF_PARQUET_API_URLS.get(split)
            fallback_url = HF_PARQUET_RESOLVE_FALLBACK.get(split)
            if not primary_url:
                raise ValueError(f"Unknown split '{split}'. Available: {list(HF_PARQUET_API_URLS.keys())}")

            print(f"[*] {split:10s} : Downloading from Hugging Face...")
            t0 = time.time()
            try:
                downloaded_size = _download_stream_to_file(primary_url, target_parquet)
            except Exception as e:
                print(f"[WARN] Primary API endpoint failed ({e}). Trying fallback resolve endpoint...")
                if not fallback_url:
                    raise
                downloaded_size = _download_stream_to_file(fallback_url, target_parquet)

            elapsed = time.time() - t0
            print(f"[OK] {split:10s} : Downloaded ({downloaded_size:,} bytes in {elapsed:.2f}s) -> {target_parquet}")
            results[split] = target_parquet

        # Optional CSV export for student transparency
        if save_csv:
            csv_path = os.path.join(output_dir, f"{split}.csv")
            if not os.path.exists(csv_path) or force:
                _export_parquet_to_csv(target_parquet, csv_path)

    # Summary verification
    print("-" * 70)
    _verify_dataset_summary(results)
    print("=" * 70)
    print("Dataset acquisition complete. All splits ready for modeling.")
    return results


def _export_parquet_to_csv(parquet_path: str, csv_path: str) -> None:
    """Exports a parquet file to CSV using polars or pandas if available."""
    try:
        import polars as pl
        df = pl.read_parquet(parquet_path)
        df.write_csv(csv_path)
        print(f"    -> Exported CSV: {csv_path}")
        return
    except ImportError:
        pass

    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"    -> Exported CSV: {csv_path}")
        return
    except ImportError:
        print("    [NOTICE] Neither polars nor pandas installed; skipped CSV export.")


def _verify_dataset_summary(results: Dict[str, str]) -> None:
    """Validates row counts against expected academic standard."""
    for split, path in results.items():
        row_count = None
        try:
            import polars as pl
            row_count = pl.read_parquet(path).shape[0]
        except ImportError:
            try:
                import pandas as pd
                row_count = len(pd.read_parquet(path))
            except ImportError:
                pass

        expected = EXPECTED_ROW_COUNTS.get(split)
        if row_count is not None and expected is not None:
            status = "MATCH" if row_count == expected else f"MISMATCH (expected {expected})"
            print(f"Validation: {split:10s} | Rows: {row_count:,} [{status}]")
        else:
            file_size = os.path.getsize(path)
            print(f"Validation: {split:10s} | Size: {file_size:,} bytes [OK]")


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache the UIT-VSFC dataset splits for sentiment and topic classification."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Local directory to store parquet files (default: data/uit_vsfc)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist in local cache",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Also export human-readable CSV files alongside parquet splits",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Specific splits to download (default: train validation test)",
    )

    args = parser.parse_args()
    download_vsfc_dataset(
        output_dir=args.output_dir,
        force=args.force,
        save_csv=args.save_csv,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
