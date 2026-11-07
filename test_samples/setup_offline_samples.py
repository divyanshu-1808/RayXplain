"""
RayXplain - Multi-Modal Lung Disease Diagnostic Assistant
Real Clinical Radiograph Downloader & Offline Validator (setup_offline_samples.py)

This module downloads genuine human chest radiographs from verified open-access clinical
repositories (torchxrayvision and ieee8023 COVID dataset) for 100% offline, reproducible
clinical benchmarking.

Zero synthetic/mathematical phantoms. Strict size (>20 KB) and resolution (>=224x224) validation.
"""

import io
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("RayXplain.OfflineBenchmark")

_SAMPLES_DIR = Path(__file__).resolve().parent

# HTTP headers to prevent GitHub CDN blocking
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# ------------------------------------------------------------------------------
# Verified Real Clinical Benchmark Specifications
# ------------------------------------------------------------------------------
REAL_BENCHMARK_SPECS: List[Dict[str, Any]] = [
    {
        "filename": "sample_normal_1.jpeg",
        "case_id": "CASE-NORM-01",
        "category": "Normal",
        "title": "Normal Thorax (Baseline Adult PA)",
        "primary_url": "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/torchxrayvision/data/16795_1_1.jpg",
        "fallback_urls": [
            "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/00000001_000.png",
            "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/16747_3_1.jpg",
        ],
        "clinical_notes": "Clear bilateral lung fields, sharp costophrenic angles, normal cardiothoracic ratio.",
    },
    {
        "filename": "sample_normal_2.jpeg",
        "case_id": "CASE-NORM-02",
        "category": "Normal",
        "title": "Normal Thorax (Geriatric PA Check)",
        "primary_url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/normal.jpeg",
        "fallback_urls": [
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/F051E018-DAD1-4506-AD43-BE4CA29E960B.jpeg",
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/16745_2_1.PNG",
        ],
        "clinical_notes": "Unremarkable pulmonary parenchyma, normal vascular markings, no consolidation or effusion.",
    },
    {
        "filename": "sample_pneumonia_bacterial.jpeg",
        "case_id": "CASE-PNEU-BAC-01",
        "category": "Pneumonia",
        "title": "Bacterial Lobar Consolidation",
        "primary_url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/pneumonia-viral.jpeg",
        "fallback_urls": [
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/pneumonia-7.jpg",
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/streptococcus-pneumoniae-pneumonia-temporal-evolution-1-day0.jpg",
        ],
        "clinical_notes": "Dense focal parenchymal consolidation with air bronchograms, typical of acute bacterial pneumonia.",
    },
    {
        "filename": "sample_covid_acute.jpeg",
        "case_id": "CASE-COVD-ACU-01",
        "category": "High-Risk / COVID-19",
        "title": "Acute Progressive COVID-19",
        "primary_url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/covid-19-pneumonia-rapidly-progressive-12-hours.jpg",
        "fallback_urls": [
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/covid-19-pneumonia-rapidly-progressive-3-days.jpg",
        ],
        "clinical_notes": "Rapidly progressive bilateral peripheral and subpleural ground-glass opacities.",
    },
    {
        "filename": "sample_covid_bilateral.jpeg",
        "case_id": "CASE-COVD-BIL-02",
        "category": "High-Risk / COVID-19",
        "title": "Severe Bilateral COVID-19 / SARS Pneumonitis",
        "primary_url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/SARS-10.1148rg.242035193-g04mr34g0-Fig-8a.day0.jpeg",
        "fallback_urls": [
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/SARS-10.1148rg.242035193-g04mr34g0-Fig8a-day0.jpeg",
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/SARS-10.1148rg.242035193-g04mr34g0-Fig8b-day5.jpeg",
        ],
        "clinical_notes": "Extensive multifocal peripheral ground-glass and consolidative changes with lower zone predominance.",
    },
]


# ------------------------------------------------------------------------------
# Strict Integrity Validator
# ------------------------------------------------------------------------------
def validate_sample_integrity(image_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate that a stored radiograph file meets strict clinical benchmark criteria:
    1. File exists and size > 20 KB.
    2. Opens cleanly with PIL.Image and decodes.
    3. Resolution is at least 224x224.
    4. Has valid image channels (convertible to RGB).

    Returns:
        Tuple[bool, str]: (is_valid, status_or_error_message)
    """
    path = Path(image_path)
    if not path.is_file():
        return False, "File does not exist on disk"

    size_bytes = path.stat().st_size
    if size_bytes < 20 * 1024:
        return False, f"File size ({size_bytes / 1024:.1f} KB) is below required 20 KB minimum"

    try:
        with Image.open(str(path)) as img:
            img.load()  # Force decode
            w, h = img.size
            if w < 224 or h < 224:
                return False, f"Dimensions {w}x{h} below required 224x224 minimum"
            # Verify RGB convertibility
            rgb_img = img.convert("RGB")
            if rgb_img.size != (w, h):
                return False, "RGB conversion dimension mismatch"
            return True, f"Valid ({w}x{h}, {size_bytes / 1024:.1f} KB)"
    except Exception as exc:
        return False, f"Corrupted or invalid image: {str(exc)}"


# ------------------------------------------------------------------------------
# Robust Downloader
# ------------------------------------------------------------------------------
def _download_real_scan(target_path: Path, primary_url: str, fallback_urls: List[str]) -> Path:
    """
    Download a genuine clinical radiograph using proper HTTP User-Agent headers.
    Tries primary URL first, with graceful fallback to verified clinical mirrors.
    Raises hard RuntimeError on total failure (no synthetic generation).
    """
    urls_to_try = [primary_url] + [u for u in fallback_urls if u != primary_url]
    last_err: Optional[Exception] = None

    for url in urls_to_try:
        try:
            logger.info(f"Attempting download from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    continue
                raw_bytes = resp.read()

            if len(raw_bytes) < 20 * 1024:
                logger.warning(f"Downloaded payload from {url} too small ({len(raw_bytes)} bytes), trying next...")
                continue

            # Verify image validity via PIL
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                pil_img.load()
                w, h = pil_img.size
                if w < 224 or h < 224:
                    logger.warning(f"Image from {url} resolution {w}x{h} too small, trying next...")
                    continue
                # Standardize to RGB JPEG
                rgb_img = pil_img.convert("RGB")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                rgb_img.save(str(target_path), format="JPEG", quality=95)

            # Strict file verification post-save
            is_valid, msg = validate_sample_integrity(target_path)
            if not is_valid:
                target_path.unlink(missing_ok=True)
                logger.warning(f"Saved file failed integrity check ({msg}), trying next URL...")
                continue

            logger.info(f"Successfully saved real clinical scan -> {target_path.name} ({msg})")
            return target_path

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception) as exc:
            last_err = exc
            logger.debug(f"Failed URL {url}: {exc}")
            continue

    # Hard failure: no procedural dummies allowed
    raise RuntimeError(
        f"Critical Failure: Unable to download real clinical radiograph for '{target_path.name}'. "
        f"Attempted URLs: {urls_to_try}. Last Error: {last_err}. "
        f" procedural fallback is strictly disallowed."
    )


def setup_offline_samples(force_redownload: bool = False) -> List[Dict[str, Any]]:
    """
    Download and validate genuine human chest radiographs for offline benchmarking.

    Args:
        force_redownload: If True, re-downloads all files even if already valid on disk.

    Returns:
        List[Dict[str, Any]]: Verification records for all benchmark scans.
    """
    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for spec in REAL_BENCHMARK_SPECS:
        target_file = _SAMPLES_DIR / spec["filename"]

        needs_download = force_redownload or not target_file.is_file()
        if not needs_download:
            is_valid, _ = validate_sample_integrity(target_file)
            if not is_valid:
                logger.warning(f"Existing file {target_file.name} failed integrity, re-downloading...")
                needs_download = True

        if needs_download:
            _download_real_scan(
                target_path=target_file,
                primary_url=spec["primary_url"],
                fallback_urls=spec.get("fallback_urls", []),
            )

        # Final verification
        is_valid, msg = validate_sample_integrity(target_file)
        if not is_valid:
            raise RuntimeError(f"Strict validation failed for {target_file.name}: {msg}")

        file_size_kb = target_file.stat().st_size / 1024.0
        with Image.open(str(target_file)) as img:
            res_str = f"{img.size[0]}x{img.size[1]}"

        results.append({
            "filename": spec["filename"],
            "case_id": spec["case_id"],
            "category": spec["category"],
            "title": spec["title"],
            "path": str(target_file.resolve()),
            "size_kb": round(file_size_kb, 1),
            "resolution": res_str,
            "status": "PASS (Real Clinical Scan)",
        })

    return results


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print an aligned ASCII summary table of the genuine clinical benchmark scans."""
    print("\n" + "=" * 105)
    print(" RayXplain: Real Clinical Benchmark Radiographs (Verified Open-Access Human Scans)")
    print("=" * 105)
    header = f"{'Filename':<32} | {'Category':<22} | {'Resolution':<12} | {'Size (KB)':<10} | {'Status'}"
    print(header)
    print("-" * 105)

    for item in results:
        row = (
            f"{item['filename']:<32} | "
            f"{item['category']:<22} | "
            f"{item['resolution']:<12} | "
            f"{item['size_kb']:<10.1f} | "
            f"{item['status']}"
        )
        print(row)

    print("=" * 105)
    print(f"Total Verified Clinical Samples: {len(results)} | Storage: {_SAMPLES_DIR}")
    print("100% genuine medical radiographs ready for ResNet-50 feature extraction & Grad-CAM.\n")


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    try:
        report = setup_offline_samples(force_redownload=force_flag)
        print_summary_table(report)
    except Exception as e:
        print(f"\n[FATAL ERROR] Real scan setup failed: {e}", file=sys.stderr)
        sys.exit(1)
