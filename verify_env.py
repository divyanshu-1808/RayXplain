"""
RayXplain - Environment & Hardware Verification Script
Phase 1: Diagnostic check for PyTorch, CUDA/CPU, OpenCV, Gradio, Joblib, and model artifacts.
"""

import sys
import os
from pathlib import Path

# Terminal formatting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")


def check_module(module_name: str, import_name: str = None):
    target = import_name or module_name
    try:
        mod = __import__(target)
        version = getattr(mod, "__version__", "installed (no __version__)")
        print(f"  [{GREEN}PASS{RESET}] {module_name:<18} -> v{version}")
        return True, mod
    except ImportError as e:
        print(f"  [{RED}FAIL{RESET}] {module_name:<18} -> NOT FOUND ({e})")
        return False, None
    except Exception as e:
        print(f"  [{YELLOW}WARN{RESET}] {module_name:<18} -> Import error: {e}")
        return False, None


def main():
    print(f"\n{BOLD}RayXplain | Environment Diagnostic & Verification Suite{RESET}")
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Working Directory : {os.getcwd()}")

    # 1. Dependency Checks
    print_header("1. Core Dependencies Check")
    modules_to_test = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("opencv-python", "cv2"),
        ("gradio", "gradio"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
        ("pillow", "PIL"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]

    all_passed = True
    imported_modules = {}

    for name, imp in modules_to_test:
        passed, mod = check_module(name, imp)
        if passed:
            imported_modules[name] = mod
        else:
            all_passed = False

    # 2. Hardware Acceleration Check (PyTorch & CUDA)
    print_header("2. Hardware & Acceleration Diagnostics")
    if "torch" in imported_modules:
        torch = imported_modules["torch"]
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            print(f"  [{GREEN}PASS{RESET}] CUDA Acceleration : {BOLD}ENABLED{RESET}")
            print(f"  [{GREEN}INFO{RESET}] Active GPU         : {gpu_name} ({device_count} device(s) found)")
            print(f"  [{GREEN}INFO{RESET}] Total VRAM        : {vram_gb:.2f} GB")
            print(f"  [{GREEN}INFO{RESET}] PyTorch CUDA Build : {cuda_version}")
            print(f"  [{GREEN}INFO{RESET}] cuDNN Version      : {cudnn_ver}")

            # Quick Tensor Compute Verification
            try:
                x = torch.randn(512, 512, device="cuda")
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
                print(f"  [{GREEN}PASS{RESET}] CUDA Kernel Test   : Matrix multiplication executed on GPU successfully")
            except Exception as e:
                print(f"  [{RED}FAIL{RESET}] CUDA Kernel Test   : Failed to run computation on GPU: {e}")
                all_passed = False
        else:
            print(f"  [{YELLOW}WARN{RESET}] CUDA Acceleration : {BOLD}DISABLED{RESET} (Running on CPU)")
            print(f"         PyTorch is configured for CPU-only execution or GPU drivers/toolkit are absent.")
            try:
                x = torch.randn(256, 256)
                y = torch.matmul(x, x)
                print(f"  [{GREEN}PASS{RESET}] CPU Compute Test   : Matrix multiplication executed on CPU successfully")
            except Exception as e:
                print(f"  [{RED}FAIL{RESET}] CPU Compute Test   : Failed: {e}")
                all_passed = False
    else:
        print(f"  [{RED}SKIP{RESET}] PyTorch is not installed. Skipping hardware acceleration diagnostics.")

    # 3. RayXplain Artifacts Validation
    print_header("3. Project Artifacts Validation")
    root_path = Path(__file__).resolve().parent

    # Model Weights Check
    model_path = root_path / "weights" / "best_multimodal_model.pth"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  [{GREEN}PASS{RESET}] Model Weights      : Found -> {model_path.name} ({size_mb:.2f} MB)")
        if "torch" in imported_modules:
            try:
                # Safe inspection of checkpoint structure
                checkpoint = imported_modules["torch"].load(str(model_path), map_location="cpu")
                if isinstance(checkpoint, dict):
                    keys_count = len(checkpoint.keys())
                    print(f"  [{GREEN}INFO{RESET}] Checkpoint State   : Valid dictionary containing {keys_count} layer/state keys")
                else:
                    print(f"  [{GREEN}INFO{RESET}] Checkpoint Type    : {type(checkpoint).__name__}")
            except Exception as e:
                print(f"  [{YELLOW}WARN{RESET}] Checkpoint Check   : Could not inspect weight tensors ({e})")
    else:
        print(f"  [{RED}FAIL{RESET}] Model Weights      : MISSING at {model_path}")
        all_passed = False

    # Scaler Check
    scaler_path = root_path / "data" / "scaler.joblib"
    if scaler_path.exists():
        size_kb = scaler_path.stat().st_size / 1024
        print(f"  [{GREEN}PASS{RESET}] Clinical Scaler    : Found -> {scaler_path.name} ({size_kb:.2f} KB)")
        if "joblib" in imported_modules:
            try:
                scaler = imported_modules["joblib"].load(str(scaler_path))
                scaler_name = type(scaler).__name__
                n_features = getattr(scaler, "n_features_in_", "Unknown")
                print(f"  [{GREEN}INFO{RESET}] Scaler Object      : {scaler_name} (expects {n_features} clinical features)")
            except Exception as e:
                print(f"  [{YELLOW}WARN{RESET}] Scaler Inspection  : Could not load scaler object ({e})")
    else:
        print(f"  [{RED}FAIL{RESET}] Clinical Scaler    : MISSING at {scaler_path}")
        all_passed = False

    # Summary
    print_header("Verification Summary")
    if all_passed:
        print(f"  {BOLD}{GREEN}ALL CHECKS PASSED:{RESET} RayXplain Phase 1 environment is fully operational!")
        sys.exit(0)
    else:
        print(f"  {BOLD}{YELLOW}ATTENTION REQUIRED:{RESET} Some dependencies or artifacts require configuration.")
        print("  Please review the warnings/failures above and install missing packages.")
        sys.exit(1)


if __name__ == "__main__":
    main()
