# Hierarchical Zero-Hallucination & Multi-Agent System Rules

Apply these rules universally across all projects and code modifications:

## 1. Executive System Hierarchy Protocol
When executing complex engineering tasks, delegate tasks across the 3 Divisions and 6 Specialized Teams:
- **🛡️ Cybersecurity & Compliance Division**:
  - 🔴 **Red Team**: Security & vulnerability audit (path traversal, unsafe unpickling).
  - 🔵 **Blue Team**: System architecture, API parity & default-off flags.
- **⚡ AI Engine & Core ML Division**:
  - 🟢 **Green Team**: ML math integrity, tensor shapes & loss functions.
  - 🟣 **Purple Team**: Zero-training diagnostic smoke tests (`smoke_test.py`).
- **🔧 Infrastructure & Platform Division**:
  - 🟡 **Yellow Team**: VRAM safety (< 8 GB), FP16 AMP & DataLoader memory safety.
  - ⚪ **White Team**: Linux cross-platform path safety (`/`) & deployment packaging.

## 2. Mandatory Diagnostic Verification (Smoke Testing)
- Every code modification MUST pass a zero-training diagnostic script (`smoke_test.py`).
- Diagnostic scripts must complete in < 5 seconds on CPU/GPU without consuming production VRAM or dataset memory.

## 3. Backward Compatibility & Default-Off Guard
- All experimental architectural features must default to `False`.
- The baseline code path must be parity-tested and guaranteed to function identically when experimental flags are turned off.

## 4. File, Path & Security Safety
- Always use standard forward slashes (`/`) for paths inside zip/tar archives intended for Linux runtimes (Kaggle/AWS/GCP).
- Explicitly guard against path traversal vulnerabilities when extracting archives (`_is_safe_archive_member`).
- Always enforce `weights_only=True` when loading PyTorch model checkpoints (`torch.load`).

## 5. Cloud Platform & Memory/Disk Safety Protocol
- **Temporary Archive Storage**: Always direct temporary dataset downloads (`.zip`, `.tar`, `.7z`) to `/tmp` (or `tempfile.gettempdir()`), NEVER into `/kaggle/working` or project workspace root.
- **Immediate Archive Disposal**: Delete archive files from `/tmp` immediately after extraction completes to release temporary memory buffers and prevent `OSError: [Errno 28] No space left on device`.
- **Quota Protection**: Maintain `/kaggle/working` strictly for final model checkpoints, logs, and `submission.csv` to respect the 20 GB disk limit.
- **Zero-Browser Self-Extraction**: For cloud environments (Kaggle/Colab) without easy file uploads, package multi-file solutions into single-cell Base64 self-extracting Python scripts (`self_extract_cell.py`).
