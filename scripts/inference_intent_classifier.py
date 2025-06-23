import torch
from packaging import version

# Enforce torch version >= 2.6 for security (CVE-2025-32434)
if version.parse(torch.__version__) < version.parse("2.6.0"):
    raise RuntimeError(
        f"torch >= 2.6.0 is required due to CVE-2025-32434. Current version: {torch.__version__}. "
        "Please upgrade torch: pip install --upgrade 'torch>=2.6'"
    )

# ...rest of your inference code...
