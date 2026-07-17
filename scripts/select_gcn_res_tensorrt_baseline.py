"""Atomically select one validated TensorRT production manifest without moving artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import gcn_res_tensorrt_phase8d_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = common.load_json(manifest_path)
    status = manifest.get("status")
    if status == "production_baseline":
        qualification = manifest.get("qualification", {})
        if qualification.get("promotion_status") != "TENSORRT_CUB_STRICT_FP32_TASK_EQUIVALENT_BASELINE_PROMOTED":
            raise RuntimeError("Production manifest does not contain a successful promotion decision")
        manifest_type = "production"
    elif status == "rollback_baseline_available" and manifest.get("manifest_type") == "rollback":
        engine = Path(manifest["engine_path"]).resolve()
        plugin = Path(manifest["plugin_path"]).resolve()
        common.assert_hash(engine, manifest["engine_sha256"], "rollback engine")
        common.assert_hash(plugin, manifest["plugin_sha256"], "rollback plugin")
        manifest_type = "rollback"
    else:
        raise RuntimeError(f"Cannot select manifest with status {status!r}")
    pointer = PROJECT_ROOT / "deployment/tensorrt/current_baseline.json"
    payload = {
        "deployment_id": manifest["deployment_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": common.sha256(manifest_path),
        "selected_at": common.now_iso(),
        "selection_mode": "explicit_manifest_switch",
        "manifest_type": manifest_type,
        "artifacts_moved_or_deleted": False,
    }
    common.dump_json(pointer, payload)
    print(json.dumps(payload, ensure_ascii=False))
    print("TENSORRT_BASELINE_POINTER_UPDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
