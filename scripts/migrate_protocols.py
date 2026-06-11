#!/usr/bin/env python3
"""迁移 config.json：从 `upstream_protocol` 标量迁移到 `upstream_protocols` 集合。

用法：
  python3 scripts/migrate_protocols.py --dry-run   # 只看 diff，不写
  python3 scripts/migrate_protocols.py --apply     # 备份后写回

行为：
  - 如果 model 只有 `upstream_protocol` 标量 → 包成 `upstream_protocols: [标量]`
  - 如果 model 既有标量又有 `upstream_protocols` 数组 → 合并去重（集合化）
  - 迁移后删除 `upstream_protocol` 标量字段
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def migrate_config(config_path: Path) -> list[tuple[str, str, list[str]]]:
    """读取 config，生成 (model_id, action, new_protocols) 列表。
    不写文件，由调用方决定是否 apply。
    """
    with open(config_path) as f:
        cfg = json.load(f)

    changes: list[tuple[str, str, list[str]]] = []
    for model_id, model_cfg in cfg.get("models", {}).items():
        scalar = model_cfg.get("upstream_protocol")
        existing = model_cfg.get("upstream_protocols")

        if scalar is None and existing is None:
            continue

        if existing is not None and scalar is None:
            # 已经是新格式，但确保是集合（list 去重）
            merged = sorted(set(existing))
            if merged != sorted(existing):
                changes.append((model_id, "dedup", merged))
            continue

        if existing is not None and scalar is not None:
            # 合并
            merged = sorted(set(existing) | {scalar})
            changes.append((model_id, "merge", merged))
        else:
            # 只有标量
            changes.append((model_id, "wrap", [scalar]))

    return changes


def apply_migration(config_path: Path, changes: list[tuple[str, str, list[str]]]) -> str:
    """应用迁移：备份原文件 → 改写 → 返回备份路径。"""
    with open(config_path) as f:
        cfg = json.load(f)

    for model_id, action, new_protocols in changes:
        model_cfg = cfg["models"][model_id]
        if action == "dedup":
            model_cfg["upstream_protocols"] = new_protocols
        else:
            # wrap / merge
            model_cfg["upstream_protocols"] = new_protocols
            if "upstream_protocol" in model_cfg:
                del model_cfg["upstream_protocol"]

    backup = config_path.with_suffix(
        f".bak.migration-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    shutil.copy(config_path, backup)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return str(backup)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="迁移 config.json: upstream_protocol 标量 → upstream_protocols 集合"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="只看 diff，不写文件")
    group.add_argument("--apply", action="store_true", help="备份后写回 config.json")
    parser.add_argument(
        "--config",
        default="config.json",
        help="config 路径（默认 config.json）",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}", file=sys.stderr)
        return 1

    changes = migrate_config(config_path)

    if not changes:
        print("✅ No migration needed — all models already use upstream_protocols set.")
        return 0

    print(f"📋 Migration plan ({len(changes)} models):\n")
    for model_id, action, new_protocols in changes:
        print(f"  [{action:>5}] {model_id}")
        print(f"          → upstream_protocols: {new_protocols}")

    if args.dry_run:
        print("\n🔍 Dry-run mode — no changes written.")
        print("   Run with --apply to write migration.")
        return 0

    backup = apply_migration(config_path, changes)
    print(f"\n✅ Migrated {len(changes)} models.")
    print(f"   Backup: {backup}")
    print(f"   Updated: {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

