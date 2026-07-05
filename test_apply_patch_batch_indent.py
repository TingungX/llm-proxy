#!/usr/bin/env python3
"""测试 batch 模式下 apply_patch 反向转换是否丢失缩进。"""

import sys
sys.path.insert(0, "/workspace")

from llm_proxy.protocol.responses_chat.tool_replacement import reverse_tool_args_to_apply_patch

# 模拟上游模型返回的 batch 结构化参数
args = {
    "action": "batch",
    "operations": [
        {
            "action": "update_file",
            "filePath": "/tmp/a.py",
            "old_str": "    def old_a(self):\n        return 1",
            "new_str": "    def new_a(self):\n        return 2",
        },
        {
            "action": "update_file",
            "filePath": "/tmp/b.py",
            "old_str": "    def old_b(self):\n        return 3",
            "new_str": "    def new_b(self):\n        return 4",
        },
    ],
}

reversed_dsl = reverse_tool_args_to_apply_patch(args)
print("=== 反向转换后的 DSL ===")
print(reversed_dsl)
print()
print("=== 每行 repr ===")
for i, line in enumerate(reversed_dsl.splitlines()):
    print(f"{i+1}: {line!r}")
