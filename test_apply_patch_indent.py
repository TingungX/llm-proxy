#!/usr/bin/env python3
"""测试 apply_patch 反向转换是否丢失第一行缩进。"""

import sys
sys.path.insert(0, "/workspace")

from llm_proxy.protocol.responses_chat.tool_replacement import (
    parse_apply_patch_to_simple,
    tool_result_to_action_call,
    reverse_tool_args_to_apply_patch,
)

# 原始 DSL：第一行 replace 的内容有缩进
original_dsl = """*** Begin Patch
*** Update File: /tmp/test.py
@@
 def old_func():
-    a = 1
+    a = 2
     return a
*** End Patch"""

print("=== 原始 DSL ===")
print(original_dsl)
print()

parsed = parse_apply_patch_to_simple(original_dsl)
print("=== 解析结果 ===")
for op in parsed:
    print(op)
print()

action_call = tool_result_to_action_call(parsed[0])
print("=== action_call ===")
print(action_call)
print()

reversed_dsl = reverse_tool_args_to_apply_patch(action_call)
print("=== 反向转换后的 DSL ===")
print(reversed_dsl)
print()

print("=== 比较第一行 replace 的缩进 ===")
orig_lines = original_dsl.splitlines()
rev_lines = reversed_dsl.splitlines()
for i, (o, r) in enumerate(zip(orig_lines, rev_lines)):
    if o != r:
        print(f"第 {i+1} 行不同:")
        print(f"  原始: {o!r}")
        print(f"  反向: {r!r}")
