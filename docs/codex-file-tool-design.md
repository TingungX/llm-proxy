# Codex 文件浏览/搜索工具的设计哲学调研

> **背景**：Codex 新增了三个结构化文件工具 `list_dir` / `read_file` / `grep_files`。
> 本调研比较 Codex、Claude Code、OpenCode 三家的设计决策，回答一个问题：
> **这些工具应该多复杂？复杂查询留在工具内，还是回归 Bash？**
>
> **结论先行**：Codex 当前方向反了——3 个工具的参数复杂度比 Claude Code 单个 Glob 还高。
> 应该走"极简工具 + 智能截断 + 显式 Bash 引导"路线。

---

## 一、三方工具对比

| 维度       | Codex（当前）              | Claude Code        | OpenCode               |
| ---------- | -------------------------- | ------------------ | ---------------------- |
| 工具粒度   | 3 个结构化工具             | 15 个分离工具      | ~10 个分离工具         |
| `Glob` 参数 | —                          | **2 个**（pattern, path） | 类似                |
| 搜索限制   | 软（`limit` 按文件截断 30）| **硬截断 100 条**  | 类似                   |
| 截断引导   | ❌ 无                      | ✅ 主动引导其他工具 | 类似                  |
| 复杂查询   | 工具缺失，回归 Bash        | ✅ 直接 Bash 兜底  | 类似                   |

**核心观察**：Claude Code / OpenCode 都遵循**"工具分离 + Bash 兜底"**模式，不试图在
单一工具里塞 ripgrep 完整参数集。Codex 把 3 个工具揉得比 Claude Code 单个工具还重，
方向相反。

---

## 二、Claude Code 关键源码证据

### 2.1 Glob 工具 —— 参数极简

来自 `cnblogs.com/bluestorm/p/19369058`（Claude Code 架构解析，含 TS 源码）：

```typescript
async *call({ pattern, path }, { abortController }) {
  const start = Date.now()
  const { files, truncated } = await glob(
    pattern,
    path ?? getCwd(),
    { limit: 100, offset: 0 }, // ← 硬编码 limit 100
    abortController.signal,
  )
  const output: Output = {
    filenames: files,
    durationMs: Date.now() - start,
    numFiles: files.length,
    truncated,
  }
}
```

**只有 `pattern` 和 `path` 两个参数**。没有 `include` / `exclude` / `sort_by`。
`limit` 硬编码 100，结果超限会被截断。

### 2.2 截断时主动引导用户用其他工具

```typescript
const MAX_LINES = 4
const MAX_FILES = 1000
const TRUNCATED_MESSAGE = `
  There are more than ${MAX_FILES} files in the repository.
  Use the LS tool (passing a specific path), Bash tool, and other tools
  to explore nested directories. The first ${MAX_FILES} files are included below:
`
```

**关键 insight**：不在 Glob 里塞复杂参数，而是截断时直接告诉模型
"用 LS(具体路径) 或 Bash"。

### 2.3 Claude Code 工具分工表（来自官方文档）

| 工具         | 职责             | 复杂度   |
| ------------ | ---------------- | -------- |
| `Read`       | 读文件           | 极简     |
| `Glob`       | 按 pattern 找文件 | **仅 2 个参数** |
| `Grep`       | 内容搜模式       | 仅搜索 + context |
| `LS`         | 列目录           | 极简     |
| `Bash`       | 兜底复杂操作     | 全权     |
| `Write/Edit/MultiEdit` | 写操作 | 各司其职 |

---

## 三、Codex 当前工具的痛点（实测发现）

通过 `~/Projects/github/llm-proxy` 实测，发现以下问题：

### 3.1 `list_dir` 的 `include` 参数语义不清

- `list_dir(depth=1, include="*.py")` 实际行为不符合直觉
- `proxy.py`（顶层 .py 文件）被过滤掉了
- 但 `.bak` / `.db` 文件又没被过滤
- **`include` 的语义边界需要明确，否则会成为调用出错源**

### 3.2 `grep_files` 的 `limit` 行为不符合预期

- `limit` 是 **per-file 截断**，不是全局限制
- 实测 `apply_patch` 模式一次返回 150 个匹配，无法用 `limit=30` 全局截断
- 高频场景："只列匹配的文件名" 做不到（`rg -l` 无对应）

### 3.3 结构化工具的边界缺失

| 场景                       | 结构化工具           | shell                  |
| -------------------------- | -------------------- | ---------------------- |
| 浏览项目结构               | ✅ 自动过滤噪音      | `find` 要 `-not -path` 一堆 |
| 找代码引用                 | ✅ 带上下文 + 总数   | `grep -n` 只有行号     |
| 精准读片段                 | ✅ `offset/limit`    | `sed -n` 拼起来        |
| 只列匹配文件               | ❌                    | `rg -l` 一行          |
| 统计 + 排序                | ❌                    | `wc -l \| sort -rn`    |
| 反转匹配                   | ❌                    | `rg -v`               |
| 自定义输出格式             | ❌                    | `find -printf`         |

**估算比例**：~60% 用结构化工具，~40% 退化到 Bash。
但实际"该用结构化工具的场景"还没被结构化工具完全覆盖。

---

## 四、对 Codex 后续设计的建议

### 4.1 优先级 P0（强烈建议）

1. **`list_dir` 移除 `include` 参数**
   - 语义不清，是 bug 源
   - 如需过滤，用 `path` 指定目录 + 配合 `grep_files` 的 `include`

2. **`grep_files` 加全局 `max_matches` 限制**
   - 当前 `limit` 只 per-file 截断，无全局上限
   - 改为硬截断（如 100）+ `truncated: true` 标记

3. **`grep_files` 加 `files_only: bool`**
   - 对应 `rg -l`，是"快速看哪些文件涉及"的高频场景
   - 一个布尔值，不增加复杂度

### 4.2 优先级 P1（建议）

4. **截断时增加引导文本**
   ```json
   {
     "truncated": true,
     "total_matches": 150,
     "hint": "Use Bash with `rg -l <pattern>` to list matching files only."
   }
   ```

5. **移除 `grep_files` 的 `context_before` / `context_after` 复杂参数**
   - 默认带 2 行上下文即可
   - 需要更多上下文时，用 `read_file` 精准读

### 4.3 优先级 P2（明确不做）

- ❌ 不加 `sort_by` / `files_count` / `invert` / `output_mode=json` 等
- ❌ 不让 `list_dir` 支持按时间/大小排序
- ❌ 不试图让结构化工具覆盖 ripgrep 完整参数集

### 4.4 设计原则（可直接写入 Codex 工具 docs）

> **"Each file tool does one thing well. Complex queries (statistics, sorting,
> multi-stage filtering) belong in Bash. When a tool truncates output, it must
> tell the model which other tool to use next."**

---

## 五、参考来源

| 来源 | URL | 用途 |
| ---- | --- | ---- |
| Claude Code 架构解析（含源码） | https://www.cnblogs.com/bluestorm/p/19369058 | Glob 工具源码、截断引导设计 |
| Claude Code 配置及使用 | https://www.cnblogs.com/jaydenChu/p/19607755 | 工具列表（Read/Glob/Grep/LS/Bash 等） |
| Claude Code 官方文档 | https://docs.claude.com/en/docs/claude-code/settings | settings.json + 工具权限语法 |
| OpenCode 源码索引 | https://github.com/sst/opencode | `packages/opencode/src/tool/` 下的 `glob.ts` / `grep.ts` / `read.ts` |

---

## 六、变更日志

- **2026-06-10**：初始版本，基于实测对比 Codex / Claude Code / OpenCode 三方设计。
