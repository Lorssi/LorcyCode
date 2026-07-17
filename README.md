# Lorcy Code

Lorcy Code 是一个运行在终端中的 AI 编程助手。它基于 LangChain/LangGraph，支持多模型配置、持久化会话、文件与 Shell 工具、技能包、子 Agent、人工审批以及 Git 检查点。

## 功能概览

- 交互式终端对话，支持流式回答和推理内容展示。
- 读取、写入、编辑、搜索文件以及执行跨平台 Shell 命令。
- 使用 SQLite checkpointer 保存和恢复历史会话。
- 从用户目录或工作区发现并按需加载技能。
- 调用专用子 Agent 并控制工具权限与输出预算。
- 自动初始化 Git 仓库，使用检查点辅助消息回溯。
- Common 人工审批模式和 Yolo 自动批准模式。
- 主模型失败后按配置切换备用模型。

## 环境要求

- Python 3.14 或更高版本。
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理 Python 和依赖。
- Git 可选；安装后会启用工作区检查点功能。
- 至少配置一个 OpenAI 兼容模型服务的 API Key。

## 安装

```powershell
git clone <repository-url>
cd LorcyCode
uv sync
```

需要运行测试时安装开发依赖：

```powershell
uv sync --group dev
```

## 快速开始

首次运行交互模式：

```powershell
uv run lorcy_code
```

也可以通过 Python 模块启动：

```powershell
uv run python -m lorcy_code
```

首次运行且不存在模型配置时，程序会启动配置向导。也可以提前通过环境变量提供 API Key：

| 环境变量 | 默认服务 |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI |
| `BIGMODEL_API_KEY` | 智谱 GLM |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `DASHSCOPE_API_KEY` | 通义千问 |
| `ANTHROPIC_API_KEY` | Anthropic Claude |

启用自动批准模式：

```powershell
uv run lorcy_code --yolo
```

Yolo 模式会跳过工具执行前的人工确认，只应在可信工作区使用。

## CLI 命令

```text
lorcy_code [--yolo] [--version]
lorcy_code config [show|new|edit|switch]
lorcy_code gui
```

- `config show`：显示当前模型配置，API Key 会被掩码处理。
- `config new`：添加新的模型配置。
- `config edit`：编辑当前默认模型。
- `config switch`：在默认模型和备用模型之间切换。
- `gui`：保留的兼容命令；当前版本尚未提供 GUI。

## 交互命令

进入对话后可使用以下斜杠命令：

| 命令 | 作用 |
| --- | --- |
| `/new` | 创建新会话 |
| `/history` | 查看并载入历史会话 |
| `/model` | 新建、编辑或切换模型 |
| `/messages` | 编辑、分叉或删除历史消息 |
| `/compress` | 压缩当前会话上下文 |
| `/skill` | 选择、安装、查看或删除技能 |
| `/mode` | 在 Common 和 Yolo 模式间切换 |
| `/git` | 查看 Git 和检查点状态 |
| `/workdir` | 切换当前工作目录 |
| `/tools` | 显示内置与 MCP 工具 |
| `/mcp` | 添加、删除、启停、测试和刷新 MCP 服务 |
| `/help` | 显示帮助 |
| `/quit` | 退出程序 |

## 内置工具

Agent 可使用以下工具：

- `bash`：在持续维护工作目录的 Shell 会话中执行命令。
- `read_file`、`write_file`、`edit`：读取、创建和精确修改文件。
- `glob`、`grep`、`list_dir`：查找文件、搜索内容和浏览目录。
- `todo_write`：保存当前会话的结构化任务列表。
- `agent`：启动具有限定工具集的子 Agent。
- `load_skill`：按需读取已启用技能的完整说明。

Shell 输出和工具结果会自动截断并保存过长内容，避免占满模型上下文。

## MCP 工具

LorcyCode 支持本地 `stdio` 和远程 `Streamable HTTP` MCP 服务，并兼容 Claude
等客户端使用的 `mcpServers` JSON。第三方提供的配置可以直接复制到
`~/.lorcy/mcp.json` 或项目的 `.lorcy/mcp.json`。执行 `/mcp`
进入交互管理，也可以使用 `/mcp list`、`/mcp add`、`/mcp enable <name>`、
`/mcp disable <name>`、`/mcp test <name>` 和 `/mcp tools <name>`。

用户级配置位于 `~/.lorcy/mcp.json`，项目级配置位于 `.lorcy/mcp.json`；同名时
项目配置完整覆盖用户配置。远程凭据必须通过 `${env:NAME}` 引用，例如：

```json
{
  "mcpServers": {
    "cloud": {
      "type": "http",
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer ${env:CLOUD_MCP_TOKEN}"}
    }
  }
}
```

带 `command` 的服务会自动识别为 `stdio`，无需额外填写 `transport`。LorcyCode
仍兼容早期的 `version + servers` 格式，但新配置默认保存为通用格式。明文 `env`
可以读取以兼容第三方示例，不过会显示安全提示；建议将真实密钥替换为
`${env:NAME}`。

项目级 `stdio` 配置首次运行以及可执行配置发生变化时会要求确认信任。

### 示例：注册计算器 MCP Server

仓库自带一个使用官方 `FastMCP` 编写的计算器服务：
`examples/mcp_servers/calculator_server.py`。在 Windows 上可将以下配置加入
`~/.lorcy/mcp.json` 或项目的 `.lorcy/mcp.json`：

```json
{
  "mcpServers": {
    "calculator": {
      "command": "D:/CodeProject/LorcyCode/.venv/Scripts/python.exe",
      "args": [
        "D:/CodeProject/LorcyCode/examples/mcp_servers/calculator_server.py"
      ]
    }
  }
}
```

也可以通过 `/mcp add calculator` 添加：传输选择 `stdio`，启动命令填写上面的
Python 路径，命令参数填写计算器脚本路径。保存后执行 `/mcp tools calculator`
即可看到 `add`、`subtract`、`multiply`、`divide`、`power` 和 `square_root`。

## 配置与数据目录

用户级配置位于 `~/.lorcy`：

```text
~/.lorcy/
├── model.json          # 默认模型和备用模型配置
├── mcp.json            # 用户级 MCP 服务
└── lorcyagent.json     # 最近使用的工作目录等设置
```

每个工作区的数据位于 `<workspace>/.lorcy`：

```text
.lorcy/
├── sessions/           # SQLite 检查点、会话名称和历史数据
├── skills/             # 当前工作区安装的技能
├── mcp.json             # 项目级 MCP 服务
└── skill_selection.json
```

用户级技能可放在 `~/.lorcy/skills`。工作区技能优先于用户级技能。API Key 会以明文保存在用户自己的 `model.json` 中，请妥善保护该文件。

## 项目目录结构

```text
lorcy_code/
├── __init__.py
├── __main__.py
├── agents/
├── cli/
├── config/
├── integrations/
├── mcp/
├── sessions/
├── shared/
├── skills/
└── tools/
    └── shell/
tests/
```

### 顶层与 CLI

| 文件 | 作用 |
| --- | --- |
| `lorcy_code/__init__.py` | 包说明和版本号 |
| `lorcy_code/__main__.py` | `python -m lorcy_code` 入口 |
| `cli/app.py` | Typer CLI、模型配置命令和 REPL 启动 |
| `cli/repl.py` | 交互会话生命周期、消息流、审批与命令处理器 |
| `cli/commands.py` | 斜杠命令解析和路由 |
| `cli/input.py` | prompt-toolkit 历史、补全和提示文本转换 |
| `cli/display.py` | Rich 渲染、进度状态和上下文用量展示 |
| `cli/prompts.py` | 异步选择框、确认框和模型配置表单 |

### Agent

| 文件 | 作用 |
| --- | --- |
| `agents/builder.py` | 组装主 Agent、中间件、工具和 checkpointer |
| `agents/context.py` | Agent 运行上下文、上下文窗口和 SQLite checkpointer |
| `agents/errors.py` | 跨 Agent/CLI 共用的控制流异常 |
| `agents/model.py` | OpenAI 兼容模型适配和流式响应增强 |
| `agents/retry.py` | 备用模型选择和重试状态管理 |
| `agents/middleware.py` | 模型加载、消息修正、审批、重试和预算中间件 |
| `agents/definitions.py` | 子 Agent 定义及内置 Agent 描述 |
| `agents/loader.py` | 从 Markdown/frontmatter 加载 Agent 定义 |
| `agents/subagents.py` | 子 Agent 构建、权限约束和执行流程 |

### 配置、会话和集成

| 文件 | 作用 |
| --- | --- |
| `config/paths.py` | 用户级与工作区配置路径计算 |
| `config/storage.py` | 模型、工作区和技能选择配置持久化 |
| `config/models.py` | 模型发现、配置、编辑、切换和连接测试 |
| `sessions/manager.py` | 会话 ID、名称、摘要、列表和删除操作 |
| `integrations/git.py` | Git 可用性检测、初始化、检查点和回滚 |

### 技能和工具

| 文件 | 作用 |
| --- | --- |
| `skills/loader.py` | 技能扫描、缓存、解析、安全校验与安装 |
| `skills/manager.py` | 技能列表、启用选择和交互式管理 |
| `tools/registry.py` | 内置 LangChain 工具定义和 `ALL_TOOLS` 注册表 |
| `tools/config.py` | grep、Todo 等工具使用的常量 |
| `tools/result_pipeline.py` | 工具输出清理、持久化、截断和轮次预算 |
| `tools/shell/provider.py` | Bash 与 PowerShell 提供器抽象 |
| `tools/shell/session.py` | Shell 子进程执行、目录跟踪和超时处理 |
| `tools/shell/output.py` | Shell 输出截断与临时文件持久化 |
| `tools/shell/result.py` | Shell 执行结果数据结构 |
| `tools/shell/semantics.py` | 对 grep、diff 等特殊退出码进行语义解释 |

### 通用模块

| 文件 | 作用 |
| --- | --- |
| `shared/json.py` | JSON 原子写入、mtime 缓存和默认配置构造 |
| `shared/frontmatter.py` | YAML frontmatter 解析 |
| `shared/text.py` | 多模态文本提取和 API Key 掩码 |
| `shared/messages.py` | 历史消息分组及消息 ID 收集 |

## 技能包格式

技能目录至少包含一个带 YAML frontmatter 的 `SKILL.md`：

```markdown
---
name: example-skill
description: 示例技能
---

这里写提供给 Agent 的技能说明。
```

支持安装 `.zip`、`.tar.gz`、`.tgz` 和 `.tar.bz2`。安装器会拒绝路径穿越、压缩包链接以及不安全的技能名称。

## 开发与验证

```powershell
uv sync --group dev
uv run pytest
uv run python -m compileall -q lorcy_code tests
uv run python -m lorcy_code --help
uv run lorcy_code --help
```

测试覆盖 CLI 入口和 Yolo 参数传递、配置路径和原子写入、模型切换异常、技能包安全校验、文本/frontmatter 以及 Shell 输出语义。

## 当前限制

- GUI 命令仅为兼容旧入口，尚未实现图形界面。
- 模型服务需兼容当前使用的 OpenAI/LangChain 调用协议。
- Yolo 模式会自动批准潜在的文件与 Shell 操作，应谨慎使用。
