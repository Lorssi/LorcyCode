---
name: risk-alert-experience-retrieval
description: Retrieve historical design, failure, and evaluation experience for PRRS abortion abnormal risk alert model optimization. Use when Codex needs to search prior best designs, similar metric patterns, or previous failed attempts before proposing the next model iteration.
---

# Risk Alert Experience Retrieval

## Purpose

Use this skill to reuse previous optimization experience without forcing the current iteration to restart from a naive baseline.

## Procedure

1. Build a query from the current design plan, evaluation diagnosis, or failure message.
2. Run `scripts/retrieve_experience.py` when a vector store is available.
3. Read `references/retrieval-policy.md` before applying retrieved content.
4. Return useful lessons as guidance, not as a full replacement for current reasoning.

## Script Usage

Retrieve similar historical experience:

```powershell
python D:\Code\毕业论文\工作点2\risk-alert-experience-retrieval\scripts\retrieve_experience.py `
  --vectorstore D:\Code\毕业论文\Multi-Agent\data\external\vectorstore\designer_agent_output_vectorstore `
  --query "mask-aware multi-task model has high AUC but low F1 on 15-21 day horizon" `
  --top-k 3
```

Optional model filter:

```powershell
python D:\Code\毕业论文\工作点2\risk-alert-experience-retrieval\scripts\retrieve_experience.py `
  --vectorstore D:\Code\毕业论文\Multi-Agent\data\external\vectorstore\designer_agent_output_vectorstore `
  --query "initial architecture for PRRS abortion alert" `
  --model-filter qwen-plus
```

Inspect vector store contents:

```powershell
python D:\Code\毕业论文\工作点2\risk-alert-experience-retrieval\scripts\inspect_vectorstore.py `
  --vectorstore D:\Code\毕业论文\Multi-Agent\data\external\vectorstore\designer_agent_output_vectorstore
```

Environment requirement:

- `DASHSCOPE-API-KEY` must be available for `DashScopeEmbeddings`.

If retrieval dependencies or API key are unavailable, continue without retrieval and state that no historical context was loaded.

## Output Contract

Return:

- `matched_experience`: retrieved historical notes or a statement that none were found.
- `applicable_points`: what can be reused in the current iteration.
- `risks`: what should not be copied blindly.
- `query_used`: the retrieval query.