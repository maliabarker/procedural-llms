# EvoProc Procedures

Domain-specific **procedures, schemas, prompts, runners, and backends** built on top of
[`evoproc`](https://pypi.org/project/llm-procedure-generation-ga/).  
Use this package to define **Pydantic procedure models**, strict **answer schemas**, reusable **prompt builders**, a **stateful runner**, and an **Ollama** query backend for experiments (e.g., GSM8K).

---

## ✨ What’s inside

- **Models** – Pydantic types for `Procedure`, `Step`, IO fields.
- **Schemas** – JSON Schemas for final answers (e.g., GSM8K numeric, ARC multiple-choice).
- **Prompts** – Builders for procedure creation and per-step execution.
- **Runners** – Deterministic, global-state step executor with strict JSON I/O.
- **Query backends** – `ollama` adapter (`query`, `hard_query`) with seed & format support.
- **Builders (optional)** – Convenience orchestration (create → repair → run).

> This package *depends on* the core GA: `evoproc`.

---

## 📦 Install

From a mono-repo layout:

```bash
# Core GA (if not already installed)
pip install -e projects/core

# This plugin package
pip install -e projects/procedures
```
