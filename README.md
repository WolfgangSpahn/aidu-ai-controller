# AIDu AI Controller

`aidu.ai.controller` is the orchestration and runtime layer of the AIDu ecosystem.

The controller owns the execution loop and coordinates interactions between agents.

Agents do **not** call each other directly. Instead, agents:

* consume artifacts,
* produce artifacts,
* emit recommendations.

The controller:

* executes agents,
* evaluates recommendations,
* schedules future executions,
* manages runtime state,
* decides when execution stops.

This architecture separates local reasoning inside agents from global control and planning.

---

# Execution Model

Execution is artifact-driven.

```text
Artifact
    ↓
 Agent
    ↓
Artifacts + Recommendations
    ↓
Controller
    ↓
Next Agent
```

A recommendation proposes a possible next action.

The controller applies a selection policy (currently highest utility) and converts the selected recommendation into a future execution event.

This enables:

* iterative reasoning,
* planning through recommendation chains,
* future integration of costs, risks, uncertainty, and learning objectives.

---

# Example

```text
SymbolicArtifact(0)
        ↓
    DummyAgent
        ↓
SymbolicArtifact(1)
        ↓
  Recommendation
        ↓
    Controller
        ↓
    DummyAgent
        ↓
SymbolicArtifact(2)
```

The controller continues execution until:

* no recommendation is produced, or
* a stop event is generated, or
* a safety limit is reached.

---

# Development

## Install Local Dependencies

The controller currently depends on a local editable installation of:

```text
aidu-ai-llm
```

via:

```toml
[tool.uv.sources]
aidu-ai-llm = { path = "../aidu-ai-llm", editable = true }
```

---

## Run Interactive Runtime

```bash
python -m aidu.ai.controller.main
```

---

## Run Smoke Tests

```bash
python -m aidu.ai.controller.controller
```

The included smoke tests verify:

* recommendation selection,
* controller-managed execution loops,
* artifact propagation across agents.

---

# Design Goals

The controller is intentionally minimal.

Future versions may support:

* richer event types,
* multiple artifact streams,
* uncertainty-aware recommendation selection,
* agent marketplaces,
* curriculum-aware planning,
* belief and student models,
* distributed execution.

---

# License

MIT License.

Copyright (c) 2026 Wolfgang Spahn, PHBern.

Please follow standard academic practice when using this software in research or publications.
