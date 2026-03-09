Use `.github/prompts/project_context.md` as the main project context.
Follow `.github/prompts/copilot_instructions.md`.
Use `docs/copilot_workplan.md` to understand the implementation order and long-term direction.

We are working on **Phase 1: Refactor training and config system**.
The repository is an existing scientific PoC that must be refactored incrementally, not rewritten from scratch.
Prefer simple, coherent, local-first design choices consistent with the project context.

## Task
Refactor the current training entrypoint and related logic into a more modular, config-driven structure.

The goal of this task is to establish the foundation for the later MLOps phases by introducing:

- reusable training logic under `src/`
- a clear config-driven execution path
- a minimal run context abstraction
- a CLI flow that remains easy to run locally
- minimal disruption to existing behavior where possible

You should inspect the current repository structure and propose the smallest coherent refactor that improves modularity and prepares the codebase for later additions such as tracking, artifacts, reports, and sweeps.

## What to do

Please:

1. Identify the current training/evaluation entrypoints and the main training flow.
2. Refactor training-related logic into reusable modules under `src/` where appropriate.
3. Introduce or clean up a config-driven execution pattern using the existing config system, extending it only if needed.
4. Add a lightweight run context object or equivalent mechanism to capture core run metadata such as:
   - run id
   - timestamp
   - config snapshot or config path
   - dataset identifier if available
   - split identifier if available
   - git commit hash when easily accessible
5. Keep execution local and simple.
6. Preserve existing scientific behavior as much as possible.
7. Add or update a CLI script for training so that the refactored flow is runnable end-to-end.
8. If needed, leave explicit TODOs for pieces that belong to later phases rather than overengineering them now.

## Autonomy and reasoning scope

You should make reasonable design decisions autonomously.
Do not wait for further clarification on small architectural choices if a coherent option is available.

However:

- do not rewrite the repository from scratch
- do not introduce heavy frameworks or orchestration
- do not add speculative abstractions with unclear immediate value
- do not implement full tracking, registry, or sweep features yet unless a tiny stub is needed to support the refactor

Prefer thin abstractions and minimal file churn.

## Constraints...
- Keep the repo lightweight and demo-friendly.
- Execution must remain local-first.
...
- Wandb integration is not the focus of this task; if referenced, it should only appear as a future integration point.
- Separate scientific logic from MLOps plumbing.
- Prefer modifying the smallest sensible number of files.
- Avoid breaking the current runnable workflow.
- If the current codebase has inconsistencies, choose the most practical path and document it briefly.

## Expected repository changes

At minimum, aim to produce:

- a reusable training module under `src/...`
- a clearer config-driven training path
- a lightweight run context abstraction
- one runnable training CLI script
- small updates to configs if necessary

Possible outputs may include files such as:

- `src/conformal_predictions/training/trainer.py`
- `src/conformal_predictions/mlops/run_context.py`
- `scripts/run_train.py`
- `configs/...`

Adapt this to the actual repository structure rather than forcing these exact paths.

## Deliverable format

Please provide:

1. A short plan before coding.
2. The concrete file changes.
3. A brief explanation of the chosen design.
4. Any TODOs intentionally deferred to later phases.

## Acceptance criteria

The task is successful if, after the refactor:

- training logic is less script-centric and more reusable
- training can be launched through a config-driven CLI flow
- a minimal run context exists and is populated during execution
- the implementation remains coherent with the project context
- the code stays local-first, lightweight, and easy to explain in a demo
- previously existing behavior is preserved as much as reasonably possible
- deferred pieces are clearly marked instead of being overdesigned

## Important implementation attitude

Favor a clean and pragmatic refactor over completeness.
When multiple reasonable designs are possible, choose the simplest one that unlocks later phases cleanly.
You may reorganize modules and scripts if needed, as long as the resulting structure is simpler, more reusable, and still easy to run locally.