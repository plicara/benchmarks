# DEC-001: Keep lab metadata at the repository root

- **Date:** 2026-09-28
- **Status:** decided
- **Context:** `adventurebench/` carried its own `AGENTS.md`, `.plicara/README.md`, `.plicara/project.yaml` and `.agents/skills/README.md`. The `AGENTS.md` repeated the root file with one boundary line changed and the skills README was a placeholder. The lab now keeps one record per repository (foundation_lab DEC-003).
- **Decision:** Remove the folder copies. The root `project.yaml` stops listing child projects and declares the `adventurebench` Python environment that its folder record declared, and keeps its `published-snapshot-of` link to `adventurebench` under `related`. The rule that existed only in the folder copy, keeping claims traceable to local evidence, moves into the root `AGENTS.md`.
- **Consequences:** Adventure Bench no longer has its own lifecycle status or ID in metadata. The table below preserves the record as it stood before removal.

| Folder | ID | Name | Kind | Status | Description |
| --- | --- | --- | --- | --- | --- |
| `adventurebench` | adventurebench-release | Adventure Bench releases | publication | maintained | Published benchmark definitions and audited release records. |
