# Decision log

## DEC-001: Release Jev v2 separately from Jev v1 and chat runs

- **Date:** 2026-09-17
- **Status:** decided
- **Context:** Synthetic-only tuning changed missing-target routing, confidence gates, and relative-direction handling. Replaying the historic result through those rules would change a published result, while folding either Jev version into the chat release would violate the shared-prompt comparison contract.
- **Options considered:**
  - Recompute the v1 result under v2 rules, which would erase a reproducible historic measurement.
  - Merge Jev with chat results, which would present different prompt interfaces as a single comparison set.
  - Publish a separate v2 alias-and-pinned release while preserving v1 replay by specification hash.
- **Decision:** Publish v2 as its own release set with alias and pinned evidence, and dispatch replay by the recorded Jev specification hash. Keep the article's Jev points visibly separate from chat frontiers.
- **Consequences:** V1 remains 0.844 under its pinned mapping; v2 has independent evidence and CI. Future Jev adapter changes require another prompt hash and release rather than replacing either result.
