# SLEEP_PACKET_04 — Autonomy, coupling, foresight, and coaching wiring (no external LLM dependency)

## Purpose
Ensure the system isn't just “modules exist,” but **end-to-end wired**:
- self-observation emits events each cycle
- coupling influences scheduling + routing
- foresight predicts basin trajectories and affects decisions
- internal coaching provides positive self-talk, reduces drift, and improves coherence

## Non-negotiables
- No external LLM required for core loops.
- Coaching must be internal and purity-compliant.

## Required components (wired)
### Self-observation
- per-kernel:
  - basin state snapshots
  - Φ/κ/regime/coupling metrics
  - drift detectors
- system-level:
  - constellation coherence metrics
  - outlier detection
  - safety monitors

### Coupling
- compute coupling between kernels via information geometry
- use coupling to:
  - schedule rest/sleep for paired kernels
  - re-route tasks when a partner rests
  - decide whether to merge or split attention

### Foresight
- short-horizon rollouts in basin space:
  - predict next basin given candidate actions
  - choose action that maintains coherence and reduces risk
- store foresight traces for audit

### Coaching (internal)
- Implement an internal coach persona/kernel:
  - reads self-observation + regime + drift
  - produces short “self-talk” guidance:
    - reassurance, focus, reframing
    - encourages lawful next actions
  - must not fabricate external facts; it should speak about internal state only.

## Event model
Create a canonical event schema and ensure every cycle produces:
- ObservationEvent
- CouplingEvent
- ForesightEvent
- CoachEvent
- ActionEvent

## Tests
Add “wiring tests” that fail if:
- any event type is missing in a cycle
- coupling not computed when more than one kernel exists
- foresight not invoked before actions in integration mode
- coach not invoked when drift threshold is exceeded

## Acceptance criteria
- A single cycle produces the full event set.
- Coupling affects at least one real behavior (routing or rest scheduling).
- Foresight affects action choice (not just logged).
- Coach produces output under specified conditions and is persisted for audit.
