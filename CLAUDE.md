# DRIFTLESS — working context

AI-ML based Intelligent Dead Reckoning (IDR) with GNSS fusion, for
**Smart India Hackathon, PS #26168 (ISRO)**. Team project.

## The bet

Most Indian vehicles have no factory INS — navigation runs off a phone in a
dashboard mount, and GNSS drops entirely in tunnels, parking structures, and
urban canyons. DRIFTLESS turns a bare smartphone IMU into a self-contained
dead-reckoning system that holds lane-level position through a GNSS blackout
and fuses back onto GNSS the instant it returns, using the road network
itself (map-matching + non-holonomic constraints) to keep drift bounded
rather than trusting the IMU integration alone.

## Constraints

- Deadline for idea submission: **2026-09-20**. SIH finale is months out —
  this is a long-cycle project, not a weekend hackathon.
- Screening requires preliminary AI models + position-plot results
  inferenced from a subset of the **IO-VNBD** dataset
  (github.com/onyekpeu/IO-VNBD), submitted as part of the proposal.
- Deliverable is two-sided: an on-device mobile app (phone IMU + GNSS,
  10Hz fusion) and a separate edge-deployable engine for external/FOG IMU
  sensors (~200Hz), trained offline and exported for on-device inference.
- Performance bar: <10% positional drift over distance in a GNSS blackout
  (e.g. <100m drift over 1km at 60kmph); 10Hz position update on phone.
- Judged/defended in front of a panel — architecture and modeling choices
  (why this filter, why this fusion approach) need to be defensible live.

## Stack

Undecided as of scaffolding (2026-08-29) — stack pick (mobile framework,
training framework, map-matching approach) is a nontrivial choice deferred
to the first real working session, not decided here.

Decision log convention and working defaults are in `~/.claude/CLAUDE.md` —
this file holds only what's specific to this project.
