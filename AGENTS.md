Project documentation is written in English. `target.md` records the user's long-term
objective; `docs/roadmap.md` tracks implementation and verification milestones.

## CODEMAP Navigation Protocol

Mode: maintenance. Start navigation at `CODEMAP.md`.

- Match Task Guide first; Target is the initial read set. Also Check requires task relevance,
  evidence from target code, or public-contract impact.
- Without a match, descend through matching Domain routes and local Task Guides. Expand
  maintenance reads for a concrete reason. Verify stale paths and ranges against source.
- Read imports for a needed interface contract and consumers for public signature, return,
  or semantics changes. Batch independent targeted reads; filter large result sets first.
- For files over 1000 lines, use companion `.analysis.md` Feature Index ranges, with
  Logical Sections as fallback.
- Evaluate additions, deletions, and modifications. Update affected canonical entries,
  parent routes, exports and ranges. Every included file/directory has one local owner;
  excluded/boundary-only interiors follow the root Scope and Exclusions.
- Retire superseded paths after checking real consumers and replacements. Retain
  compatibility only for a current contract with a checkable removal condition.
- Keep verification evidence separate from navigation/maintenance fingerprints. Report
  incomplete verification explicitly. Learning mode keeps source read-only.

## Colab CLI and Hardware Isolation Protocol

- Use the `colab` CLI (`colab new`, `colab drivemount`, `colab ssh -s <name> -- <cmd>`,
  `colab upload`, `colab download`, `colab stop`) for cloud execution.
- Enforce strict instance separation to conserve Compute Units: rent a CPU instance
  (`colab new -s fa-cpu [--high-mem]`) for CPU builds, tests, benchmarks, and Drive cache
  preparation; rent a GPU instance (`colab new -s fa-gpu --gpu <T4|L4|A100|G4>`) strictly
  for GPU builds, GPU tests, and GPU benchmarks. Never run CPU workflows on a GPU instance.
- Mount Google Drive (`/content/drive`) and maintain a persistent copy of the source
  archive, vendored Cargo crates, and cached toolchain under
  `/content/drive/MyDrive/fast-audio-colab` to accelerate cross-session I/O, while
  compiling and running hot workloads on local ephemeral disk (`/content/fast-audio`).
  Stop active sessions (`colab stop -s <name>`) immediately after syncing artifacts.
  See `docs/colab-hardware-evaluation.md`.
