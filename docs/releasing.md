# Versioning and release decisions

Development proceeds on `main`. Accumulate user-visible changes under `Unreleased` in the changelog. A merge, passing CI run, website deployment or elapsed interval does not trigger a crate release. The crate manifest, lockfile entry, changelog, registry version and annotated `v<version>` tag identify the same candidate.

| Stage | Evidence for promotion |
| --- | --- |
| Alpha | A concrete feature slice works through its public entry point. List known limits, test the packaged default build in a downstream Cargo project and identify the specific evaluation need for another preview. |
| 0.1.0 | The proposed public surface has been reviewed against real audio use cases. Run a complete external training or inference example, compare signatures and behavior against the last preview, and record CPU/GPU evidence for the supported backend claims. Triage outstanding correctness and compatibility defects. |
| 1.0.0 | Document stable items, features, numerical contracts, checkpoint migration and the supported toolchain/hardware matrix. Exercise a real upgrade from a previous public release. Record the maintainer's explicit stability decision with the release notes. |

Use a patch for a compatible fix and a new minor line for an incompatible change before 1.0. A new version does not close unfinished design or correctness issues. For each user-visible break, explain the use case, alternatives, affected callers and migration. Prefer deprecation when both paths can coexist with a concrete removal condition. Keep experimental APIs and their limitations identifiable in the public documentation.

Review new public abstractions against a concrete downstream use. Record ownership, numerical semantics, backend behavior and a way to retire any superseded path. Promote an implementation only when it has a verified caller and evidence for the behavior it promises.

## CI responsibilities

On pull requests and pushes to main, GitHub Actions selects checks from changed paths. Rust changes run formatting, portable tests and doctests, and a check of all default targets at the declared minimum Rust version. Website changes run the locked Node install, TypeScript check and static build. Packaging changes also verify a Cargo source archive. Documentation changes can pass the shared CI gate without rebuilding the runtime.

The public runner verifies the portable backend. Run hardware-specific correctness or benchmark work on qualifying changes using the separate CPU and GPU environments in [the hardware protocol](colab-hardware-evaluation.md). Link the matching numerical, resource and performance evidence to the release decision. A portable CI result alone qualifies only the portable path.

CI checks that the manifest, lockfile and changelog agree. Once the current Cargo compatibility line has a stable release tag, CI compares the default-feature API against its newest merged stable tag. Preview-to-preview breaks and behavior changes require explicit review.

## Publishing a candidate

1. Review the changelog, compatibility impact and outstanding defects. Update the version once, in the release change. Check an external consumer and the [API compatibility review](api-stability.md#release-review).
2. Require passing CI for the candidate commit. Inspect `cargo package --locked --list`; run `cargo publish --dry-run --locked` on that checkout. Inspect the package's license, dependencies and vendor sources. Use `cargo semver-checks` against the applicable published baseline after the first release.
3. A maintainer publishes that exact commit with `cargo publish --locked`. Verify the crates.io checksum and install the exact version into a fresh consumer. Tag that commit and record the release notes. Deploy the static website independently.
4. If a published crate is defective, document the impact and fix it in a new candidate. Yanking is available when a published version should be removed from new dependency resolution. Registry versions are immutable.
