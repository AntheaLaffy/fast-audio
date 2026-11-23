# API stability

fast-audio follows Cargo's compatibility conventions. The first publication is a preview, `0.1.0-alpha.1`. An exact requirement of `=0.1.0-alpha.1` keeps an experiment reproducible. A plain `"0.1"` requirement starts accepting releases at 0.1.0. The [release process](releasing.md) defines promotion gates.

| Release | Compatibility commitment |
| --- | --- |
| 0.1.0-alpha.N | Public previews for concrete evaluation milestones. Changes to preview APIs require an explicit compatibility note and a migration example when they affect callers. |
| 0.1.x | Preserve documented public Rust items, existing Cargo features, supported signatures, the declared minimum Rust version, and observable contracts for supported inputs. Additive functionality must preserve existing callers. |
| 0.2.0 and later 0.y.0 releases | May introduce incompatible changes. Identify each affected symbol or behavior in the release notes and show the replacement or migration. Deprecate in an earlier compatible release when practical. |
| 1.0 and later | Announce the stable surface and supported backends at 1.0. Incompatible changes to that surface then require a major release. |

Documented tensor layouts, output shapes, ownership rules and promised numerical behavior are part of the contract. Changes to internal scheduling, kernel selection or measured latency are implementation changes unless they break a documented guarantee. Document feature-gated APIs and hardware prerequisites with their examples.

Treat serialized checkpoints separately. Record their format version and supported read paths. A change that makes a supported checkpoint unreadable requires either a migration or a compatibility-breaking release. The crate version describes the Rust package and does not replace the format version.

## Release review

Compare each candidate with the latest published release in the same compatibility line using `cargo semver-checks check-release --baseline-version <published-version>` on the declared feature sets and available hardware. Also build a downstream example from the package archive and exercise any changed numerical or checkpoint contracts. Record breaking changes and migrations before publishing. The first release establishes the comparison baseline.

Before promising 1.0, review the public tensor and training APIs, Cargo features, supported backend policies, checkpoint upgrade story and downstream usage. Announce the scope alongside the release.
