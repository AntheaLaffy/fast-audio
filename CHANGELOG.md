# Changelog

## Unreleased

Package the integration-test support modules and check every target from the packaged sources in CI.

## 0.1.0-alpha.1

First public preview of the Rust audio framework. The portable CPU backend, checked tensor layouts, prepared spectral transforms, reverse-mode tape and optimizer APIs are available for evaluation. Intel and NVIDIA execution require the corresponding vendor SDKs. See the [operator matrix](docs/operator-coverage.md) for qualified FP32 capabilities.

The minimum Rust version is 1.87. This release establishes the initial API comparison baseline. Public compatibility commitments, experimental features and release criteria are described in the [stability policy](docs/api-stability.md) and [release process](docs/releasing.md).

The published alpha.1 archive omits the integration tests' support modules. Library consumers and examples work normally. Run the complete suite from the `v0.1.0-alpha.1` Git tag.
