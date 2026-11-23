# fast-audio developer website

The [live site](https://anthealaffy.github.io/fast-audio/) provides English, Simplified Chinese and Japanese installation guides, tutorials, an API index and dated benchmarks. Its React, TypeScript and Vite source lives here. Fonts, the Three.js signal projection and all data are bundled.

## Development

Use Node.js 22 or later. From the website directory run:

```sh
npm ci
npm run dev
```

## Static deployment

```sh
npm run build
npm run preview
```

Publish the **contents** of `dist/` on any static host. Vite emits relative asset URLs. Routes and locale are in the fragment, so a direct link such as `#/ja/tutorials/spectrum` works at the domain root or any nested path with normal directory index serving. No rewrite or external font service is required.

The current GitHub Pages deployment serves the root of the `gh-pages` branch. Source development and CI use `main`. Website deployments are independent of the crate release cycle. Preserve the existing branch and any host-owned files when replacing the static output.

## Updating reference data

Generate the default-backend API index from the repository root:

```sh
cargo doc --locked --no-deps --lib
python3 website/scripts/index-api.py target/doc/fast_audio
```

Tutorials render the downloadable Rust files in `src/examples/` directly. Run these as examples of the framework when their code changes. The benchmark explorer identifies each historical suite, backend, comparator and unit. Keep measured values paired with their original execution modes.
