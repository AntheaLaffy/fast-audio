# Project website

| Task | Target | Also check |
| --- | --- | --- |
| Change the homepage or signal explorer | `Hero.tsx`, `AudioCanvas.tsx`, `StoryPillars.tsx`, `src/index.css` | Signal presets, motion preference and both viewport sizes |
| Add a guide, route or locale | `src/data/pages.ts`, `src/data/i18n.ts`, `src/App.tsx` | Search, guide navigation, alternate locale and historical inbound links |
| Update reference or measured data | Matching component and data module below | Published Rust API, operator contract or versioned measurement record |

Component filenames refer to `src/components/`. This table owns the website tree.

| Path | Purpose |
| --- | --- |
| `README.md` | Development, deployment, API index generation and portability checks |
| `index.html`, `src/main.tsx`, `src/App.tsx`, `src/vite-env.d.ts` | Metadata, startup, route content, accessible navigation and Vite asset types |
| `src/data/i18n.ts`, `src/data/pages.ts`, `src/data/release.ts` | Translations, locale-aware fragment paths, search catalog and project distribution links |
| `src/index.css` | Local fonts, responsive composition, reading and data views |
| `Navbar.tsx`, `Footer.tsx` | Search dialog, global navigation, locale control and project links |
| `Hero.tsx`, `AudioCanvas.tsx`, `StoryPillars.tsx` | Interactive signal surface, guide illustrations and measured example |
| `QuickStart.tsx` | Dependency and source build selectors for portable CPU, Intel and NVIDIA |
| `TutorialPage.tsx`, `src/data/tutorials.ts`, `src/examples/{first_tensor,spectral_roundtrip,train_gain}.rs` | Three localized lessons with downloadable, executable Rust examples |
| `CodeShowcase.tsx`, `CodeBlock.tsx`, `SyntaxHighlight.tsx`, `src/data/codeExamples.ts` | Searchable signatures, example selection, highlighting, clipboard and download |
| `scripts/index-api.py`, `src/data/api-index.json` | Generate and serve the default-feature public signature index from Rustdoc |
| `BenchmarkArena.tsx`, `src/data/benchmarks.ts` | Dated CPU/GPU comparisons, model controls, history, workload filters and CSV |
| `ArchitectureStack.tsx`, `CoverageMatrix.tsx`, `RoadmapSection.tsx`, `src/data/coverage.ts` | Execution layers, FP32 capabilities and release milestones |
| `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `public/.nojekyll` | Dependencies, type checks and deployable static output |

Dependencies, build output, TypeScript build state and checkout-local hosting metadata are excluded.
