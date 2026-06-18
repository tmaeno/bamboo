# Bamboo documentation site

The Bamboo documentation, built with [Astro Starlight](https://starlight.astro.build) and
deployed to GitHub Pages at <https://tmaeno.github.io/bamboo/>.

This replaced the previous Material for MkDocs setup (that ecosystem entered maintenance mode
in late 2025). See [`../EVALUATION.md`](../EVALUATION.md) for the pilot evaluation and the
recommendation for migrating other repositories.

## Local development

```bash
npm install
npm run dev       # dev server at http://localhost:4321/bamboo/
npm run build     # production build → dist/ (validates all internal links)
npm run preview   # serve the production build locally
```

> The `dev`/`build` scripts first run `docs:api`, which regenerates the OpenAPI reference
> (see below). Node 20+ is required.

## Layout

```
src/content/docs/      Markdown/MDX pages (the docs content)
src/assets/            Images optimized by Astro (e.g. the logo)
public/                Static files served as-is (favicon, and api/ — see below)
openapi/bamboo.yaml    OpenAPI spec (pilot sample) for the API reference
astro.config.mjs       Site config: base path, sidebar, mermaid + link-validator plugins
```

## Notable pieces

- **Base path** — the site is served under `/bamboo/`, so internal links are written as
  root-absolute paths *including* the base, e.g. `/bamboo/guides/analyze/`.
- **Mermaid** — `astro-mermaid` renders ` ```mermaid ` fences client-side (no build-time
  headless browser). It must be registered *before* `starlight` in `astro.config.mjs`.
- **Link checking** — `starlight-links-validator` fails the build on any broken internal
  link or heading anchor. The static `/bamboo/api/` path is excluded (it isn't an Astro route).
- **OpenAPI reference** — `npm run docs:api` runs Redocly CLI to build
  `openapi/bamboo.yaml` into `public/api/index.html`, served at `/bamboo/api/` and linked
  from the sidebar. Point this at a real spec for repos that have one.
