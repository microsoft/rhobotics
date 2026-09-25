# rhobotics project website

The source for <https://microsoft.github.io/rhobotics/>, built with
[Astro](https://astro.build). Everything in this folder is website-only; it is
independent of the Python package in the rest of the repo.

## Prerequisites

Node.js 18+ (20 LTS recommended), which includes `npm`.

> On WSL, `npm` may resolve to the *Windows* install
> (`/mnt/c/Program Files/nodejs/npm`) while `node` is not on the Linux `PATH`,
> which breaks local builds. Install Node inside WSL (e.g. via
> [nvm](https://github.com/nvm-sh/nvm)) or run the commands below from
> PowerShell instead.

## Local preview

```bash
cd website
npm install      # one-time: install dependencies
npm run dev      # dev server with hot reload
```

Open the URL it prints — by default <http://localhost:4321/rhobotics> (the
`/rhobotics` prefix comes from `base` in `astro.config.mjs`).

```bash
npm run build    # production build into ./dist
npm run preview  # serve the production build locally
```

## Adding content

| What you want to do | Where |
| --- | --- |
| Edit the landing page (title, authors, sections, BibTeX) | `src/pages/index.astro` |
| Add another page, e.g. `/rhobotics/results` | create `src/pages/results.astro` |
| Add a blog-style page from Markdown | create `src/pages/whatever.md` with `layout: ../layouts/BaseLayout.astro` in its frontmatter |
| Change the top banner / wordmark | `src/components/Ribbon.astro` |
| Change footer links | `src/components/Footer.astro` |
| Change the HTML shell (meta tags, fonts) | `src/layouts/BaseLayout.astro` |
| Change colors, fonts, spacing | `src/styles/global.css` (CSS variables at the top) |
| Add images, videos, PDFs | drop them in `public/` |

Every `.astro` or `.md` file under `src/pages/` becomes a route: `index.astro`
→ `/rhobotics/`, `results.astro` → `/rhobotics/results`,
`papers/vla.astro` → `/rhobotics/papers/vla`.

### Linking to assets and other pages

Because the site is served under the `/rhobotics/` prefix, never hard-code a
leading-slash path. Prefix with the base:

```astro
---
import { withBase } from "../lib/url";
---
<img src={withBase("teaser.png")} alt="Teaser" />
<a href={withBase("results")}>Results</a>
```

(`public/teaser.png` is served at `/rhobotics/teaser.png`.) `withBase` lives in
`src/lib/url.ts` and handles the base prefix with or without a trailing slash;
use it instead of `import.meta.env.BASE_URL` directly.

## Publishing

The site is published at <https://microsoft.github.io/rhobotics/>.
`.github/workflows/deploy-website.yml` is currently gated to
`workflow_dispatch` only, so pushes do not deploy automatically.

To deploy an update:

1. Open the **Actions** tab → "Deploy website to GitHub Pages" → **Run
   workflow**, and select the `main` branch. Do not use **Re-run jobs** on an
   older deployment because that rebuilds the original commit.
2. Check the result at
   <https://microsoft.github.io/rhobotics/>.
3. To deploy on every website change, uncomment the `push` trigger at the top of
   `.github/workflows/deploy-website.yml`. Pushes to `main` touching
   `website/**` will publish automatically from then on.

The repo must be public (or have GitHub Pages enabled for private repos on the
org's plan) for `https://microsoft.github.io/rhobotics/` to be reachable.
