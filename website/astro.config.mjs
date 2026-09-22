// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  // === GitHub Pages deployment ===
  // Served as a project page at https://microsoft.github.io/rhobotics/
  // `site` is the origin only; `base` is the repo-name path prefix.
  site: 'https://microsoft.github.io',
  base: '/rhobotics',
  trailingSlash: 'ignore',
});
