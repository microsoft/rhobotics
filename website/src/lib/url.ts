// The site is served under a path prefix (`base` in astro.config.mjs), and
// `import.meta.env.BASE_URL` may or may not carry a trailing slash. Use this
// helper for every internal link and asset URL so both cases work.
const BASE = import.meta.env.BASE_URL.replace(/\/+$/, "");

/** withBase("teaser.png") -> "/rhobotics/teaser.png"; withBase() -> "/rhobotics/" */
export function withBase(path = ""): string {
  const clean = path.replace(/^\/+/, "");
  return `${BASE}/${clean}`;
}
