# Identity and logo guidance

Use this reference when a task creates, changes, or invents visible product identity: logo, wordmark, monogram, app icon, favicon, nav/sidebar brand mark, or a placeholder identity for a product that has no verified asset.

The first rule is simple:

```text
no verified logo + task is not branding -> do not invent a fake logo to fill the slot
```

Use a plain text wordmark/product name, an existing verified asset, or leave the decorative mark out. An invented brand mark is not neutral filler; it changes the product's identity.

## Distinguish the artifact

Do not collapse these into one shape:

- **logo / brand mark** — identity that should stand on different surfaces;
- **wordmark** — typographic treatment of the product/brand name;
- **app icon** — OS-specific square icon that may be masked/rounded by the platform;
- **favicon** — tiny browser identity optimized for small sizes;
- **avatar fallback** — initials/letter tile representing a person/entity when no image exists.

A rounded-square container can be appropriate for an OS app icon or avatar fallback. That does not make it a good default logo.

## Generic identity-tile tell

Treat the following as a high-signal generated/default pattern when there is no existing brand reason:

```text
first letter / initials
+ ordinary system/default font
+ colored or gradient rounded square/circle
+ product name beside it
```

Common variants include:

- uppercase first letter centered in a rounded square;
- generic two-letter monogram inside a tinted tile;
- gradient/glossy app-tile shape used as the navbar logo;
- a random abstract spark/bolt/chat bubble/globe placed in a rounded square;
- the same icon-chip visual language used for both product logo and feature icons.

Do not "fix" this merely by changing the letter, color, gradient, or corner radius. The problem is that no product-specific identity decision was made.

## Existing identity wins

Before creating any mark:

1. inspect repository assets, manifest/PWA icons, favicons, brand docs, social preview, screenshots, and shipped header/sidebar;
2. prefer a verified existing logo/wordmark even if it is visually simple;
3. do not redraw or replace an existing brand because the agent prefers another style;
4. if multiple assets conflict, resolve which is current before changing identity.

Do not substitute a third-party logo or guessed brand asset.

## If actual logo design is requested

Logo work should not be a one-shot decorative completion.

First distill the product to a small set of identity inputs:

```text
product purpose / core idea
audience
name and language constraints
existing brand equity to preserve
where the mark must work: nav, favicon, app icon, print, dark/light, etc.
what competitors/default marks to avoid
```

If identity direction is weak or delegated, use the reference-research gate. Inspect a bounded set of relevant brand/design references and extract principles without copying another brand's trade dress.

Generate multiple meaningfully different concepts internally before selecting one. Variation should come from different ideas/constructions, not the same letter tile with different gradients.

A candidate should be able to answer:

- what product idea does this mark encode?
- why this geometry/letter treatment rather than a generic symbol?
- does it work without a decorative tile around it?
- is it recognizable in monochrome?
- does it remain legible at favicon/navigation size?
- does light/dark treatment preserve the same identity?
- could the same mark belong to five unrelated SaaS products? If yes, it is weak.

## Wordmarks before fake marks

When there is no real logo and the task only needs a header/navigation identity, a restrained wordmark is usually safer than inventing a synthetic icon.

Use typography that already belongs to the product's design system. Do not create brand theater through random letterspacing, gradient text, or a boxed initial.

## App icons are a separate problem

When the user explicitly requests an app icon, platform shape constraints may justify a square/rounded-square canvas. In that case, the underlying symbol still needs product meaning and small-size legibility.

Do not infer that because an app icon uses a rounded square, the in-app logo should also be the same boxed tile. Prefer a free-standing mark/wordmark where the UI context does not require the app-icon bezel.

## Verification

For material identity work, inspect at least:

- full wordmark/mark at normal header size;
- small mark around 16-24 px or actual favicon size;
- monochrome or single-color fallback;
- light and dark backgrounds;
- the mark without its optional app-icon/container bezel where applicable.

A logo that only works in one large showcase is not complete identity work.

## Source inspiration

This guidance is independently written, but the explicit distinction between a free-standing logo and an OS app-icon tile, the rejection of meaningless letter-in-a-tile output, and the generate-many/judge-before-shipping workflow were reinforced by the public MIT `ajjucoder/logo-maker` agent skill. See the companion `NOTICE.md` for source notes. No external runtime dependency is required.
