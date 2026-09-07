# Design discovery contract

## Goal

Before changing a substantial user-facing web surface, determine what visual/product system already exists and what the task actually requires. Discovery prevents a coding agent from replacing a coherent product language with a generic template.

Repository discovery answers **what already exists**. When high-impact intent remains ambiguous after that scan, continue with [adaptive-design-discovery.md](adaptive-design-discovery.md) to decide whether clarification is worthwhile and how many questions to ask.

The agent's first visual idea is not evidence that the design is good. When the task contains a consequential design choice and local/user evidence is too weak to choose confidently, continue with [reference-research-and-critique.md](reference-research-and-critique.md) before committing to a direction.

## Evidence priority

Use this order:

1. explicit user request and supplied references;
2. repository-owned design/brand/product guidance;
3. design tokens, themes, component primitives, icons, typography, assets;
4. repeated patterns in shipped/current UI;
5. relevant external design/reference evidence when the first four do not resolve a consequential decision;
6. this companion's fallback guidance.

A screenshot, mockup, or accepted live page supplied by the user is strong acceptance evidence for the scope it shows. Do not infer hidden interaction or inaccessible states from a static image alone.

## Compact repository scan

Look only where relevant. Typical signals:

```text
DESIGN.md / DESIGN_SYSTEM.md / brand docs
README / product specs / ADRs
Storybook / component docs
src/components / ui / design-system
CSS custom properties / theme files
Tailwind config / token packages
font imports and type scale
icon packages and local SVG conventions
layout shell / navigation / modal / form primitives
existing visual tests or screenshots
```

Do not spend a long turn exhaustively searching every stylesheet. Stop once enough evidence exists to preserve the product's system.

## Reference-research trigger

Do not browse by default for every visual edit. Use a bounded reference pass when the design decision is expensive to reverse and the repository/user evidence does not already determine it.

Typical triggers:

- new product/brand direction;
- new navigation shell or multi-screen visual foundation;
- user delegates taste and several directions are equally plausible;
- first render reads generic/template-like or exposes weak hierarchy;
- logo/identity/app-icon work without an authoritative brand system;
- theme/responsive variants reveal that the assumed design system is incoherent.

Prefer real products with comparable interaction constraints, official design systems, mature open-source interfaces, or specialized design references with explicit reasoning. Extract relationships and rules rather than copying another product's trade dress.

Use [reference-research-and-critique.md](reference-research-and-critique.md) for the bounded research budget, source hierarchy, synthesis, and self-critique gate.

## Design Read / Design DNA input

Capture the facts the task actually needs. A minimal working view can start with:

```text
surface
primary user job
visual tone / philosophy
density
existing system to preserve
signature decision
responsive risk
interaction/state risk
```

For new or materially redesigned surfaces, expand this through [design-taxonomy.md](design-taxonomy.md) only as useful. Do not fill every design axis mechanically.

### Surface

Choose the closest useful class, not a perfect taxonomy:

- product: application workspace, CRUD/productivity UI, tools;
- dashboard: analytics, monitoring, dense overview;
- landing: marketing/conversion page;
- auth/onboarding: sign-in, sign-up, setup, activation;
- settings: preference/configuration forms;
- content: docs, article, editorial, knowledge;
- commerce: catalog, product, cart, checkout;
- other: describe briefly.

### Primary user job

State the user's main action or decision, not the implementation task. Example: `compare incidents and resolve the urgent one`, not `build cards`.

### Visual tone

Use product-relevant language: restrained, editorial, technical, premium, playful, institutional, utilitarian, calm, energetic, etc. Do not invent a brand personality when the product already communicates one.

### Density

- compact: data-heavy or power-user UI;
- balanced: ordinary product/consumer UI;
- spacious: marketing/editorial/premium contexts where lower density is intentional.

### Existing system to preserve

Name the durable evidence: e.g. `shadcn primitives + existing 8px spacing scale + Inter + neutral slate palette`. Component-library presence does not mean every default visual should be kept; distinguish primitives from product styling.

### Signature decision

One intentional choice can stop a generated surface from becoming generic. Prefer a choice tied to real content/workflow/state, for example:

- a distinctive information hierarchy;
- an unusual but useful fold composition;
- a dense command-oriented header;
- a restrained editorial type treatment;
- a spatial relation between navigation and content;
- a product-specific visualization or control;
- context-aware emphasis based on time/task/state.

Do not force a signature flourish onto utility surfaces where consistency and speed are the actual product value.

## Identity discovery

Before inventing a visible product mark, inspect existing logos, favicons, manifests/PWA icons, brand assets, shipped headers/sidebars, and accepted screenshots.

If there is no verified logo and the task is not actually branding, do not synthesize a decorative boxed initial just because a navigation slot appears to need an icon. A plain product-name wordmark is safer than silently changing the product identity.

Read [identity-and-logo.md](identity-and-logo.md) for the distinction between logos, wordmarks, app icons, favicons, avatar fallbacks, and the generic initial-in-a-rounded-tile pattern.

## Questions policy

Do not ask broad taste questionnaires when repository evidence or the user's prompt already answers the design direction.

When consequential uncertainty remains, use [adaptive-design-discovery.md](adaptive-design-discovery.md): classify facts as `KNOWN / INFERRED / UNKNOWN`, estimate question value from impact/uncertainty/rework/reversibility, and ask only high-value questions in ordinary user language.

If the user says `you decide`, stop optional clarification. Choose from product/repository evidence and, when the decision remains materially underdetermined, use the bounded reference-research pass rather than treating the agent's preferred default as authoritative.

## Persistent design memory

A repository may already use:

```text
DESIGN.md
DESIGN_SYSTEM.md
.superdesign/design-system.md
brand guideline docs
Storybook
visual regression baselines
product current-status/history docs
```

Respect the existing convention. Do not add another competing design-memory file.

If no durable design memory exists, an in-turn Design DNA is enough for ordinary one-off or narrow work. When substantial multi-screen work, repeated agent changes, or visible design drift make missing design authority a material risk, establish the smallest repository-owned durable source that fits the existing architecture. Prefer tokens/theme/component primitives and existing docs; create a compact persistent design-system document only when cross-screen rules cannot be expressed clearly elsewhere and repository guidance allows it.

Read [design-system-authority.md](design-system-authority.md) for the durable-authority trigger, subtractive-first gate, token/component discipline, reference locking, theme coherence, and deviation contract.
