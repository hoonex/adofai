# Design system authority

Use this reference when a substantial UI task spans multiple components/screens, when the repository has visible design drift, when repeated agent work is likely to introduce new visual conventions, or when the user explicitly wants the product's design system to become durable.

The goal is not to freeze creativity. The goal is to stop every implementation turn from inventing a new local style.

## Core rule

Treat every material UI change as a delta against an authoritative design system, not as a fresh design exercise.

```text
existing durable system -> preserve and extend deliberately
no durable system + one-off small surface -> keep compact in-turn Design DNA
no durable system + multi-screen/repeated work/drift risk -> establish durable design authority
```

Do not create a second competing system when one already exists.

## What counts as design authority

Prefer the strongest existing durable source in roughly this order:

1. explicit user-approved product/design direction;
2. repository-owned design-system/brand documentation;
3. tokens/theme files and component primitives;
4. accepted Storybook/component examples;
5. accepted rendered screenshots or visual regression baselines;
6. repeated shipped patterns that are internally coherent.

When these disagree, do not silently average them. Resolve the contradiction from user/repository intent and record the winning rule in the existing durable source when the task changes the system.

## Design constitution

A durable design system does not need a large document. It needs enough authority to constrain later decisions.

When durable design authority is justified and the repository lacks one, establish the smallest appropriate source of truth. Reuse an existing theme/token/component file when possible. A compact `DESIGN_SYSTEM.md` or equivalent is appropriate only when the repository needs cross-screen design memory that cannot be expressed clearly in code alone.

Capture only decisions that future work must preserve:

- product tone and visual attitude;
- primary information hierarchy and density posture;
- typography roles and type scale logic;
- semantic color roles and theme behavior;
- spacing rhythm and layout/grid rules;
- radius, border, shadow, elevation, and material grammar;
- component primitives and when each is appropriate;
- interaction feedback and state language;
- motion posture and reduced-motion behavior;
- responsive reprioritization rules;
- icon/illustration/media conventions;
- one or two signature product-specific decisions;
- explicit anti-patterns that have already caused drift.

Do not document every pixel value. Prefer tokens, principles, and representative examples over prose duplication.

## Subtractive-first gate

Generated UI often becomes worse by adding plausible but unnecessary structure. Before adding a new visible element, container, section, decoration, metric, helper text, badge, chip, icon, illustration, or effect, require a product reason.

A new element should normally satisfy at least one of these:

- enables an actual user action;
- exposes necessary state or feedback;
- improves comprehension of real information;
- creates semantic grouping that spacing/alignment alone cannot express;
- resolves a responsive/accessibility problem;
- carries product identity or domain meaning that would otherwise be absent.

If none applies, do not add it merely because the layout looks empty or because similar products often contain it.

Use this deletion test after implementation:

```text
If this element vanished, would the user's task, understanding, state awareness,
or product identity materially degrade?
```

If not, strongly prefer removing it.

## Component introduction gate

Before creating a new component or visual primitive:

1. inspect existing primitives and variants;
2. prefer composition/variant extension over a near-duplicate component;
3. create a new primitive only when the semantic role or interaction contract is genuinely distinct;
4. if the new primitive introduces a new visual convention, decide whether the system itself is changing.

Do not create a new card type, pill type, shadow level, radius tier, spacing scale, button hierarchy, icon container, or surface material for one local convenience when an existing system can express the need.

## Token discipline

Do not introduce raw one-off visual values when an authoritative token exists.

When no token exists:

- local implementation detail with no reuse/identity consequence -> keep it local and restrained;
- recurring or system-level visual decision -> add/update the appropriate token deliberately;
- uncertain system change -> do not proliferate nearby arbitrary values while deciding.

A new token is not automatically better than a raw value. Tokens should encode stable design meaning, not catalog every number used once.

## Hierarchy budget

Each surface should have a clear primary purpose and a limited number of competing emphasis levels.

Before adding another prominent panel, CTA, accent color, headline, badge, or animated object, ask what existing element it is competing with. If everything is emphasized, hierarchy has failed.

Prefer hierarchy from:

1. placement and composition;
2. typography and density;
3. semantic color/state;
4. boundaries/elevation;
5. decorative effects last.

Do not use cards as the default answer to grouping and do not use empty space as an excuse to manufacture content.

## Deviation contract

A design system is a default authority, not a prison. A deliberate deviation is valid when a product requirement, new interaction, accessibility constraint, or explicit user direction requires it.

For a material deviation, be able to state internally:

```text
existing rule:
why it fails here:
new rule or scoped exception:
where the exception applies:
```

If the deviation should recur, update the durable system. If it is intentionally local, keep its scope narrow.

## Reference locking

When the user supplies a reference image/site/screenshot or explicitly approves a rendered direction, extract the reusable rules instead of copying isolated pixels.

Lock the durable parts:

- hierarchy;
- density;
- spacing rhythm;
- type relationships;
- material/surface behavior;
- color roles;
- interaction feel;
- responsive behavior;
- distinctive composition decisions.

Do not copy unrelated content, branding, or decorative artifacts from a reference merely because they are visible.

Accepted screenshots may serve as visual baselines when the repository has no stronger source, but source code and tokens should still express the system where practical.

## Theme coherence

Light/dark/high-contrast modes are variants of one system, not independent redesigns.

Do not derive dark mode by simple color inversion. Preserve semantic relationships such as:

- page vs raised/sunken surface separation;
- primary vs secondary text hierarchy;
- selected/hover/pressed/focus distinction;
- border/elevation legibility;
- brand/accent role;
- translucent/material behavior over the actual background.

When a component looks correct in light mode but awkward in dark mode, inspect token ownership and surface relationships before adding a local dark-mode override.

## Completion gate

For substantial UI work, completion requires more than visual plausibility.

Check:

- no unnecessary visible structure was added;
- new components/tokens have a system-level reason or are intentionally local;
- the result remains recognizably part of the same product;
- hierarchy is clearer, not merely more decorated;
- light/dark/responsive states preserve the same design grammar;
- any deliberate deviation is scoped or incorporated into durable design authority;
- rendered evidence supports visual claims when available.

The target is not `more designed`. The target is `more coherent, more necessary, and more specific to the product`.
