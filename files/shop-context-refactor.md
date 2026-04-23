# Shop Context Refactor — Instructions for Claude Code

> **Target repo:** the Medusa 2.x + Next.js shop (Anna's ceramics & prints).
> **Companion doc:** `vault-context-refactor.md` — the `ecommerce-vault` repo side. That doc's Phases V0–V5 **must be complete** before this doc's Phase S6 (Stop hook), because S6 writes into folders the vault refactor creates.
>
> **Mode of work.** Plan-first, small verifiable tasks, senior-engineer quality. Before each phase: propose concrete approach, surface assumptions and blockers, get approval, then execute. Non-destructive where possible — preserve the old CLAUDE.md as `CLAUDE.md.bak` until the refactor is proven in daily use.

---

## What this repo is for (reset of intent)

The shop repo is where coding work happens. Its `.claude/` configuration is the **context routing layer** that decides which rules, skills, sub-agents, and MCPs load for which files. The goal of this refactor is to move from a monolithic CLAUDE.md + a single Qdrant MCP toward a layered retrieval stack where each layer does what it's best at and doesn't duplicate the others.

The vault (in a separate repo) is one of six retrieval sources, not the only one.

---

## Target architecture

```
┌─────────────────────────────────────────────────────────────┐
│ CLAUDE CODE (Opus 4.7 main, Haiku 4.5 for sub-agents/hooks) │
├─────────────────────────────────────────────────────────────┤
│ CONTEXT ROUTING (this repo)                                 │
│   Root CLAUDE.md         <150 lines, stack + commands only  │
│   apps/backend/CLAUDE.md    lazy-load, Medusa rules         │
│   apps/storefront/CLAUDE.md lazy-load, Next.js rules        │
│   .claude/rules/*.md     path-scoped rules (paths: glob)    │
│   .claude/skills/        progressive-disclosure skills      │
│   .claude/agents/        isolated sub-agent context         │
│   .claude/settings.json  hooks + MCP + misc config          │
├─────────────────────────────────────────────────────────────┤
│ RETRIEVAL (tiered, cheapest first)                          │
│   1. Grep/Glob/Read   native, 60%+ of code queries          │
│   2. Serena MCP       LSP symbols, find-refs, renames       │
│   3. ast-grep         structural patterns (as skill)        │
│   4. Context7 MCP     Next.js / React / Prisma / libs       │
│   5. Medusa MCP       official, version-tracked Medusa docs │
│   6. Qdrant MCP  ─────┬─► points at ecommerce-vault         │
│                       │   (separate repo, see companion)    │
└───────────────────────┴─────────────────────────────────────┘
```

---

## Phase S0 — Audit & report (read-only)

Produce a single report at `/tmp/shop-audit.md` and show it to me before proceeding.

1. Read the current root `CLAUDE.md`. Report: line count, top-level sections, and classify each section as "always-relevant" or "path-scoped" (belongs in a subtree or `.claude/rules/`).
2. Check for existing subtree files: `apps/backend/CLAUDE.md`, `apps/storefront/CLAUDE.md`. If present, summarize.
3. List `.claude/` contents if the folder exists: `rules/`, `skills/`, `agents/`, `hooks.json`, `settings.json`, `commands/`, plus any files I didn't name.
4. Run `claude mcp list` and report which MCP servers are configured, at which scope (user / project / local), and which are connected vs. disconnected.
5. For the existing `qdrant-mcp` config, report: endpoint, collection name, any auth config. We need this to verify it still points at the right place after the vault refactor.
6. Report project size: `cloc apps/backend apps/storefront` or `tokei` — lines, file counts per language. Confirms we're in the "small-to-medium, no code-RAG needed" size regime.
7. Report any pre-existing hooks, their triggers, and what they do.

**Do not make recommendations in the audit.** Facts only.

---

## Phase S1 — CLAUDE.md refactor

**Target:** root CLAUDE.md under 150 lines. Subtree CLAUDE.md in each app. Long rules in `.claude/rules/*.md` with `paths:` frontmatter.

**Steps:**

1. Copy current `CLAUDE.md` to `CLAUDE.md.bak` (gitignored). Rollback insurance.

2. Propose the split first. Take every section of the current CLAUDE.md and assign to one of:
   - **Root** (< 150 lines total): project name, stack summary, primary commands (`pnpm dev`, `pnpm test`, `pnpm typecheck`), environment vars pattern, quick pointers to subtree docs and rules.
   - **`apps/backend/CLAUDE.md`**: anything Medusa-backend-specific.
   - **`apps/storefront/CLAUDE.md`**: anything Next.js/storefront-specific.
   - **`.claude/rules/<n>.md`**: long-form rules that only apply to specific paths.

   Show me the proposed split as a table before moving content.

3. For each `.claude/rules/*.md` file, use this frontmatter:
   ```yaml
   ---
   paths: ["apps/backend/src/modules/**/*.ts"]
   description: "Medusa 2.x module conventions"
   ---
   ```
   Rules load only when Claude touches matching paths. Keep each file under 40 lines.

4. Create these specific rule files (content templates below in §S1a):
   - `.claude/rules/medusa-modules.md` → `apps/backend/src/modules/**`
   - `.claude/rules/medusa-workflows.md` → `apps/backend/src/workflows/**`
   - `.claude/rules/medusa-api-routes.md` → `apps/backend/src/api/**`
   - `.claude/rules/medusa-admin-widgets.md` → `apps/backend/src/admin/**`
   - `.claude/rules/storefront-server-components.md` → `apps/storefront/app/**`
   - `.claude/rules/storefront-client-components.md` → `apps/storefront/components/**`
   - `.claude/rules/shadcn-forms.md` → `apps/storefront/components/**/*form*.tsx`

5. Root CLAUDE.md ends with explicit retrieval routing:
   > **For library API questions** (Next.js, React, Prisma/MikroORM, Medusa, shadcn/ui), resolve via Context7 MCP or Medusa MCP before writing code. Do not rely on training data.
   > **For project decisions and patterns**, query Qdrant MCP against the `ecommerce-vault` collection.
   > **For code navigation** (find refs, symbols, renames), prefer Serena over grep.

Commit per file group: `refactor(claude): split CLAUDE.md into subtree and rules`.

### S1a. Canonical rule content to seed

**`.claude/rules/medusa-modules.md`:**
```
---
paths: ["apps/backend/src/modules/**/*.ts"]
description: "Medusa 2.x module conventions"
---
# Medusa module rules

- ORM is MikroORM via `model.define` — NOT Prisma. Do not write Prisma schema.
- Migrations: `npx medusa db:generate <module-name>`.
- Generated method naming from MedusaService factory:
  - `createFoos`, `listFoos`, `deleteFoos` — PLURAL
  - `retrieveFoo` — SINGULAR
- Never import another module's service directly. Cross-module WRITES go
  through workflows. Cross-module READS use the Query graph.
- Types: `InferTypeOf<typeof Model>`. Never import from `.medusa/types/*` —
  that directory is generated and forbidden.
```

**`.claude/rules/medusa-workflows.md`:**
```
---
paths: ["apps/backend/src/workflows/**/*.ts"]
description: "Workflows must be idempotent and compensable"
---
# Workflow rules

- Use `createStep` and `createWorkflow` from `@medusajs/framework/workflows-sdk`.
- Every step that writes state MUST have a compensation function.
- Workflows are the only place cross-module writes are allowed.
- Do not throw; return `StepResponse` with a rollback payload.
```

Seed similar short files for the others. Keep each under 40 lines.

---

## Phase S2 — MCP stack

Install in this order. Stop after each and verify with `claude mcp list` before continuing.

### S2.1 Verify existing qdrant-mcp

Before adding anything new, confirm the existing `qdrant-mcp` is pointing at the post-refactor vault collection. Consult `/tmp/vault-handoff.md` (produced by the vault refactor) for:
- `qdrant_collection` — the new collection name
- `embedder` — the chosen dense model

If the collection name changed during the vault refactor, update the `qdrant-mcp` config accordingly. Run a test query and confirm a known ADR/pattern/gotcha comes back.

### S2.2 Medusa official MCP

```bash
claude mcp add --scope user medusa-docs --transport http https://docs.medusajs.com/mcp
```

Then check the Medusa Claude Code plugins repo. **Verify plugin names at execution time** — the ecosystem moves weekly:

```bash
gh repo view medusajs/medusa-claude-plugins --json description,url 2>/dev/null \
  || echo "Plugin repo not found — search medusajs GitHub org for current AI tooling"
```

If plugins exist and match our stack, install them. Likely candidates: `medusa-dev`, `ecommerce-storefront`, `learn-medusa`. If the layout has changed, report what you found and ask before installing.

### S2.3 Serena

```bash
claude mcp add --scope user serena -- \
  serena start-mcp-server --context claude-code --project-from-cwd
```

In `.claude/settings.json`, set `MCP_TIMEOUT=60000`.

Add the "re-prime" hook pattern from Serena's docs — Claude Code's dynamic tool loading causes Serena tools to drop from mid-session awareness. Look up the current canonical snippet in Serena's docs; do **not** copy a stale one.

### S2.4 Context7

```bash
claude mcp add --scope user context7 --transport http https://mcp.context7.com/mcp
```

Free tier is ~1k requests/month as of early 2026. If we hit the limit, budget ~$10/mo for Pro. Record this and quotas for other MCPs in `.claude/runbooks/mcp-quotas.md` (or add to the vault as `runbooks/mcp-quotas.md` — prefer vault since it's durable operational knowledge).

### S2.5 Verify final MCP list

Expected after this phase:
```
serena
medusa-docs
context7
qdrant-mcp         (pre-existing, reconfigured)
infakt             (pre-existing, unrelated)
```

Commit: `feat(claude): layered MCP retrieval stack`.

---

## Phase S3 — Skills

Create `.claude/skills/<skill-name>/SKILL.md` files with frontmatter-triggered progressive disclosure. Each skill body under 100 lines; longer content goes into `reference/*.md` inside the skill folder.

### S3.1 `ast-grep` skill

```
.claude/skills/ast-grep/SKILL.md
```

Triggers: "find all async functions that...", "rename every usage of X", "codemod", "structural search". Body: ast-grep CLI essentials, YAML pattern syntax, 3–5 worked examples. Bundled example scripts in `scripts/` subfolder.

### S3.2 `medusa-workflow` skill

```
.claude/skills/medusa-workflow/SKILL.md
```

Triggers: "create a workflow", "checkout workflow", "order workflow". Body: canonical `createWorkflow` + `createStep` template with compensation; 2–3 concrete examples (cart-to-order, inventory reservation rollback, external payment capture with compensation).

### S3.3 `medusa-module` skill

```
.claude/skills/medusa-module/SKILL.md
```

Triggers: "scaffold a module", "new medusa module", "module service". Body: MedusaService factory pattern, model definition template, migration command, container wiring, service method naming reminder.

### S3.4 `shadcn-form` skill

```
.claude/skills/shadcn-form/SKILL.md
```

Triggers: "form with loading state", "shadcn form", "controlled form submit". Body: canonical submit handler (loading + error + disabled button + destructive error text), based on the StepAddress pattern from the vault's `patterns/shadcn-form-submit.md`.

Commit: `feat(claude): skills for ast-grep, medusa-workflow, medusa-module, shadcn-form`.

---

## Phase S4 — Sub-agents

Create `.claude/agents/<n>.md` files. Each sub-agent runs with a fresh isolated context window when invoked.

### S4.1 `medusa-architect`

- Model: Opus 4.7
- Tools: Read, Grep, Glob, WebFetch, Medusa MCP, Context7 MCP, Qdrant MCP (NO Write/Edit)
- Purpose: design-review feature plans **before** implementation. Invoked for non-trivial work — anything touching multiple modules or crossing the backend/storefront boundary.
- System prompt: senior Medusa architect, thinks in modules/workflows/links, surfaces hidden coupling, asks what compensates what, verifies against Medusa MCP before recommending APIs.

### S4.2 `pr-reviewer`

- Model: Sonnet 4.6
- Tools: Read, Grep, Bash(git), Serena
- Purpose: review staged changes against repo conventions. Accumulates learned conventions in `.claude/agent-memory/pr-reviewer/MEMORY.md`.
- Invocation: slash command or Stop hook after commits.

### S4.3 `repo-explorer`

- Model: Haiku 4.5
- Tools: Read, Grep, Glob, Serena (read-only)
- Purpose: onboarding exploration for new features — "what exists for X already, who imports Y, what's the shape of Z." Returns compact summary so the main agent doesn't burn context crawling.
- System prompt forces summary shape: "Return at most 20 bullets grouped by: Existing / Related / Gaps."

Test each sub-agent with one realistic prompt before moving on. Tighten system prompts if output is bloated.

Commit: `feat(claude): sub-agents for architecture, PR review, and repo exploration`.

---

## Phase S5 — Hooks (deterministic guardrails)

Edit `.claude/settings.json` (or `hooks.json` — check the current Claude Code docs for the authoritative location). Start with these three. Test each individually.

### S5.1 PreToolUse — secret scan

Block Write/Edit if tool_input.content contains patterns matching:
- AWS keys (`AKIA[0-9A-Z]{16}`)
- Stripe live keys (`sk_live_`)
- `-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----`
- Lines that look like env values being hardcoded into `.ts`/`.tsx` (heuristic: assignment to an identifier containing `SECRET`, `TOKEN`, `KEY`, `PASSWORD` with a literal string on the RHS)

Block, don't warn.

### S5.2 PreToolUse — protect generated directories

Block Write/Edit where `tool_input.file_path` matches:
- `apps/backend/.medusa/**`
- `apps/storefront/.next/**`
- `**/node_modules/**`
- `**/dist/**`
- `**/*.generated.ts`

### S5.3 PostToolUse — typecheck after writes

After Write or Edit to a `.ts`/`.tsx` file, run `pnpm -r typecheck --filter <affected-package>`. Surface failures back to the main agent immediately so errors don't accumulate.

**Test each hook:**
- Secret scan: try to write a fake AWS key. Verify blocked.
- Protect generated: try to write to `apps/backend/.medusa/test.ts`. Verify blocked.
- Typecheck: introduce a deliberate type error, verify the failure surfaces.

Commit per hook: `feat(hooks): <hook name>`.

---

## Phase S6 — Stop hook: session distillation into vault

> **Prerequisite:** companion doc `vault-context-refactor.md` Phase V5 must be complete. Confirm `vault/inbox/{adr,patterns,gotchas,sessions}/` exists before writing this hook.

Add a Stop hook (fires when a Claude Code session ends) that distills the session into the vault via Haiku.

1. Read paths from `/tmp/vault-handoff.md` (produced at the end of the vault refactor). Paste these into `.claude/settings.json` as config:
   ```json
   {
     "vault": {
       "absolute_path": "/absolute/path/to/ecommerce-vault",
       "inbox": {
         "sessions": "inbox/sessions",
         "adr":      "inbox/adr",
         "patterns": "inbox/patterns",
         "gotchas":  "inbox/gotchas"
       }
     }
   }
   ```

2. Hook implementation (pseudocode — use Haiku via the Anthropic API):
   ```
   on Stop:
     transcript = read_current_session_transcript()
     if not transcript.substantive_work:
       return
     classified = haiku(
       system="""
         You distill a coding session into durable knowledge.
         Classify each learning into one of:
           - ADR        (a decision made, with rationale)
           - pattern    (a reusable recipe that worked)
           - gotcha     (a thing that broke, with root cause)
         Always produce a session log regardless.
         Output a YAML manifest listing files to write, then the file contents.
         Every file MUST have frontmatter: status: proposed, created: <today>, tags: [...].
       """,
       user=transcript
     )
     for file in classified.files:
       write file to <vault>/<inbox_path>/<slug>.md
     run: cd <vault> && git add inbox/ && git commit -m "docs(inbox): distill session <date>"
   ```

3. Use Anthropic prompt caching on the system prompt — it's stable across sessions.

4. **Start conservative.** First two weeks: all outputs (including non-session classifications) land in `inbox/`, frontmatter `status: proposed`. They do NOT get indexed (Phase V2 sync script filter blocks `proposed`). You review weekly and promote good ones by moving the file and changing status to `accepted`.

5. After the two-week trial, decide whether to graduate any category (likely: session logs) to direct writes into the permanent folder.

6. Test: run the hook manually against a representative session transcript. Verify files land in the expected paths with correct frontmatter.

Commit: `feat(hooks): Stop hook — session distillation into vault inbox`.

---

## Phase S7 — Verification (shop side)

Run through and report status.

- [ ] Root `CLAUDE.md` under 150 lines
- [ ] `CLAUDE.md.bak` exists as rollback
- [ ] `apps/backend/CLAUDE.md` and `apps/storefront/CLAUDE.md` exist
- [ ] `.claude/rules/` contains 7+ path-scoped rule files
- [ ] `.claude/skills/` contains: `ast-grep`, `medusa-workflow`, `medusa-module`, `shadcn-form`
- [ ] `.claude/agents/` contains: `medusa-architect`, `pr-reviewer`, `repo-explorer`
- [ ] `claude mcp list` shows: serena, medusa-docs, context7, qdrant-mcp
- [ ] `qdrant-mcp` returns results when queried for a known vault ADR
- [ ] Medusa MCP responds to a test query about workflows
- [ ] Context7 responds to `resolve-library-id` for "next.js"
- [ ] Serena responds to `find_referencing_symbols` on a known symbol
- [ ] Secret-scan hook blocks a test write of a fake AWS key
- [ ] Protect-generated hook blocks a test write to `apps/backend/.medusa/test.ts`
- [ ] Typecheck hook surfaces a deliberate type error
- [ ] Stop hook writes a test session to `<vault>/inbox/sessions/`

---

## Rollback

- **CLAUDE.md refactor:** restore from `CLAUDE.md.bak`; `git revert` the rule/skill/agent commits.
- **MCPs:** `claude mcp remove <n>`.
- **Hooks:** set `enabled: false` in `.claude/settings.json` per offender rather than deleting — keeps the pattern visible for reference.

One commit per phase. Don't stack phases without committing.

---

## First action

Phase S0 audit. Propose the commands you plan to run before running them, confirm with me, then execute and produce `/tmp/shop-audit.md`.
