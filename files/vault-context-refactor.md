# Vault Refactor — Instructions for Claude Code

> **Target repo:** `ecommerce-vault` (the markdown knowledge base fed into Qdrant).
> **Companion doc:** `shop-context-refactor.md` — the Medusa shop repo side. Execute this doc's Phases V0–V5 **before** running Phase S6 (Stop hook) in the shop, because S6 writes into folders this doc creates.
>
> **Mode of work.** Plan-first, small verifiable tasks, senior-engineer quality. Before each phase: propose concrete approach, surface assumptions and blockers, get approval, then execute. Non-destructive where possible — `git mv` over `rm`, snapshot Qdrant before dropping collections.

---

## What this repo is for (reset of intent)

`ecommerce-vault` is the **distilled-knowledge corpus** that feeds Qdrant for fuzzy retrieval inside Claude Code. It is **not** a project doc dump, not a mirror of the shop's source code, and not a personal journal. Everything here should be durable, retrieval-friendly prose with a reason to be searched weeks or months from now.

Execution-level detail (phase-by-phase implementation specs) is kept in `archive/` for human reference but **excluded from the index**. The index is the part that matters for Claude Code quality.

---

## Corpus shift (the core change)

```
BEFORE                          AFTER
──────                          ─────
vault/                          vault/
 └─ projects/medusa/             ├─ adr/        ← why decisions were made
     front-end/                  ├─ domain/     ← durable mental models
     ├─ phase-1-*.md             │   ├─ cart/
     ├─ phase-2-*.md             │   ├─ checkout/
     ├─ ...                      │   ├─ inventory/
     └─ phase-N-*.md             │   ├─ fulfillment/
                                 │   ├─ pricing/
indexed → execution detail,      │   ├─ customer/
stale after phase completes,     │   └─ admin/
hard to generalize               ├─ patterns/   ← reusable recipes
                                 ├─ gotchas/    ← "don't do X because Y"
                                 ├─ runbooks/   ← operational how-tos
                                 ├─ sessions/   ← auto-distilled session logs
                                 ├─ inbox/      ← drafts pending review
                                 └─ archive/
                                     └─ phases/ ← old phase docs, NOT indexed
```

---

## Phase V0 — Audit & report (read-only)

Produce a single report at `/tmp/vault-audit.md` and show it to me before proceeding.

1. `ls` the vault root and two levels deep. Count markdown files per folder.
2. For each file currently under `projects/`, extract: filename, first h1, approximate line count.
3. Query Qdrant REST at `http://localhost:6333/collections` — list collections, point counts, vector configs. For the main collection report: named vectors present (sparse, dense, which dense models), dimensions, distance metric.
4. Fetch one sample point from the collection and inspect `payload` keys. Confirm whether both `nomic-embed-text-v2-moe` and `fast-bge-large-en-v1.5` dense vectors are stored per-point, or only one.
5. Locate the sync script that pushes markdown into Qdrant. Report its path, how it's triggered (git hook, file watcher, manual), and summarize what it does in ≤ 5 bullets.
6. Report any non-markdown assets in the repo (images, scripts, config) that the sync script touches.

**Do not make recommendations in the audit.** Facts only.

---

## Phase V1 — Folder restructure (non-destructive)

**Steps:**

1. **Propose a classification mapping first.** Walk every file currently under `projects/medusa/front-end/` (and any other content folders) and assign each to one of:
   - `archive/phases/` — default for phase implementation specs. Kept for reference, **excluded from index**.
   - `domain/<bounded-context>/` — if the doc contains durable domain knowledge that will still be relevant in 6 months.
   - `patterns/` — if it contains a reusable recipe (form submit pattern, error handling shape, etc.).
   - `gotchas/` — if it contains "don't do X because Y."
   - `runbooks/` — operational how-tos (how to reset the DB, how to regenerate Medusa types, etc.).

   Show me the proposed mapping as a table. Ask for approval before moving anything.

2. Create the target folder tree:
   ```
   vault/adr/
   vault/domain/{cart,checkout,inventory,fulfillment,pricing,customer,admin}/
   vault/patterns/
   vault/gotchas/
   vault/runbooks/
   vault/sessions/
   vault/inbox/
   vault/archive/phases/
   ```

3. Add a short README.md (2–5 lines) in each folder describing what belongs there and what doesn't. Example for `patterns/README.md`:
   > Reusable recipes. One pattern per file. Each file has: Problem, Solution, Code snippet, When not to use. If it's specific to one feature only, it belongs in `domain/` instead.

4. `git mv` files into their targets per the approved mapping. Preserve history.

5. Seed these files with templates ready to fill in:
   - `adr/0001-embedding-model-choice.md` (will be filled in Phase V3)
   - `adr/0002-cart-mutation-strategy.md`
   - `adr/0003-vault-scope-and-boundaries.md` — capture the reasoning behind this refactor itself
   - `gotchas/medusa-cross-module-writes.md` — pre-populate: "Never import another module's service. Cross-module writes go through workflows; reads use Query graph."
   - `gotchas/medusa-service-naming.md` — pre-populate: "MedusaService factory generates plural for create/list/delete (`createFoos`) and singular for retrieve (`retrieveFoo`)."
   - `gotchas/medusa-orm-is-mikroorm.md` — pre-populate: "ORM is MikroORM via `model.define`. Not Prisma. Migrations via `medusa db:generate <module>`."
   - `gotchas/medusa-generated-types-forbidden.md` — pre-populate: "Never import from `.medusa/types/*`. Use `InferTypeOf<typeof Model>` instead."
   - `patterns/shadcn-form-submit.md` — generalize the StepAddress submit handler (loading + error + disabled button).

6. Establish frontmatter convention — add a minimal frontmatter block to every seeded file:
   ```yaml
   ---
   status: accepted      # accepted | proposed | deprecated
   created: YYYY-MM-DD
   tags: [medusa, backend]
   ---
   ```
   Only `status: accepted` files will be indexed (enforced in Phase V2).

7. Commit: `chore(vault): restructure for layered retrieval`.

**Verification:** `find . -name '*.md' | wc -l` is unchanged relative to pre-refactor. No data loss.

---

## Phase V2 — Sync script: scope change + frontmatter filter

Modify the sync script so the indexed corpus is genuinely the durable prose, not everything in the repo.

1. Switch from blacklist to **whitelist** of indexable folders:
   ```python
   INDEXABLE_FOLDERS = ["adr", "domain", "patterns", "gotchas", "runbooks", "sessions"]
   ```
   `inbox/`, `archive/`, and everything else is skipped.

2. Add a frontmatter filter: skip files whose frontmatter has `status: proposed` or `status: deprecated`. Only `status: accepted` (or no status field, for backwards compat during transition) gets indexed.

3. Improve the payload. Each point should carry at minimum:
   ```
   path            — full vault-relative path
   folder          — top-level category (adr / domain / etc.)
   doc_type        — inferred from folder
   h1              — document title
   h2, h3          — section headings
   tags            — from frontmatter
   status          — from frontmatter
   chunk_index     — already present, keep
   modified_at     — already present, keep
   ```

4. Log a summary at the end of each sync run:
   ```
   Indexed 142 chunks from 38 files.
   Skipped: 27 files in archive/, 12 files with status: proposed, 3 files with no markdown body.
   ```
   Store the last run's summary in `.sync-last-run.json` at repo root (gitignored).

5. Test locally before enabling the git hook: run the script manually, verify counts match the log, verify one known `archive/` file is NOT in the index, verify one `status: proposed` file is NOT in the index.

Commit: `feat(sync): whitelist indexable folders and filter by status`.

---

## Phase V3 — Embedding consolidation & re-index

**Decide first:** if V0 audit showed both `nomic-embed-text-v2-moe` and `fast-bge-large-en-v1.5` indexed per-point, pick one.

- **Default recommendation:** keep `fast-bge-large-en-v1.5`. Higher quality for English prose, strong fastembed/qdrant support, consistent with the sparse+dense hybrid pattern already in use.
- **Keep nomic instead** if: the vault has meaningful Polish content you want retrievable (nomic is multilingual). Flag this and we discuss.

Record the decision in `adr/0001-embedding-model-choice.md` with: Options considered, Choice, Rationale, Date. One page.

**Execute the re-index:**

1. **Snapshot the existing collection first.** Use the Qdrant snapshot API:
   ```bash
   curl -X POST http://localhost:6333/collections/<collection-name>/snapshots
   ```
   Store the snapshot path. This is your rollback.

2. Drop the old collection.

3. Create a new collection configured for a single named dense vector + sparse, using the chosen dense model's dimensions.

4. Run the modified sync script (Phase V2) to populate it from scratch.

5. Sanity-check point count against the sync script's log.

6. Run three retrieval smoke tests via `qdrant-mcp` (or the REST API with a generated query vector):
   - "why do we default country to PL in checkout"
   - "how do we handle cross-module writes in Medusa"
   - "shadcn form submit with loading and error state"

   Each should return the corresponding seeded file as a top-3 hit. If not, stop and debug — likely a chunking or payload issue, not an embedding issue.

Commit: `feat(index): consolidate on bge-large, drop dual-embedder setup`.

---

## Phase V4 — Contextual retrieval on vault (optional, skip if V3 is good)

Anthropic's contextual-retrieval pattern: before embedding each chunk, prepend a 50–100 token context line generated by Haiku that situates the chunk within its document. This is the single highest-leverage retrieval upgrade if V3 quality feels weak.

**Skip this phase** if the V3 smoke tests returned the right files with good margin. Log it as a future upgrade in `runbooks/vault-improvements.md` and move on.

**If proceeding:**

1. Extend the sync script:
   ```
   for each chunk in doc:
     context = haiku(
       system="Generate a 50-100 token context line describing what this chunk
               is about, what document it came from, and why a developer might
               search for it. Output plain text, no preamble.",
       user=f"Document: {doc_path}\nFull doc: {full_doc}\nChunk:\n{chunk}"
     )
     embedded_text = f"{context}\n\n{chunk}"
     payload["context"] = context      # store for inspection
   ```

2. Use Anthropic prompt caching on the `full_doc` part — cost drops to roughly $0.01 per 1000 chunks.

3. Re-run the three smoke tests. If margin over non-contextual is small, don't bother adding a reranker on top. If margin is large, consider adding a reranker as a follow-up (deferred).

Commit: `feat(index): contextual retrieval preprocessing`.

---

## Phase V5 — Inbox convention for automated drafts

The shop repo's Stop hook (see `shop-context-refactor.md` Phase S6) will write automatically-distilled learnings **into this vault**. This phase establishes the convention so those writes land in a reviewable place rather than polluting the index immediately.

1. In `inbox/README.md`:
   > Drafts from the session-distillation Stop hook land here. Review weekly. Promote good ones to their proper folder (`adr/`, `patterns/`, `gotchas/`) by moving the file and changing frontmatter `status` from `proposed` to `accepted`. Delete the rest.

2. Subfolder layout inside `inbox/`:
   ```
   inbox/adr/
   inbox/patterns/
   inbox/gotchas/
   inbox/sessions/     ← session logs land here directly, promoted to sessions/ nightly
   ```

3. Nightly promotion: a simple cron or git hook that moves `inbox/sessions/*.md` older than 24h into `sessions/YYYY-MM/` (monthly subfolders to avoid a flat directory with thousands of files). Optional but recommended.

4. Add a review slash command pattern (documented in `runbooks/inbox-review.md`): a weekly routine that lists new inbox items, reads each, and either promotes or deletes. Keep humans in the loop for the first two months before considering any auto-promotion logic.

5. Update the sync script to explicitly skip `inbox/` (already handled by the whitelist in V2, but add a test asserting this).

Commit: `feat(vault): inbox convention for automated session distillation`.

---

## Phase V6 — Verification (vault side)

Run through and report status.

- [ ] `find . -name '*.md' | wc -l` ≥ pre-refactor count
- [ ] Folder structure matches the target tree
- [ ] Each folder has a README.md explaining its scope
- [ ] Sync script uses whitelist + frontmatter filter
- [ ] Qdrant collection uses a single dense embedder
- [ ] `adr/0001-embedding-model-choice.md` documents the choice
- [ ] Three retrieval smoke tests return correct top hits
- [ ] Qdrant snapshot from Phase V3 is preserved as rollback
- [ ] `inbox/` exists with README and subfolders
- [ ] `archive/` files do NOT appear in Qdrant queries
- [ ] `status: proposed` files do NOT appear in Qdrant queries

---

## Rollback

- **Folder restructure:** `git revert` the restructure commit.
- **Sync script changes:** `git revert` the sync commits.
- **Embedding consolidation:** restore from the Phase V3 snapshot via the Qdrant snapshot restore API.

One commit per phase. Don't stack phases without committing.

---

## Handoff to the shop repo

Once V0–V5 are complete, the shop repo can proceed with `shop-context-refactor.md`. Specifically:

- **Phase S2 (MCPs):** `qdrant-mcp` must point at the new collection. Verify connection string / collection name.
- **Phase S6 (Stop hook):** writes to `<absolute-path-to-vault>/inbox/`. The exact path is configured in the shop's `.claude/settings.json`.

Surface any paths, collection names, or config values the shop side will need at handoff time in a short `/tmp/vault-handoff.md`:
```
vault_path: /absolute/path/to/ecommerce-vault
qdrant_collection: <name>
embedder: fast-bge-large-en-v1.5
inbox_sessions_path: <vault>/inbox/sessions/
inbox_adr_path: <vault>/inbox/adr/
inbox_patterns_path: <vault>/inbox/patterns/
inbox_gotchas_path: <vault>/inbox/gotchas/
```

---

## 15. First action

Phase V0 audit. Propose the commands you plan to run before running them, confirm with me, then execute and produce `/tmp/vault-audit.md`.
