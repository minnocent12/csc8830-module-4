# Module 4 Git Workflow

This file adapts the course-level Git protocol for the independent Module 4 repository.
Module 4 is rooted at Assignments/Module_4 and must remain independently runnable and
submittable.

## Repository safety

- Use one task branch per implementation task.
- Do not implement ordinary work directly on main or master.
- Do not force-push, reset hard, clean untracked files, delete branches or repositories,
  remove .git, rewrite published history, or rebase published history.
- Do not modify Module 2 or Module 3 history or create runtime dependencies on them.
- Do not create or change a GitHub remote, push, merge, or submit externally without explicit
  user authorization.
- Preserve the user's configured Git identity. Never add AI attribution to commits, branches,
  pull requests, or repository content.

## Normal workflow

Before implementation:

    git status --short --branch
    git branch --show-current
    git rev-parse --show-toplevel

Confirm that the top-level path is exactly the Module 4 directory and that unrelated changes
are understood. Create a task branch such as:

    git switch -c task/module-4-foundation

During work, inspect changes with:

    git status --short --branch
    git diff
    git diff --cached

Stage explicit paths only. Review the staged diff before committing. Use normal descriptive
commit messages without AI or agent attribution. After an important operation, verify the
branch and working-tree state again.

## Module 4 repository boundary

The repository root must remain:

    Assignments/Module_4

The Assignments root, Module 2, and Module 3 are separate filesystem/repository contexts.
Module 4 source code may not import from Module 2 or Module 3. The root dashboard may mount
Module 4 through its structural page-provider contract, but it is not a Module 4 dependency.

## Academic integrity

Never commit fabricated experiment results, SAM2 masks, metrics, dataset provenance, physical
measurements, or validation claims. Until real user experiments exist, use explicit pending,
not measured, or PENDING USER EXPERIMENT status. Never commit secrets, credentials, API keys,
virtual environments, caches, large datasets, or generated artifacts that the module's
.gitignore excludes.

## Phase-gated work

Each approved phase must be implemented separately. At the end of a phase:

1. run the relevant tests and checks;
2. inspect the diff and status;
3. report every changed file;
4. report exact test commands/results and skipped checks;
5. record deviations and acceptance status; and
6. stop until the user approves the next phase.
