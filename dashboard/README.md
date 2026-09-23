# Dashboard

Not started. This is Phase 5 in the project roadmap — after the core adaptive-training
framework, Jev integration, and evaluation harness are proven (Phases 1–4), not before.

**Planned approach** (see `docs/architecture.md` and ADR 0002): a FastAPI + HTMX/Jinja2
app, shipped as an optional `[dashboard]` extra, reading directly from the
`ExperienceStore` (SQLite) and the evaluation harness's outputs — read-only, with no
write path back into training. Launched locally via a console-script entry point
(`corefinity dashboard --db path/to/experience.sqlite`), the same pattern as
`mlflow ui` or `tensorboard --logdir`.
