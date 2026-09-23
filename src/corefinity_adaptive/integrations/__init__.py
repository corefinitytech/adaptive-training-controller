"""Optional downstream sinks for the experience layer (e.g. MLflow).

Not yet implemented. The `ExperienceStore` remains the source of truth; integrations
here are opt-in, read-only sinks, never a replacement for it. Lands once the core store
is stable, per the project's build-sequence plan (`docs/adr/`).
"""
