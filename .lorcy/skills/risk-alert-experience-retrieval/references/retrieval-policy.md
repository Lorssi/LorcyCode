# Retrieval Policy

Historical experience is advisory, not authoritative.

Use retrieval when:

- starting the first design in a new run,
- evaluation has stagnated for multiple iterations,
- a failure resembles a known project-specific error,
- a metric pattern matches a previous successful repair.

Avoid retrieval when:

- the current execution failure has a clear local stack trace,
- the retrieved design conflicts with mandatory code constraints,
- the historical model depends on unavailable features or changed data schema.

When applying a retrieved design:

1. Extract the reusable principle.
2. Check it against current feature schema and model constraints.
3. Prefer small modifications over copying full old code.
4. Record why the retrieved experience is relevant.
