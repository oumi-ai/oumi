# Telemetry

Oumi collects anonymous usage analytics to help us debug issues and prioritize new features. This page explains our privacy principles and how to opt out.

## What We Collect

We only collect anonymous, aggregate information about how Oumi is used:

- **Usage patterns**: Which commands are run, success/failure rates, performance metrics, training type, etc
- **Environment**: Hardware specs, OS, package versions
- **Errors**: Stack traces to help us fix bugs

We never collect:

- Your training data, model weights, or outputs
- File paths, usernames, or any information that could identify you

## Opting Out

Set the environment variable:

```bash
export DO_NOT_TRACK=1
```

Or edit `~/.oumi/telemetry.json`:

```json
{
  "analytics_enabled": false
}
```
