# Security

## Reporting A Vulnerability

Please report security issues privately to `contact@abstractcore.ai`.

Include:

- affected version or commit
- reproduction steps
- impact assessment
- any workaround you already verified

## Scope Notes

- Abstract3D is a local-first package and does not ship a network service of its own.
- The validated backend downloads model artifacts from Hugging Face and a pinned upstream source snapshot from GitHub when not already present.
- If you discover a supply-chain or artifact-integrity issue in that path, include the exact repo id, revision, and file names.
