# Security Policy

## Supported Versions

physics-lab is an educational project and is not formally versioned or released. Security fixes are applied to the `main` branch only.

| Version | Supported |
|---------|-----------|
| `main`  | ✅        |
| older commits | ❌  |

## Reporting a Vulnerability

If you discover a security issue, please **do not open a public issue**. Instead, email the maintainer directly:

**joaohenriquebarbosa21@gmail.com**

You can expect an initial acknowledgement within **72 hours**. Please include enough detail to reproduce the issue (affected file, steps, and impact).

## Process

1. **Report** the issue privately via the email above.
2. **Acknowledge** — we confirm receipt within 72 hours.
3. **Assess** — we reproduce the issue and determine its severity and scope.
4. **Fix** — we prepare a patch on `main` and validate it.
5. **Disclose** — once a fix is in place, we credit the reporter (if desired) and describe the issue publicly.

## Scope

This project runs untrusted-free, local Python scripts that render graphics and write video files. The relevant security concerns are:

- **Injection or unsafe input handling** — e.g. command-line arguments or file paths flowing into the `ffmpeg` subprocess calls.
- **Data or credential exposure** — the repo should contain no secrets; report any that slip in (API keys, tokens, absolute paths tied to a personal machine).
- **Dependency vulnerabilities** — issues in NumPy, matplotlib, Pygame, ModernGL, Pillow, or CuPy that affect this project.

Out of scope: the physical accuracy of the simulations, the known approximations documented in the README, and issues that require an already-compromised host.
