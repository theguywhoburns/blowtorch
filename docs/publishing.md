# Publishing to PyPI

Releases are fully automatic once configured. The whole pipeline is:

    bump version in pyproject.toml
    git tag vX.Y.Z && git push --tags
    (workflow builds, drafts the GitHub release, publishes to PyPI)

No secrets are required — the workflow uses PyPI's trusted publishing
(GitHub OIDC). A `PYPI_TOKEN` secret in the GitHub repo works as a fallback.

## Prerequisites (one-time)

1. **Name**: the distribution name in `pyproject.toml` must be free on PyPI.
   `crematorium` is free as of 2026-09-02 (`https://pypi.org/pypi/crematorium/json`
   404s; search only returns `pyrokinetics` typo). Confirm before first publish.

2. **Trusted publishing** (no token path): on PyPI, create the project and
   register this repo as a publisher (PyPI -> project -> Publishing ->
   "Add a new pending publisher"):
   - Publisher: GitHub
   - Owner: `theguywhoburns`
   - Repository: `crematorium`
   - Workflow name: `release.yml`

   The same must be repeated on TestPyPI for the dry-run channel to work.

   Token path instead: GitHub repo -> Settings -> Secrets and variables ->
   Actions -> New repository secret `PYPI_TOKEN` (a PyPI API token with
   "Upload packages" scope).

## Workflow

- **Every tag `v*`**: `uv build` (sdist + wheel), a GitHub release with the
  artifacts attached, and a publish to PyPI.
- **Pre-release tags containing `rc`** (e.g. `v0.1.0rc1`): additionally a
  dry-run publish to TestPyPI so an accidental bad build never hits the real
  index.
- **Every push / PR**: the `test` workflow runs `pytest` and `pyright`; the
  release workflow never runs unless a tag is pushed.

## Releasing a new version

1. Bump `version` in `pyproject.toml` and commit.
2. `git tag v<version>` and `git push --tags`.
3. Watch the "release" workflow in GitHub Actions; the PyPI upload lands
   automatically.

## Manual dry-run (no tag needed)

    uv build
    uv publish --publish-url https://test.pypi.org/legacy/   # TestPyPI
    uv publish                                               # real PyPI

Trusted publishing works for these too, as long as the environment can mint
an OIDC token (GitHub Actions) or `UV_PUBLISH_TOKEN` is set.
