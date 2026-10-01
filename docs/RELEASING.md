# Releasing the OCR service

A release is a `vX.Y.Z` tag on `main`. Pushing it publishes the images, deploys
production, and only then moves the tags self-hosted installs follow. The
workflow is `.github/workflows/docker-publish.yml`; this document is the
procedure and the rules it enforces.

## What each kind of push publishes

| Push | GHCR | Production (Cloud Run) |
|---|---|---|
| merge to `main` | `:edge`, `:edge-local` | nothing |
| tag `vX.Y.Z` | `:vX.Y.Z`, `:vX.Y.Z-local`; then `:vX.Y`, `:vX.Y-local`, `:latest`, `:local` | the tag's digest, after a smoke test |

`:latest` and `:local` mean **latest release**, never latest commit. The app's
local-OCR compose file runs `:local`, so a merge to `main` reaches no sidecar.

## Levels

The version describes the **wire contract** (the screen-definitions README,
Consumer Contract → Wire contract v1), because that is what callers depend on.
Nothing ships outside the image, so the app's host-file rules do not apply.

| Level | When |
|---|---|
| major | the contract's version is bumped, or a capability is removed (a category, a field) |
| minor | a new category, or a new optional field within the current contract |
| patch | fixes: reading accuracy, tuning, dependencies, docs |

The first release is `v1.0.0`.

## Procedure

1. **Write the `CHANGELOG.md` entry in the PR** that will become the release, as
   `## [vX.Y.Z] — YYYY-MM-DD`. The release notes are cut from it, and the
   Publish job refuses a tag without one before anything is pushed.
2. **Merge, then wait for the checks on the merge commit.** `Tests` and
   `Docker Build` run on `main`; a squash merge is a new commit with its own
   run, so the PR's green checks do not count. Publish refuses a tag whose
   commit has no successful run of both, or that is not on `main`.
3. **Tag the merge commit from a personal clone:**
   ```bash
   git fetch origin && git checkout origin/main
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
   A tag pushed by an action with the default `GITHUB_TOKEN` triggers no
   workflow, so the release would build nothing and look like success.
4. **Watch the four jobs:**
   - **Publish** — the checks above, then one cloud-image build pushed to GHCR
     *and* the private Artifact Registry repository (same digest), and the
     local image to GHCR. Version tags only.
   - **Deploy** (`production` environment) — a `--no-traffic` revision on that
     digest, tagged `candidate`; `/health` on its URL must report the tag, and
     one `/process-batch` of `tests/fixtures/smoke/weekly_frame.png` must return
     rows; then traffic moves with `--to-latest`.
   - **Promote** — retags `:latest`, `:local`, `:vX.Y`, `:vX.Y-local` by
     manifest (no rebuild) and creates the GitHub Release from the changelog.
5. **Tag the app after production runs the release**, when an app release
   depends on it. The app checks `/health` before using a capability and
   refuses readably against an older service, but the order keeps production
   from ever meeting that.

## When a release fails

| Failed in | What was published | Remedy |
|---|---|---|
| Publish, before a push (release check, federation) | nothing | delete the tag, fix, re-tag the same version |
| Publish, after a push | `:vX.Y.Z` and/or `:vX.Y.Z-local` | do **not** re-tag: fix and release the next patch |
| Deploy | the version tags; production unchanged (no traffic moved) | fix and release the next patch; delete the failed revision if wanted |
| Promote | everything but the floating tags or the Release | re-run the failed job; it retags existing bytes |

**Never rebuild a published version.** Two different images under one tag is
the one thing a version must never mean.

## Rolling back production

The previous revision stays deployable until its image leaves the Artifact
Registry repository, which keeps the three most recent versions
(`deploy/artifact-registry-cleanup.json`):

```bash
gcloud run revisions list --service lastwar-ocr-service --region us-east1
gcloud run services update-traffic lastwar-ocr-service --region us-east1 \
  --to-revisions <previous-revision>=100
```

An older release can always be pulled from GHCR, which keeps every version.
