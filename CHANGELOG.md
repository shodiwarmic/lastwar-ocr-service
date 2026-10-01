# Changelog

Every release of the OCR service, newest first. The version describes the wire contract
(see [docs/RELEASING.md](docs/RELEASING.md)); each entry is written in the pull request that
becomes the release, and the release notes are cut from it.

## [v1.0.0] — unreleased

The first versioned release. Every earlier image was unversioned: GHCR's `:latest` and
`:local` moved on every merge, and production ran a hand-built image tied to no commit.

### Contract

- **Wire contract v1, written down and negotiated.** Requests may carry `schema_version`
  (absent means 1); anything else is refused with `400 {code: "schema_not_supported",
  supported_versions}`. Responses carry a top-level `schema_version`, and diagnostics name
  the `service_version` and `service_commit`. An unknown category is refused with
  `code: "category_not_supported"` and the supported list. Every addition is optional, so
  callers written before it keep working.
- **`/health` reports capabilities:** `version`, `commit`, `schema_versions`, `categories`.

### Added

- **The three post-event mails:** categories `alliance_exercise`, `zombie_siege` and
  `desert_storm`, read when the request names them (the classifier never picks a mail). The
  Alliance Exercise MVP card comes back as rank 1; a Zombie Siege row with zero waves is kept
  (a score may now be 0); a row whose score cell could not be read is kept with
  `score_unread: true`. Each section reports the mail's `mail_timestamp`, and a collapsed
  list is noted as `no_rows_below_header`.
- **Ranks.** Every row carries the `rank` read beside it, or `rank_inferred: true` where its
  neighbours settle it, and each section reports a rank checksum (`ranks`: gaps,
  duplicates, out-of-order) and `order_violations`. Values are never repaired.
- **Categories come from the screen definitions**, so a new screen's category is not a code
  change. Alliance Contribution keys now resolve to their own definition's settings.

### Fixed

- The result cache was keyed on the image bytes alone, so the same frames sent under a second
  category returned the first category's results. It is now keyed on the bytes, the category
  and the contract version.

### Releases and deployment

- Releases are `vX.Y.Z` tags. A merge to `main` publishes only `:edge` / `:edge-local`.
  A tag publishes `:vX.Y.Z` / `:vX.Y.Z-local`, deploys that digest to production through a
  no-traffic revision and a smoke test, and only then moves `:latest`, `:local`, `:vX.Y`
  and `:vX.Y-local`. A self-hosted sidecar on `:local` no longer picks up untested merges.
- Tests and both image builds run on every pull request and gate every release.
- Production runs as a dedicated runtime account with Vision access only (it ran as the
  project's default Editor account), at `--concurrency 1 --max-instances 3`.
- The README's hand-built deploy path is retired, with its `--allow-unauthenticated` and
  `us-central1`. `docs/GCP_PERMISSIONS.md` documents every identity, the one-time setup,
  key rotation and the required budget alert.
