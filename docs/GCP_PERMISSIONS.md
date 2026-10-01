# Google Cloud: identities, permissions and billing

Every Google Cloud identity the OCR service and its caller use, what each is
granted and why, and the one-time setup that creates them. The commands use the
project's own names; a self-hoster substitutes theirs (`PROJECT`, `REGION`,
the repository id).

**The OCR service is never public.** It is deployed with
`--no-allow-unauthenticated` and only the app's service account may invoke it.
Two reasons: every request spends Vision units billed to the project, so a
public URL is an open tap on your bill; and `/health` names the exact release
and commit, which tells an attacker which known bugs apply.

## Identities

| Identity | Grants | Scope | Used for |
|---|---|---|---|
| **The app** — `lastwar-alliance-manager@PROJECT.iam.gserviceaccount.com` (JSON key uploaded in the app's Admin → Security & API) | `roles/run.invoker` | service `lastwar-ocr-service` | the app's calls to the OCR service (an ID token) |
| | `roles/storage.objectCreator` | bucket `lastwar-ocr-archive` | the OCR request archive |
| | `roles/cloudtranslate.user`, `roles/serviceusage.serviceUsageConsumer` | project | message translation |
| **OCR runtime** — `lastwar-ocr-runtime@PROJECT.iam.gserviceaccount.com` | `roles/serviceusage.serviceUsageConsumer` | project | Vision calls. Vision has no resource-level role; the release smoke test is the proof that this is enough. The service parses untrusted images, so its identity carries nothing else. |
| **Deployer** — `lastwar-ocr-deployer@PROJECT.iam.gserviceaccount.com`, reached only through Workload Identity Federation from a `v*` tag of this repository; it has no key | `roles/artifactregistry.writer` | repository `lastwar-ocr` | pushing release images (`writer` includes download) |
| | `roles/run.developer`, `roles/run.invoker` | service `lastwar-ocr-service` | deploying a revision, and the smoke test against it |
| | `roles/iam.serviceAccountUser` | the **runtime** account only | deploying a revision that runs as it. Never on the default compute account, or a release could deploy as Editor. |
| | `roles/iam.workloadIdentityUser` (held by the federated principal, on the deployer) | the deployer account | letting the tag workflow act as it |
| Default compute account — `PROJECT_NUMBER-compute@developer.gserviceaccount.com` | `roles/editor` (Google's default) | project | nothing of ours once the runtime account is in place. It is also Cloud Build's default identity, so do not strip its roles while Cloud Build is enabled. |

`run.developer` cannot change IAM, so the deploy never touches the invoker
policy. Check it with
`gcloud run services get-iam-policy lastwar-ocr-service --region REGION`: the
app's account and the deployer, nothing else, and never `allUsers`.

## One-time setup

```bash
PROJECT=bionic-feat-490600-e3
PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
REGION=us-east1
REPO_ID=1189068018   # gh api repos/<owner>/lastwar-ocr-service --jq .id
RUNTIME=lastwar-ocr-runtime@$PROJECT.iam.gserviceaccount.com
DEPLOYER=lastwar-ocr-deployer@$PROJECT.iam.gserviceaccount.com

gcloud services enable sts.googleapis.com iamcredentials.googleapis.com \
  artifactregistry.googleapis.com run.googleapis.com vision.googleapis.com --project $PROJECT

# The private registry Cloud Run pulls from, in the service's region (pulls are
# free there), keeping the three most recent versions.
gcloud artifacts repositories create lastwar-ocr --project $PROJECT \
  --location $REGION --repository-format docker \
  --description "OCR service releases (deployed by docker-publish.yml)"
gcloud artifacts repositories set-cleanup-policies lastwar-ocr --project $PROJECT \
  --location $REGION --policy deploy/artifact-registry-cleanup.json --no-dry-run

# Runtime and deployer identities.
gcloud iam service-accounts create lastwar-ocr-runtime --project $PROJECT \
  --display-name "OCR service runtime (Vision only)"
gcloud iam service-accounts create lastwar-ocr-deployer --project $PROJECT \
  --display-name "OCR service release deployer (GitHub, via WIF)"
gcloud projects add-iam-policy-binding $PROJECT --condition=None \
  --member serviceAccount:$RUNTIME --role roles/serviceusage.serviceUsageConsumer
gcloud artifacts repositories add-iam-policy-binding lastwar-ocr --project $PROJECT \
  --location $REGION --member serviceAccount:$DEPLOYER --role roles/artifactregistry.writer
for role in roles/run.developer roles/run.invoker; do
  gcloud run services add-iam-policy-binding lastwar-ocr-service --project $PROJECT \
    --region $REGION --member serviceAccount:$DEPLOYER --role $role
done
gcloud iam service-accounts add-iam-policy-binding $RUNTIME --project $PROJECT \
  --member serviceAccount:$DEPLOYER --role roles/iam.serviceAccountUser

# Workload Identity Federation: trust this repository's id (not its name, which
# can be reused after a rename) and release tags only.
gcloud iam workload-identity-pools create github --project $PROJECT \
  --location global --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc lastwar-ocr-service \
  --project $PROJECT --location global --workload-identity-pool github \
  --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping 'google.subject=assertion.sub,attribute.repository_id=assertion.repository_id,attribute.ref=assertion.ref' \
  --attribute-condition "attribute.repository_id == \"$REPO_ID\" && attribute.ref.startsWith(\"refs/tags/v\")"
gcloud iam service-accounts add-iam-policy-binding $DEPLOYER --project $PROJECT \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository_id/$REPO_ID"
```

On GitHub: Settings → Environments → **production**, deployment branches and
tags → selected → tag rule `v*`. The Deploy job runs in it.

The service itself is created by the first deploy. For a new install, create it
before granting the service-scoped roles above, with
`gcloud run deploy lastwar-ocr-service --image <image> --region $REGION
--service-account $RUNTIME --no-allow-unauthenticated --concurrency 1
--max-instances 3 --memory 1Gi --cpu 1 --timeout 120`, then grant the app's
account `roles/run.invoker` on it.

### Why `--concurrency 1 --max-instances 3`

gunicorn runs one sync worker, so an instance processes one request at a time;
any concurrency above 1 only queues requests behind it, and the wait counts
against the 120 s timeout. Three instances bound the spend rate as well as the
parallelism.

## Rehearsing the deploy

The tag workflow is the only thing that federates as the deployer. To check the
deployer's grants without a release, impersonate it (this needs
`roles/iam.serviceAccountTokenCreator` on it for your own account, which an
owner has) and run the Deploy job's commands by hand:

```bash
IMAGE=us-east1-docker.pkg.dev/$PROJECT/lastwar-ocr/lastwar-ocr-service@sha256:<digest>
gcloud run deploy lastwar-ocr-service --impersonate-service-account $DEPLOYER \
  --project $PROJECT --region $REGION --image $IMAGE --service-account $RUNTIME \
  --no-traffic --tag candidate --concurrency 1 --max-instances 3 \
  --memory 1Gi --cpu 1 --timeout 120
TOKEN=$(gcloud auth print-identity-token --impersonate-service-account $DEPLOYER \
  --audiences <service URL> --include-email)
curl -H "Authorization: Bearer $TOKEN" https://candidate---<service host>/health
curl -H "Authorization: Bearer $TOKEN" -F images=@tests/fixtures/smoke/weekly_frame.png \
  -F category=weekly https://candidate---<service host>/process-batch
gcloud run services update-traffic lastwar-ocr-service --impersonate-service-account $DEPLOYER \
  --project $PROJECT --region $REGION --to-latest
```

## Rotating the app's key

The app runs outside Google Cloud, so it needs a key. Rotate it yearly, and at
once on any suspicion it has leaked (a lost backup, a departed admin):

1. `gcloud iam service-accounts keys create new-key.json --iam-account lastwar-alliance-manager@$PROJECT.iam.gserviceaccount.com`
2. Upload `new-key.json` in the app: Admin → Security & API. Check an OCR upload
   and a translation still work.
3. `gcloud iam service-accounts keys list --iam-account …` and delete the old
   one with `gcloud iam service-accounts keys delete <KEY_ID> --iam-account …`.
4. Delete `new-key.json` from wherever you downloaded it.

## Billing: a budget alert is required

Everything the app can use bills the same account: Vision per image,
Translation per character (the app also caps this, `translation_monthly_char_cap`),
and Cloud Storage for the archive. Set a budget before the first deploy:
Console → Billing → Budgets & alerts → Create budget, scoped to the project,
with alerts at 50 %, 90 % and 100 % of a small amount (the project runs within
the free tiers, so a few dollars is plenty). A budget alerts; it does not stop
spending — the scale limit above is what caps it.

## Cloud Build leftovers

Releases build in GitHub Actions; nothing uses Cloud Build. An install that
used the old hand-built path (`gcloud builds submit`) has these to remove once
production runs a release:

- buckets `PROJECT_cloudbuild` and `run-sources-PROJECT-REGION`;
- the `cloud-run-source-deploy` repository (images from `gcloud run deploy --source`);
- optionally the Cloud Build API (`gcloud services disable cloudbuild.googleapis.com`).

Deleting an image breaks rolling back to a revision that uses it, so delete
them only once the current revision runs a release image.
