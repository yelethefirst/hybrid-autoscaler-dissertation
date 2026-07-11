# Measurement Environment as Code (§3.10-A)

This module provisions the dissertation's **measurement environment** — the AWS
EC2 host on which all final measured results are produced — so that the entire
host specification is reviewable, versioned, and recreatable with one command.
It implements Decision M.1 (`docs/IMPLEMENTATION_LOG.md`): **m6a.4xlarge**
(16 vCPU AMD EPYC / 64 GiB), eu-west-2, pinned Ubuntu 24.04 AMI, 100 GiB
encrypted gp3, SSH-only ingress, IMDSv2.

The committed defaults **are** the §3.10-A specification. Overriding any of
them for a measured run is a deviation → record it in `docs/deviations.md`.

## Workflow

```bash
cd infra/terraform

# 0. Note: `terraform validate`/`plan` need the provider plugin —
#    run `terraform init` first (resolves the committed .terraform.lock.hcl).

# 1. One-time setup
cp terraform.tfvars.example terraform.tfvars   # set allowed_ssh_cidr to <your-ip>/32
terraform init                                  # uses the committed .terraform.lock.hcl

# 2. Provision (T2.0)
terraform plan                                  # review: 3 resources
terraform apply

# 3. Environment build (T2.1) — Ansible from your Mac, no VM login needed
#   cd ../ansible && cp inventory.example.ini inventory.ini  (set the IP)
#   uvx --from ansible-core ansible-playbook -i inventory.ini provision.yml
#   then rsync the repo + `uv sync` (see ../ansible/README.md)

# 4. Between work sessions (compute billing stops; EBS ~$0.31/day continues)
$(terraform output -raw stop_command)
$(terraform output -raw start_command)          # note: public IP changes on restart

# 5. End of campaign — HARD DEADLINE ≤ 20 Jul 2026 (credit expires 23 Jul)
#    First archive results off the host (scp + git push), snapshot if wanted, then:
terraform destroy
```

## Cost model

| Item | Rate | Note |
|---|---|---|
| m6a.4xlarge running | $0.7992/h | $119.98 credit ≈ 150 h ≈ 6.2 days |
| 100 GiB gp3 (incl. stopped) | ≈ $0.31/day | negligible over the 2-week window |
| Stopped instance compute | $0 | stop between sessions |

On-demand only — **no Spot** (an interruption destroys a 5-hour collection
campaign for a saving the credit makes irrelevant).

## Design notes

- **AMI pinned, not looked up.** `ami-01bd674894e3ea876` was resolved once from
  Canonical's SSM parameter and dry-run-verified (2026-07-04); a live lookup
  would let the image drift between provisionings. `lifecycle.ignore_changes`
  guards against mid-campaign replacement.
- **SSH is the only ingress.** Locust and wrk2 run on the host against
  localhost:30080, so the cluster is never internet-exposed (§3.10-A
  co-location disclosure).
- **Heavy setup lives in `infra/ansible/provision.yml`,** not in user_data — the
  environment build is a reviewable, idempotent, checksum-pinned playbook;
  cloud-init only installs git/python and sets the hostname.
- **`.terraform.lock.hcl` is committed** so `init` resolves the identical
  provider build for any reader.
- **State is local and gitignored.** Single researcher, single host, two-week
  lifespan — remote state would add setup burden with no benefit. The state
  file contains resource IDs only (no secrets beyond the public key).

## What this module deliberately does NOT manage

- Instance run-state (stop/start) — use the output CLI commands; Terraform
  would try to "fix" a stopped instance on the next plan.
- The kind cluster, Online Boutique, observability — those are the artefact
  under study, owned by `infra/kind/`, `infra/online-boutique/` and
  `infra/observability/`, and reached via `infra/ansible/provision.yml`
  followed by `up.sh cloud`.
- IAM / account setup — the `argon` profile (IAM user `argon-admin`) is a
  pre-requisite, documented in `docs/IMPLEMENTATION_LOG.md`.
