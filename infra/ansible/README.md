# Measurement VM Provisioning (Ansible)

Declarative, idempotent, checksum-pinned environment build for the §3.10-A
host — supersedes the originally planned `bootstrap-ubuntu.sh` (decision
2026-07-05: re-run safe, diffable, and runs from the researcher's machine over
SSH; the VM needs nothing pre-installed, and neither does the Mac thanks to
`uvx`).

## Workflow

```bash
cd infra/ansible
cp inventory.example.ini inventory.ini      # set VM IP from `terraform output public_ip`

# dry-run first (shows every change it WOULD make)
uvx --from ansible-core ansible-playbook -i inventory.ini provision.yml --check --diff

# provision (idempotent — safe to re-run any time, e.g. after IP change)
uvx --from ansible-core ansible-playbook -i inventory.ini provision.yml

# ship the repo (committed tree + .git for the per-trial SHA guard),
# then install Python deps on the VM
rsync -az --exclude .venv --exclude .pytest_cache --exclude .ruff_cache \
  --exclude 'data/parquet' -e "ssh -i ~/.ssh/id_ed25519" \
  ../../ ubuntu@<VM_IP>:~/hybrid-autoscaler/
ssh -i ~/.ssh/id_ed25519 ubuntu@<VM_IP> \
  'cd ~/hybrid-autoscaler && ~/.local/bin/uv sync && ~/.local/bin/uv run pytest -q'
```

Then continue with the campaign sequence: `infra/kind/up.sh cloud` →
`infra/online-boutique/install.sh` → `infra/observability/install.sh` →
`infra/verify.sh` (T2.4).

## What it pins (== §3.10-A; changing any value is a deviation)

| Component | Version | Integrity |
|---|---|---|
| Docker Engine | 26.x (official apt repo) | apt-signed |
| kind | v0.32.0 | sha256-pinned binary |
| kubectl | v1.33.12 (matches node image; inside ±1 skew) | sha256-pinned |
| helm | v3.16.4 | sha256-pinned tarball |
| uv | 0.8.24 (matches dev Mac) | versioned installer |
| wrk2 | giltene/wrk2 @ 44a94c17 (built from source, installed as `wrk2`) | commit-pinned |
| tuning | port range, somaxconn, nofile 65535 | declarative files |

`hey` is deliberately NOT installed: the supervisor requires wrk2 on the
measurement host and hey fallback is opt-in for dev only (ALLOW_HEY_FALLBACK).

## Notes

- `inventory.ini` is gitignored (contains your VM IP); the example is committed.
- Tags allow partial runs, e.g. `--tags wrk2,verify`.
- The `verify` play is the exit criterion: it prints every tool version and
  fails if any binary is missing.
