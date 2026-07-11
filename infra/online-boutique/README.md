# Online Boutique deploy & pinning

The benchmark application is **Online Boutique** (GoogleCloudPlatform/microservices-demo), pinned to release **v0.10.5** (March 2026).

## Why we pin

§3.10 requires bit-exact reproducibility. Mutable tags like `:latest` or `:v0.10.5` can be re-pushed at the registry, changing the bytes under us. We resolve every image once to its **SHA256 content digest** and persist that.

## How

1. `install.sh` deploys the upstream v0.10.5 manifest unmodified — fastest path to a running cluster.
2. `pin-images.sh` then pulls each image, resolves its digest via `docker inspect`, and writes a pinned copy of the manifest to `pinned-manifest.yaml`. From that point on, re-deploy using the pinned manifest.

For the **final measured runs**, deploy from `pinned-manifest.yaml`. Commit `pinned-manifest.yaml` and `pinned-digests.txt` to version control — they are part of the §3.10 reproducibility package.

## Files

- `install.sh` — fetches upstream v0.10.5 release-bundle manifest and applies it.
- `pin-images.sh` — resolves and persists image digests; writes `pinned-manifest.yaml` and `pinned-digests.txt`.
- `pinned-manifest.yaml` — generated on first run of `pin-images.sh`. **Committed.**
- `pinned-digests.txt` — generated. Human-readable digest list for the dissertation §3.10 reproducibility package. **Committed.**

## Services deployed (11)

```
emailservice            recommendationservice    cartservice
checkoutservice         currencyservice          productcatalogservice
shippingservice         adservice                paymentservice
loadgenerator           frontend
```

> ℹ️ Online Boutique ships its own `loadgenerator` (a Locust container). For the controlled A/B experiments (Phase 5) we **disable that built-in load generator** and drive the system from an external Locust + wrk2 client on a separate host, per §3.9. For Phase 0 verification it is fine to leave it running.
