# Corpus Dataset Registry

`registry.yaml` is the legal and provenance gate for AI training sources.
External datasets stay disabled until their license, citation, redistribution
rights, source version, and checksum are reviewed.

Automatic download is intentionally stricter than curation:

- `license_status` must be `verified`.
- `enabled` must be `true`.
- `download_urls` must be present.
- Verified downloadable datasets must record a checksum.

Datasets marked `needs_review` may be inspected and described, but
`training/download_datasets.py` refuses to download them automatically.

Datasets marked `verified_restricted` have confirmed source/provenance but
license terms that block public-demo redistribution or automatic download.
They may still be normalized from local files for private training when the
source material is already present on the machine.
