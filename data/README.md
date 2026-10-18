# data/

Corpus and tenant data — **not committed** (see `.gitignore`).

## Structure

```
data/
  corpus/          OSCAL catalogs (fetched by scripts/fetch_corpus.sh)
    oscal/           usnistgov/oscal-content
  tenants/         Synthetic tenant configs (invented — see src/rag_control_cite/tenants.py)
```

## What is wired

`src/rag_control_cite/ingestion.py` ingests exactly three OSCAL catalogs:
**SP 800-53 rev5** (with the Low/Moderate/High/privacy baseline profiles),
**SP 800-171 rev3**, and **CSF v2.0**. FedRAMP Rev5, a real-PDF slice
(HIPAA/CMMC/PCI), and a finance slice (FFIEC/SEC/PCI) are **staged, not
wired** — they are not fetched and not parsed.

## Fetching

```bash
just sync
bash scripts/fetch_corpus.sh     # downloads OSCAL content from usnistgov/oscal-content
```

OSCAL content is public domain (US Government work). Non-US sources
(EBA, BaFin) need a licence check before redistribution — flag, do
not assume.
