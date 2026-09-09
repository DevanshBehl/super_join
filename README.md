# Fact Knowledge Layer

Author: Devansh Behl

Extracts checkable facts from PDFs, grounds every one of them in a verbatim
quote with a page reference and character offsets, and then works out whether
facts from different documents corroborate each other, contradict each other,
or only appear to contradict because of a difference in period, scope, unit,
or data vintage.

Full documentation lives in [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Quickstart

```bash
git clone https://github.com/Mayan10/superjoin.git
cd superjoin
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then put your GEMINI_API_KEY in it
python scripts/ingest_all.py
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000.

## What it looks like

Four plain pages: upload and documents, a filterable fact table where every row
expands to its verbatim quote and page reference, a conflict inbox that shows
both facts side by side with the ordered rule trace that decided them, and a
stats page carrying the funnel counts and the quarantine.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for the architecture, the data model, the
four required cases with real evidence, and the demo guide.
