"""Paginate the public Figshare article search; save each response."""
from pathlib import Path
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
query = '"essential tremor"'
all_records = []
complete = False
error = None
for page in range(1, 12):
    cached = ROOT / f"figshare_phrase_page{page}.json"
    if cached.exists():
        records = json.loads(cached.read_text())
    else:
        payload = json.dumps({"search_for": query, "page_size": 100, "page": page,
                              "order": "published_date", "order_direction": "desc"}).encode()
        req = urllib.request.Request("https://api.figshare.com/v2/articles/search", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                records = json.load(response)
        except Exception as exc:
            error = str(exc)
            print(error, flush=True)
            break
        cached.write_text(json.dumps(records, indent=2))
    all_records.extend(records)
    print(f"page={page} records={len(records)}", flush=True)
    if len(records) < 100:
        complete = True
        break
unique = {r["id"]: r for r in all_records}
summary = {"query": query, "endpoint": "https://api.figshare.com/v2/articles/search",
           "pages": page, "records": len(all_records), "unique_ids": len(unique),
           "pagination_complete": complete, "error": error}
(ROOT / "figshare_search_summary.json").write_text(json.dumps(summary, indent=2))
for r in unique.values():
    title = r["title"].lower()
    if "parkinson" in title or any(w in title for w in ["dataset", "raw", "cognitive functioning"]):
        print(r["id"], r["title"], flush=True)
