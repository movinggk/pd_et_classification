"""Read-only public-source metadata collection. No full signal archives."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "digital_assessment_figshare.json": "https://api.figshare.com/v2/articles/5969692",
    "openneuro_participants.json": "https://api.github.com/repos/OpenNeuroDatasets/ds008336/contents/participants.tsv",
    "openneuro_description.json": "https://api.github.com/repos/OpenNeuroDatasets/ds008336/contents/dataset_description.json",
    "harvard_cognitive.json": "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/ZVPGDV",
    "mendeley_et_rtms.json": "https://data.mendeley.com/public-api/datasets/tt8j3bhy5w/files?folder_id=root&version=1&$start=0&$limit=1000",
    "mendeley_vim_model.json": "https://data.mendeley.com/public-api/datasets/ktrts7r2hk/files?folder_id=root&version=1&$start=0&$limit=1000",
    "tremoroton_mmc3.docx": "https://pmc.ncbi.nlm.nih.gov/articles/instance/6961062/bin/mmc3.docx",
    "geo_rnaseq.html": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134878",
    "geo_methylation.html": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134426",
}


def fetch(item):
    name, url = item
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Public-dataset-research/1.0"})
        with urllib.request.urlopen(req, timeout=20) as response:
            body = response.read(10_000_000)
            final_url = response.url
            content_type = response.headers.get("Content-Type")
        (ROOT / name).write_bytes(body)
        return {"file": name, "url": url, "final_url": final_url,
                "bytes": len(body), "content_type": content_type, "ok": True}
    except Exception as exc:
        return {"file": name, "url": url, "ok": False, "error": str(exc)}


if __name__ == "__main__":
    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in SOURCES.items()]):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    (ROOT / "additional_fetch_log.json").write_text(json.dumps(results, indent=2))
