"""Inspect file manifests for plausible public classification datasets."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
IDS = [12947849,12947846,12096141,3812025,10754255,27556647,27551451,
       10864055,5348449,5083288,25459117,8028095,119261,129986,9551696,
       24072213,1498841,13651961,5348419]


def fetch(article_id):
    filename = ROOT / f"figshare_candidate_{article_id}.json"
    try:
        if filename.exists():
            data = json.loads(filename.read_text())
        else:
            with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article_id}", timeout=15) as r:
                data = json.load(r)
            filename.write_text(json.dumps(data, indent=2))
        return {"id": article_id, "title": data["title"],
                "files": [{"name":f["name"],"size":f["size"]} for f in data["files"]],
                "license": data.get("license"), "description": data["description"][:500]}
    except Exception as exc:
        return {"id":article_id,"error":str(exc)}


with ThreadPoolExecutor(max_workers=5) as pool:
    for future in as_completed([pool.submit(fetch, i) for i in IDS]):
        print(json.dumps(future.result()), flush=True)
