import json, re
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from pydantic import BaseModel, Field
from .base import BaseRAG


class POI(BaseModel):
    id: str
    name: str
    category: str
    address: str
    services: list[str] | None = Field(default_factory=list)
    description: str | None = ""
    rating: float | None = None
    review_count: int | None = None
    lat: float | None = None
    long: float | None = None


class POIRAG(BaseRAG):
    domain = "poi"

    def _load_docs(self):
        poi_file = self.data_dir / "poi.json"
        docs: list[Document] = []

        with poi_file.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"[poi.json] Failed to parse JSON on line {lineno}: {e}"
                    ) from e

                poi = POI(**raw)
                content = (
                    f"{poi.name}（{poi.category}）\n"
                    f"{poi.description or ''}\n"
                    f"address：{poi.address}\n"
                    f"rating：{poi.rating or 'N/A'}"
                )

                # ---------- metadata ----------
                raw_meta = poi.dict()
                meta_ok = {
                    k: (", ".join(v) if isinstance(v, list) else v) for k, v in raw_meta.items()
                }

                docs.append(
                    Document(
                        page_content=content,
                        metadata=meta_ok,
                    )
                )
        return docs

    GEO_RE = re.compile(r"(附近|哪裡|多遠|close|near)", re.I)

    async def aretrieve(self, query: str, k: int = 6, user_loc=None):
        """覆寫：可加地理距離過濾."""
        results = await super().aretrieve(query, k)
        if user_loc:
            # rudimentary distance filter; 可改用 haversine
            lat0, lon0 = user_loc
            results = sorted(
                results,
                key=lambda d: _dist(lat0, lon0, d.metadata.get("lat"), d.metadata.get("long")),
            )[:k]
        return results


def _dist(lat0, lon0, lat1, lon1):
    if None in (lat1, lon1):
        return 9e9
    return abs(lat0 - lat1) + abs(lon0 - lon1)
