import json, re, hashlib, os, math, struct
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
import config

# ============================================================
# Seedrandom (David Bau ARC4 PRNG) — Python port
# Produces identical output to the JS seedrandom library
# ============================================================

def _mixkey(seed: str, key: list):
    smear = 0
    j = 0
    mask = 0xFF
    while j < len(seed):
        idx = j & mask
        cur = key[idx] if idx < len(key) else 0
        smear = (smear ^ (cur * 19)) & 0xFFFFFFFF
        val = (smear + ord(seed[j])) & mask
        if idx < len(key):
            key[idx] = val
        else:
            key.append(val)
        j += 1

class SeededRng:
    def __init__(self, seed: str):
        key = []
        _mixkey(str(seed), key)
        keylen = len(key) or 1
        s = list(range(256))
        j = 0
        for i in range(256):
            t = s[i]
            j = (j + t + key[i % keylen]) & 0xFF
            s[i] = s[j]
            s[j] = t
        self._s = s
        self._i = 0
        self._j = 0
        self._g(256)  # RC4-drop[256]

    def _g(self, count: int) -> int:
        s = self._s
        i = self._i
        j = self._j
        r = 0
        while count > 0:
            count -= 1
            i = (i + 1) & 0xFF
            t = s[i]
            j = (j + t) & 0xFF
            si_new = s[j]
            s[j] = t
            s[i] = si_new
            r = r * 256 + s[(si_new + t) & 0xFF]
        self._i = i
        self._j = j
        return r

    def __call__(self) -> float:
        significance = 2 ** 52
        overflow = significance * 2
        startdenom = 256 ** 6
        n = self._g(6)
        d = startdenom
        x = 0
        while n < significance:
            n = (n + x) * 256
            d *= 256
            x = self._g(1)
        while n >= overflow:
            n //= 2
            d //= 2
            x >>= 1
        return (n + x) / d

def seedrandom(seed: str) -> SeededRng:
    return SeededRng(seed)

# ============================================================
# Q4 Data Generator — Python port of q4_generate.js
# ============================================================

WE = "tds-ga4-q4-data-74b0cb0ad988a5d60aa486353b85d4ff816446657b041c85"
CT = ["finance", "engineering", "marketing", "sales", "hr", "legal"]
LT = ["north_america", "europe", "asia_pacific", "latin_america"]

def generate_q4(email: str):
    email = email.strip().lower()
    rng = seedrandom(f"{WE}#{email}#q-vector-search-rerank-api#data")
    documents = []
    embeddings = {}
    for l in range(1, 501):
        doc_id = f"D{str(l).zfill(3)}"
        dept = CT[int(rng() * len(CT))]
        region = LT[int(rng() * len(LT))]
        year = 2020 + int(rng() * 7)
        documents.append({
            "doc_id": doc_id,
            "title": f"Document Title {doc_id} ({dept})",
            "department": dept,
            "year": year,
            "region": region,
            "text": f"This is the body text of document {doc_id} in department {dept} for region {region} and year {year}."
        })
        doc_rng = seedrandom(f"{WE}#{email}#q4#doc#{doc_id}")
        emb = [round(doc_rng() * 2 - 1, 4) for _ in range(100)]
        embeddings[doc_id] = emb
    reranker_scores = {}
    for l in range(1, 11):
        q_id = f"Q{str(l).zfill(3)}"
        q_rng = seedrandom(f"{WE}#{email}#q4#query#{q_id}")
        scores = {}
        for t in range(1, 501):
            d_id = f"D{str(t).zfill(3)}"
            scores[d_id] = round(q_rng(), 4)
        reranker_scores[q_id] = scores
    return documents, embeddings, reranker_scores

# ============================================================
# App Startup — generate Q4 data in memory from config.EMAIL
# ============================================================

Q4_DOCS = []
Q4_EMBEDDINGS = {}
Q4_RERANKER = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global Q4_DOCS, Q4_EMBEDDINGS, Q4_RERANKER
    try:
        docs, embs, reranker = generate_q4(config.EMAIL)
        Q4_DOCS = docs
        Q4_EMBEDDINGS = {k: np.array(v, dtype=np.float32) for k, v in embs.items()}
        Q4_RERANKER = reranker
        print(f"Q4 data generated for {config.EMAIL}: {len(Q4_DOCS)} docs.")
    except Exception as e:
        print(f"Failed to generate Q4 data: {e}")
    yield

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)

HEAD = {"Authorization": f"Bearer {config.AIPIPE_TOKEN}", "Content-Type": "application/json"}
_CACHE = {}

def _ck(*parts):
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()

import asyncio
async def chat(messages, model=None, max_tokens=800, force_json=True, retries=4):
    key = _ck("chat", model, json.dumps(messages, sort_keys=True, default=str))
    if key in _CACHE:
        return _CACHE[key]
    body = {"model": model or config.TEXT_MODEL, "messages": messages,
            "temperature": 0, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}
    last_err = None
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(retries):
            r = await c.post(f"{config.AIPIPE_BASE}/chat/completions",
                             headers=HEAD, json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:160]}"
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            _CACHE[key] = out
            return out
    raise RuntimeError(f"chat failed after {retries} retries: {last_err}")

def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"ok": True, "email": config.EMAIL}

# ================= Q3: /q3/answer =================
@app.post("/grounded-answer")
async def q3_answer(request: Request):
    body = await request.json()
    question = body.get("question", "")
    chunks = body.get("chunks", [])
    prompt = (
        "You are a highly reliable Grounded QA API for medical and legal compliance.\n"
        "Your task is to answer the user's question strictly using ONLY the provided context chunks.\n"
        "1. If the question CANNOT be answered from the chunks, you MUST return:\n"
        "   - answerable: false\n"
        "   - answer: \"I don't know\" (exact match)\n"
        "   - citations: [] (empty array)\n"
        "   - confidence: 0.1\n"
        "2. If it CAN be answered, return:\n"
        "   - answerable: true\n"
        "   - answer: <your grounded answer>\n"
        "   - citations: [<list of ONLY the chunk_ids you used>]\n"
        "   - confidence: <float between 0.8 and 1.0>\n"
        "NEVER use outside knowledge. Return strictly JSON with exactly these 4 keys.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CHUNKS:\n{json.dumps(chunks, indent=2)}"
    )
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], model="gpt-4o-mini", max_tokens=1000))
        if not out.get("answerable", False) or out.get("confidence", 1.0) <= 0.3:
            return {"answer": "I don't know", "citations": [], "confidence": 0.1, "answerable": False}
        valid_ids = [c["chunk_id"] for c in chunks]
        cites = [c for c in out.get("citations", []) if c in valid_ids]
        return {
            "answer": out.get("answer", "I don't know"),
            "citations": cites,
            "confidence": float(out.get("confidence", 0.9)),
            "answerable": True
        }
    except Exception:
        return {"answer": "I don't know", "citations": [], "confidence": 0.1, "answerable": False}

# ================= Q4: /vector-search =================
def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

@app.post("/vector-search")
async def vector_search(request: Request):
    body = await request.json()
    query_id = body.get("query_id")
    query_vector = np.array(body.get("query_vector", []), dtype=np.float32)
    top_k = body.get("top_k", 10)
    rerank_top_n = body.get("rerank_top_n", 3)
    filters = body.get("filter", {})
    # 1. Filter documents
    filtered_docs = []
    for doc in Q4_DOCS:
        match = True
        for key, condition in filters.items():
            if isinstance(condition, dict):
                if "gte" in condition and not (doc.get(key, 0) >= condition["gte"]):
                    match = False
                if "lte" in condition and not (doc.get(key, 0) <= condition["lte"]):
                    match = False
                if "in" in condition and doc.get(key) not in condition["in"]:
                    match = False
            else:
                if doc.get(key) != condition:
                    match = False
        if match:
            filtered_docs.append(doc)
    # 2. Cosine similarity
    scored_docs = []
    for doc in filtered_docs:
        doc_id = doc["doc_id"]
        doc_emb = Q4_EMBEDDINGS.get(doc_id)
        if doc_emb is not None:
            sim = cosine_sim(query_vector, doc_emb)
            scored_docs.append({"doc_id": doc_id, "sim": sim})
    # 3. Top-k (desc sim, tie-break lexicographic)
    scored_docs.sort(key=lambda x: (-x["sim"], x["doc_id"]))
    top_k_docs = scored_docs[:top_k]
    # 4. Re-rank
    rerank_scores = Q4_RERANKER.get(query_id, {})
    for doc in top_k_docs:
        doc["rerank_score"] = rerank_scores.get(doc["doc_id"], -999.0)
    top_k_docs.sort(key=lambda x: (-x["rerank_score"], x["doc_id"]))
    return {"matches": [d["doc_id"] for d in top_k_docs[:rerank_top_n]]}

# ================= Q5: GraphRAG Endpoints =================
@app.post("/extract-graph")
async def extract_graph(request: Request):
    body = await request.json()
    text = body.get("text", "")
    prompt = (
        "You are an expert GraphRAG Entity and Relationship extractor.\n"
        "Extract entities and relationships from the provided text according to these EXACT rules:\n"
        "Allowed Entity Types: Person, Organization, Product, Framework\n"
        "Allowed Relationship Types: FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, AUTHORED\n\n"
        "Return strictly JSON in this format:\n"
        "{\n"
        "  \"entities\": [{\"name\": \"Entity Name\", \"type\": \"AllowedType\"}],\n"
        "  \"relationships\": [{\"source\": \"Entity1\", \"target\": \"Entity2\", \"relation\": \"ALLOWED_RELATION\"}]\n"
        "}\n\n"
        f"TEXT:\n{text}"
    )
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], model="gpt-4o", max_tokens=1500))
        return {"entities": out.get("entities", []), "relationships": out.get("relationships", [])}
    except Exception:
        return {"entities": [], "relationships": []}

@app.post("/graph-query")
async def graph_query(request: Request):
    body = await request.json()
    question = body.get("question", "")
    graph = body.get("graph", {})
    prompt = (
        "You are a GraphRAG multi-hop reasoning agent.\n"
        "Given the knowledge graph provided (entities and relationships), answer the natural language question.\n"
        "You must determine the logical path through the graph to find the answer.\n"
        "Return strictly JSON in this format:\n"
        "{\n"
        "  \"answer\": \"Brief factual answer\",\n"
        "  \"reasoning_path\": [\"Entity1\", \"Entity2\", \"Entity3\"],\n"
        "  \"hops\": 2\n"
        "}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"GRAPH:\n{json.dumps(graph, indent=2)}"
    )
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], model="gpt-4o", max_tokens=1500))
        path = out.get("reasoning_path", [])
        return {"answer": out.get("answer", ""), "reasoning_path": path, "hops": len(path) - 1 if path else 0}
    except Exception:
        return {"answer": "", "reasoning_path": [], "hops": 0}

@app.post("/community-summary")
async def community_summary(request: Request):
    body = await request.json()
    community_id = body.get("community_id", "")
    entities = body.get("entities", [])
    relationships = body.get("relationships", [])
    prompt = (
        "You are a GraphRAG community summarizer. Summarize the following community of entities and relationships.\n"
        "The summary should be a concise paragraph explaining how these entities are connected and what their overall theme is.\n"
        "Return strictly JSON in this format:\n"
        "{\n"
        f"  \"community_id\": \"{community_id}\",\n"
        "  \"summary\": \"Your summary here.\"\n"
        "}\n\n"
        f"ENTITIES:\n{json.dumps(entities, indent=2)}\n\n"
        f"RELATIONSHIPS:\n{json.dumps(relationships, indent=2)}"
    )
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], model="gpt-4o", max_tokens=1500))
        return {"community_id": community_id, "summary": out.get("summary", "")}
    except Exception:
        return {"community_id": community_id, "summary": ""}
