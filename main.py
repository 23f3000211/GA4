import json, re, hashlib, os, math, struct, csv
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
# Q4 Data Generator — Python port of Fgenerate.js
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

        Q4_EMBEDDINGS = {
            doc_id: np.array(vector, dtype=np.float32)
            for doc_id, vector in embs.items()
        }

        Q4_RERANKER = reranker

        print(
            f"Q4 data generated for {config.EMAIL}: "
            f"{len(Q4_DOCS)} docs, "
            f"{len(Q4_EMBEDDINGS)} embeddings, "
            f"{len(Q4_RERANKER)} queries"
        )

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
# ================= Q3: /grounded-answer =================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "what", "who", "when", "where", "which", "how", "why",
    "does", "do", "did", "can", "could", "would", "should",
    "of", "in", "on", "at", "to", "for", "from", "by", "and",
    "or", "with", "about", "tell", "me", "please"
}

def normalize_text(text):
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower())


def tokens(text):
    return {
        word for word in normalize_text(text).split()
        if len(word) > 2 and word not in STOPWORDS
    }


def get_question_keywords(question):
    return tokens(question)


def chunk_supports_question(question, chunk_text):
    q = get_question_keywords(question)
    c = tokens(chunk_text)

    if not q or not c:
        return False, 0.0

    overlap = q & c
    overlap_ratio = len(overlap) / len(q)

    # Strong direct lexical support
    if overlap_ratio >= 0.5 and len(overlap) >= 2:
        return True, overlap_ratio

    # Handle questions where the key entity appears directly
    # Example: "What year was FAISS released?"
    # Chunk: "FAISS ... open-sourced in 2017."
    if len(overlap) >= 1 and overlap_ratio >= 0.34:
        return True, overlap_ratio

    return False, overlap_ratio


@app.post("/grounded-answer")
async def q3_answer(request: Request):
    try:
        body = await request.json()

        if not isinstance(body, dict):
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.1,
                "answerable": False
            }

        question = body.get("question")
        chunks = body.get("chunks")

        if not isinstance(question, str) or not question.strip():
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.1,
                "answerable": False
            }

        if not isinstance(chunks, list) or not chunks:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.1,
                "answerable": False
            }

        valid_chunks = [
            c for c in chunks
            if isinstance(c, dict)
            and isinstance(c.get("chunk_id"), str)
            and isinstance(c.get("text"), str)
            and c["chunk_id"].strip()
            and c["text"].strip()
        ]

        if not valid_chunks:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.1,
                "answerable": False
            }

        # Determine whether any provided chunk actually supports the question.
        supporting = []

        for chunk in valid_chunks:
            supported, score = chunk_supports_question(
                question,
                chunk["text"]
            )

            if supported:
                supporting.append((score, chunk))

        # IMPORTANT: deterministic answerability gate
        if not supporting:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.1,
                "answerable": False
            }

        # Rank supporting chunks by lexical support.
        supporting.sort(
            key=lambda x: (-x[0], x[1]["chunk_id"])
        )

        selected_chunks = [
            chunk for _, chunk in supporting[:5]
        ]

        # Only now ask the model to formulate the answer.
        prompt = (
            "You are a grounded question-answering system.\n"
            "Answer ONLY from the provided supporting chunks.\n"
            "Do not use outside knowledge.\n"
            "If the chunks do not contain enough information to answer "
            "the question, return answerable=false.\n\n"
            "Return strictly JSON with exactly these keys:\n"
            "{\n"
            '  "answer": "string",\n'
            '  "citations": ["chunk_id"],\n'
            '  "confidence": 0.0,\n'
            '  "answerable": true\n'
            "}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"SUPPORTING CHUNKS:\n"
            f"{json.dumps(selected_chunks, indent=2)}"
        )

        out = parse_json(
            await chat(
                [{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                max_tokens=500
            )
        )

        valid_ids = {
            c["chunk_id"] for c in valid_chunks
        }

        citations = [
            cid for cid in out.get("citations", [])
            if cid in valid_ids
        ]

        answer = str(out.get("answer", "")).strip()

        # If model fails to produce a grounded answer, fail safely.
        if (
            not answer
            or answer.lower() == "i don't know"
            or not citations
        ):
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.1,
                "answerable": False
            }

        confidence = float(out.get("confidence", 0.8))

        # Clamp confidence to valid range.
        confidence = max(0.0, min(1.0, confidence))

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "answerable": True
        }

    except Exception:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.1,
            "answerable": False
        }

# ================= Q4: /vector-search =================
def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def matches_filter(doc, filters):
    for key, condition in filters.items():

        if key not in doc:
            return False

        value = doc[key]

        if isinstance(condition, dict):

            if "gte" in condition:
                if value < condition["gte"]:
                    return False

            if "lte" in condition:
                if value > condition["lte"]:
                    return False

            if "in" in condition:
                if value not in condition["in"]:
                    return False

        else:
            if value != condition:
                return False

    return True

@app.post("/vector-search")
async def vector_search(request: Request):
    body = await request.json()

    query_id = body["query_id"]
    query_vector = np.array(
        body["query_vector"],
        dtype=np.float32
    )

    top_k = int(body["top_k"])
    rerank_top_n = int(body["rerank_top_n"])
    filters = body.get("filter", {})

    # 1. Filter documents first
    filtered_docs = [
        doc
        for doc in Q4_DOCS
        if matches_filter(doc, filters)
    ]

    # 2. Compute cosine similarity
    candidates = []

    for doc in filtered_docs:
        doc_id = doc["doc_id"]
        doc_vector = Q4_EMBEDDINGS[doc_id]

        similarity = cosine_sim(
            query_vector,
            doc_vector
        )

        candidates.append({
            "doc_id": doc_id,
            "similarity": similarity
        })

    # 3. Sort by similarity descending,
    #    then doc_id ascending
    candidates.sort(
        key=lambda x: (
            -x["similarity"],
            x["doc_id"]
        )
    )

    # 4. Keep only top_k for reranking
    top_k_docs = candidates[:top_k]

    # 5. Apply reranker scores only to retrieved docs
    query_scores = Q4_RERANKER.get(
        query_id,
        {}
    )

    for doc in top_k_docs:
        doc["rerank_score"] = rerank_scores[doc["doc_id"]]
        

    # 6. Sort by reranker score descending,
    #    then doc_id ascending
    top_k_docs.sort(
        key=lambda x: (
            -x["rerank_score"],
            x["doc_id"]
        )
    )

    # 7. Return final top rerank_top_n
    return {
        "matches": [
            doc["doc_id"]
            for doc in top_k_docs[:rerank_top_n]
        ]
    }

    # -----------------------------
    # 3. Retrieve top_k
    # Similarity descending
    # Tie-break doc_id ascending
    # -----------------------------
    candidates.sort(
        key=lambda x: (
            -x["similarity"],
            x["doc_id"]
        )
    )

    top_k_docs = candidates[:top_k]

    # -----------------------------
    # 4. Re-rank retrieved docs
    # -----------------------------
    rerank_scores = RERANKER_SCORES.get(
        query_id,
        {}
    )

    for doc in top_k_docs:
        doc["rerank_score"] = float(
            rerank_scores.get(
                doc["doc_id"],
                0.0
            )
        )

    # -----------------------------
    # 5. Final re-ranking
    # Score descending
    # Tie-break doc_id ascending
    # -----------------------------
    top_k_docs.sort(
        key=lambda x: (
            -x["rerank_score"],
            x["doc_id"]
        )
    )

    # -----------------------------
    # 6. Return top rerank_top_n
    # -----------------------------
    return {
        "matches": [
            doc["doc_id"]
            for doc in top_k_docs[:rerank_top_n]
        ]
    }
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
