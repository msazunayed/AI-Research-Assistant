import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Generator

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError, OpenAI
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("research_assistant")

# All of this is overridable from .env, see .env.example for the full list
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # your gsk_... (Groq) or xai-... (xAI) key
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")  # used for the final report only
FAST_MODEL_NAME = os.getenv("FAST_MODEL_NAME", "openai/gpt-oss-20b")  # query planning + summaries
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

MAX_QUERIES = int(os.getenv("MAX_QUERIES", "4"))
RESULTS_PER_QUERY = int(os.getenv("RESULTS_PER_QUERY", "4"))
SUMMARY_CONCURRENCY = int(os.getenv("SUMMARY_CONCURRENCY", "6"))  # parallel summarize calls
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))  # retries on top of the first attempt
LLM_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.5"))  # seconds, doubles each retry


# there is a per-request feature gap (see run_pipeline), not a boot blocker.
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your "
        "LLM provider's key before starting the server."
    )
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY is not set - /api/research will return an error until it is configured.")

llm = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
search_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

app = FastAPI(title="AI Research Assistant", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    # Last-resort safety net so a bug returns a clean 500 with a log line
    # instead of a raw traceback leaking to the client.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})


#LLM calls
# Three call sites below (queries, per-source summary, final report), all
# going through this one helper so there's a single place to tweak retries /
# logging / whatever later. model is overridable per-call - see the two
# constants above for why.

def ask_llm(system: str, user: str, max_tokens: int = 700, model: str = MODEL_NAME) -> str:
    last_error: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = llm.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        except APIError as e:
            last_error = e
            # Bad model name / bad request won't fix itself on retry - fail fast.
            if getattr(e, "status_code", None) and 400 <= e.status_code < 500 and e.status_code != 429:
                logger.error("LLM request to model '%s' failed (no retry): %s", model, e)
                raise
            if attempt < LLM_MAX_RETRIES:
                delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "LLM request to model '%s' failed (attempt %d/%d): %s - retrying in %.1fs",
                    model, attempt + 1, LLM_MAX_RETRIES + 1, e, delay,
                )
                time.sleep(delay)
    logger.error("LLM request to model '%s' failed after %d attempts: %s", model, LLM_MAX_RETRIES + 1, last_error)
    raise last_error


def generate_search_queries(topic: str) -> list[str]:
    raw = ask_llm(
        system=(
            "You turn a research topic into a set of focused, non-overlapping web "
            "search queries suitable for a professional research report - the kind "
            "an analyst would run, not a casual search. "
            f"Return ONLY a JSON array of {MAX_QUERIES} short query strings, "
            "no prose, no markdown fences."
        ),
        user=topic,
        max_tokens=150,
        model=FAST_MODEL_NAME,
    )
    # Smaller/local models love to wrap the array in a sentence or a code
    # fence no matter how nicely you ask, so just grab the first [...] we see
    # instead of trusting raw to be clean JSON.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        queries = json.loads(match.group(0)) if match else json.loads(raw)
        queries = [str(q) for q in queries][:MAX_QUERIES]
        return queries if queries else [topic]
    except Exception:
        # Worst case, just search the raw topic. Better than crashing the run.
        return [topic]


def summarize_source(topic: str, title: str, content: str) -> str:
    return ask_llm(
        system=(
            f"You are a research analyst extracting evidence relevant to '{topic}' "
            "from a single source, for inclusion in a formal briefing document. "
            "Produce 3-5 bullet points in a neutral, analytical register: concrete "
            "facts, figures, and dates only - no filler, no opinion, no restating "
            "the question. Preserve numbers, dates, and proper nouns exactly as "
            "given. Output plain bullet points with no preamble or closing remark."
        ),
        # Tavily's content field can be pretty long, truncate so we don't
        # blow the context window on one source.
        user=f"Title: {title}\n\nContent:\n{content[:6000]}",
        max_tokens=250,
        model=FAST_MODEL_NAME,
    )


def write_report(topic: str, sources: list[dict]) -> str:
    numbered = "\n\n".join(
        f"[{i+1}] {s['title']} ({s['url']})\n{s['summary']}"
        for i, s in enumerate(sources)
    )
    return ask_llm(
        system=(
            "You are a senior research analyst producing a formal written briefing "
            "for a professional audience. Write in a precise, measured, analytical "
            "register - no filler, no hedging clichés, no marketing tone, no first-"
            "person commentary. Base every claim strictly on the numbered source "
            "notes provided; do not speculate beyond what they support, and note "
            "explicitly where sources conflict or evidence is thin.\n\n"
            "Structure the report in Markdown exactly as follows, with no content "
            "before the first heading:\n"
            "1. '## Executive Summary' - 3-5 sentences giving the bottom line first, "
            "written so a reader could stop there and have the essential takeaway.\n"
            "2. 2-4 thematic '##' sections with descriptive headers, organizing "
            "findings by theme rather than walking through sources one by one.\n"
            "3. '## Limitations' - one short paragraph noting gaps, conflicting "
            "information, or areas that would benefit from further research. If "
            "none, state that the evidence was consistent across sources.\n"
            "4. '## Conclusion' - a short closing synthesis, not a repeat of the "
            "executive summary.\n"
            "5. '## Sources' - every numbered source listed as a Markdown link, in order.\n\n"
            "Cite claims inline like [1], [2] matching the source numbers given. "
            "Use Markdown tables for structured comparisons where that is clearer "
            "than prose. Do not restate the topic as a title - the document header "
            "is generated separately."
        ),
        user=f"Topic: {topic}\n\nSource notes:\n\n{numbered}",
        max_tokens=2000,
        model=MODEL_NAME,
    )


#streaming pipeline
# Frontend expects Server-Sent Events, so everything below just yields
# "event: x\ndata: {...}\n\n" strings and FastAPI streams them as they come.

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def run_pipeline(topic: str) -> Generator[str, None, None]:
    logger.info("Research run started: topic=%r", topic)

    if not search_client:
        yield sse("error", {"message": "TAVILY_API_KEY is not set on the server."})
        return

    yield sse("status", {"message": f"Planning research queries for \u201c{topic}\u201d\u2026"})
    try:
        queries = generate_search_queries(topic)
    except Exception as e:
        logger.exception("Query planning failed for topic=%r", topic)
        yield sse("error", {"message": f"Could not plan search queries: {e}"})
        return
    yield sse("queries", {"queries": queries})

    # Search first, summarize after. Tavily calls are quick network round
    # trips so we just do them one query at a time and collect every unique
    # result - the slow part was always the per-source LLM summary, and
    # that's what actually gets parallelized below.
    seen_urls: set[str] = set()
    to_summarize: list[dict] = []  # raw {title, url, content} waiting on a summary

    for q in queries:
        yield sse("status", {"message": f"Searching: {q}"})
        try:
            results = search_client.search(q, max_results=RESULTS_PER_QUERY)["results"]
        except Exception as e:
            # Don't kill the whole run over one bad query, just skip it and move on
            yield sse("status", {"message": f"Search failed for '{q}': {e}"})
            continue

        for r in results:
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            to_summarize.append({"title": r.get("title", url), "url": url, "content": r.get("content", "")})

    if not to_summarize:
        yield sse("error", {"message": "No sources found — try a different topic."})
        return

    yield sse("status", {"message": f"Reading {len(to_summarize)} sources\u2026"})

    # Fire off all the summaries at once instead of waiting on them one by
    # one - this is the main speedup. Groq (and most hosted providers) are
    # fine with a handful of concurrent requests; SUMMARY_CONCURRENCY caps
    # it so we don't hammer a free-tier rate limit.
    sources: list[dict] = []
    with ThreadPoolExecutor(max_workers=SUMMARY_CONCURRENCY) as pool:
        future_to_item = {
            pool.submit(summarize_source, topic, item["title"], item["content"]): item
            for item in to_summarize
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                summary = future.result()
            except Exception as e:
                summary = f"(couldn't summarize this source: {e})"
            source = {"title": item["title"], "url": item["url"], "summary": summary}
            sources.append(source)
            yield sse("source", source)

    yield sse("status", {"message": f"Synthesizing report from {len(sources)} sources\u2026"})
    try:
        report = write_report(topic, sources)
    except Exception as e:
        logger.exception("Report synthesis failed for topic=%r", topic)
        yield sse("error", {"message": f"Could not synthesize the final report: {e}"})
        return

    logger.info("Research run finished: topic=%r sources=%d", topic, len(sources))
    yield sse("report", {
        "markdown": report,
        "source_count": len(sources),
        # Official-document metadata, kept out of the LLM's hands so it's
        # exact and consistent - the frontend renders these in the header.
        "report_id": f"RR-{uuid.uuid4().hex[:8].upper()}",
        "generated_at": datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC"),
        "model": MODEL_NAME,
    })
    yield sse("done", {})


@app.get("/api/research")
def research(topic: str = Query(..., min_length=3, max_length=300)):
    return StreamingResponse(run_pipeline(topic), media_type="text/event-stream")


@app.get("/api/health")
def health():
    # frontend pings this on load to show a connected/offline dot, and it
    # doubles as a quick config sanity check - hit it after editing .env.
    return {
        "ok": True,
        "model": MODEL_NAME,
        "fast_model": FAST_MODEL_NAME,
        "llm_base_url": OPENAI_BASE_URL,
        "search_configured": bool(search_client),
    }
