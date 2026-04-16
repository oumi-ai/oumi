from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..dependencies import get_github
from ..github.client import GitHubClient, GitHubRateLimitError

router = APIRouter()

# ── Company / segment helpers ─────────────────────────────────────────────────

BIG_TECH = {
    "google", "alphabet", "microsoft", "amazon", "aws", "meta", "facebook",
    "apple", "netflix", "nvidia", "salesforce", "oracle", "ibm", "intel",
    "amd", "twitter", "x corp", "linkedin", "uber", "airbnb", "stripe",
    "shopify", "atlassian", "slack", "github", "gitlab", "databricks",
    "snowflake", "palantir", "openai", "anthropic", "deepmind", "hugging face",
    "huggingface", "bytedance", "tiktok", "samsung", "adobe", "vmware", "cisco",
    "qualcomm", "bloomberg", "jpmorgan", "deloitte", "accenture", "mckinsey",
}

ACADEMIC_KW = {
    "university", "université", "universidade", "universidad", "universität",
    "college", "institute", "lab ", "labs", "laboratory", "research",
    "polytechnic", "mit", "stanford", "cmu", "berkeley", "oxford",
    "cambridge", "eth ", "epfl", "inria", "mpi",
}

SEGMENT_LABELS = {
    "enterprise": "Enterprise",
    "startup": "Startup / Scale-up",
    "academic": "Academic / Research",
    "independent": "Independent",
    "unknown": "Unknown",
}

SEGMENT_COLORS = {
    "enterprise": "rgba(59,130,246,0.75)",
    "startup": "rgba(139,92,246,0.75)",
    "academic": "rgba(16,185,129,0.75)",
    "independent": "rgba(245,158,11,0.75)",
    "unknown": "rgba(156,163,175,0.75)",
}


def normalize_company(raw: str | None) -> str | None:
    if not raw:
        return None
    c = raw.strip().lstrip("@").strip()
    for suffix in [", Inc.", " Inc.", " LLC", " Ltd.", " Corp.", " GmbH", ", Inc", " Co.", " Co"]:
        if c.lower().endswith(suffix.lower()):
            c = c[: -len(suffix)].strip()
    return c if c else None


def segment_stargazer(profile: dict) -> str:
    company = (profile.get("company") or "").lower().lstrip("@").strip()
    bio = (profile.get("bio") or "").lower()
    combined = f"{company} {bio}"

    if any(t in company for t in BIG_TECH):
        return "enterprise"
    if any(kw in combined for kw in ACADEMIC_KW):
        return "academic"
    if company and len(company) > 1:
        return "startup"
    if any(kw in bio for kw in ["freelance", "independent", "consultant", "self-employed", "solo dev"]):
        return "independent"
    return "unknown"


def icp_score(profile: dict) -> int:
    score = 0
    followers = profile.get("followers", 0)
    if profile.get("company"):
        score += 20
    if followers >= 1000:
        score += 30
    elif followers >= 100:
        score += 20
    elif followers >= 10:
        score += 8
    if profile.get("location"):
        score += 5
    if profile.get("bio"):
        score += 5
    repos = profile.get("public_repos", 0)
    if repos >= 100:
        score += 15
    elif repos >= 20:
        score += 8
    return score


# ── Velocity helpers ──────────────────────────────────────────────────────────

def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def compute_velocity(events: list[dict]) -> dict:
    if not events:
        return {}

    timestamps = sorted(_parse_ts(e["starred_at"]) for e in events)
    start = timestamps[0].replace(hour=0, minute=0, second=0, microsecond=0)
    end = timestamps[-1].replace(hour=0, minute=0, second=0, microsecond=0)

    # Decide granularity
    span_days = (end - start).days + 1
    if span_days > 180:
        granularity = "week"
    else:
        granularity = "day"

    # Bucket counts
    bucket: dict[str, int] = defaultdict(int)
    for ts in timestamps:
        if granularity == "week":
            # Monday of that week
            day = ts.date() - timedelta(days=ts.weekday())
            bucket[day.isoformat()] += 1
        else:
            bucket[ts.date().isoformat()] += 1

    # Build contiguous label list
    labels: list[str] = []
    counts: list[int] = []
    current = start.date()
    step = timedelta(weeks=1) if granularity == "week" else timedelta(days=1)
    while current <= end.date():
        labels.append(current.isoformat())
        counts.append(bucket.get(current.isoformat(), 0))
        current += step

    # Cumulative
    cum = 0
    cumulative: list[int] = []
    for c in counts:
        cum += c
        cumulative.append(cum)

    # Rolling average (7 periods)
    window = 7
    rolling: list[float] = []
    for i in range(len(counts)):
        w = counts[max(0, i - window + 1): i + 1]
        rolling.append(round(sum(w) / len(w), 1))

    # Find top-5 spike periods
    indexed = sorted(enumerate(counts), key=lambda x: x[1], reverse=True)
    spikes = [{"label": labels[i], "count": c} for i, c in indexed[:5] if c > 0]

    return {
        "labels": labels,
        "counts": counts,
        "cumulative": cumulative,
        "rolling": rolling,
        "granularity": granularity,
        "spikes": spikes,
        "total": len(timestamps),
    }


def velocity_period_stats(events: list[dict]) -> dict:
    now = datetime.now(timezone.utc)
    ts_list = [_parse_ts(e["starred_at"]) for e in events]

    def count_in(days: int) -> int:
        cutoff = now - timedelta(days=days)
        return sum(1 for t in ts_list if t >= cutoff)

    last7 = count_in(7)
    prev7 = count_in(14) - last7
    last30 = count_in(30)
    prev30 = count_in(60) - last30

    def pct_change(curr: int, prev: int) -> float | None:
        if prev == 0:
            return None
        return round((curr - prev) / prev * 100, 1)

    return {
        "last_7d": last7,
        "prev_7d": prev7,
        "pct_7d": pct_change(last7, prev7),
        "last_30d": last30,
        "prev_30d": prev30,
        "pct_30d": pct_change(last30, prev30),
    }


# ── Enrichment analysis ───────────────────────────────────────────────────────

def analyze_enriched(profiles: list[dict]) -> dict:
    company_counts: Counter = Counter()
    location_counts: Counter = Counter()
    segment_counts: Counter = Counter()

    for p in profiles:
        co = normalize_company(p.get("company"))
        if co:
            company_counts[co] += 1

        loc = (p.get("location") or "").strip()
        if loc:
            # Simplify: take last comma-separated token as country proxy
            parts = [x.strip() for x in loc.split(",")]
            location_counts[parts[-1]] += 1

        segment_counts[segment_stargazer(p)] += 1

    # Multi-star companies (team adoption)
    multi_star = sorted(
        [(co, cnt) for co, cnt in company_counts.items() if cnt >= 2],
        key=lambda x: x[1],
        reverse=True,
    )

    # Influencers: sorted by followers
    influencers = sorted(
        [p for p in profiles if p.get("followers", 0) > 0],
        key=lambda x: x.get("followers", 0),
        reverse=True,
    )

    # Total reach: sum of top-100 stargazers' followers
    total_reach = sum(p.get("followers", 0) for p in influencers[:100])

    # Hot prospects: recent + ICP score > threshold
    now = datetime.now(timezone.utc)
    cutoff_30d = now - timedelta(days=30)
    hot = []
    for p in profiles:
        try:
            starred = _parse_ts(p["starred_at"])
        except Exception:
            continue
        score = icp_score(p)
        segment = segment_stargazer(p)
        hot.append({**p, "icp_score": score, "segment": segment, "is_recent": starred >= cutoff_30d})

    hot.sort(key=lambda x: (x["is_recent"], x["icp_score"]), reverse=True)

    # Scatter data for "influence map"
    scatter_by_segment: dict[str, list[dict]] = defaultdict(list)
    for p in profiles:
        f = p.get("followers", 0)
        r = p.get("public_repos", 0)
        if f == 0 and r == 0:
            continue
        seg = segment_stargazer(p)
        scatter_by_segment[seg].append({
            "x": round(math.log10(max(f, 1)), 2),
            "y": round(math.log10(max(r, 1)), 2),
            "login": p.get("login", ""),
            "company": normalize_company(p.get("company")) or "",
            "followers": f,
            "repos": r,
            "score": icp_score(p),
            "avatar": p.get("avatar_url", ""),
        })

    top_companies = company_counts.most_common(15)
    top_locations = location_counts.most_common(12)

    return {
        "top_companies": top_companies,
        "multi_star": multi_star[:10],
        "top_locations": top_locations,
        "segment_counts": dict(segment_counts),
        "influencers": influencers[:30],
        "total_reach": total_reach,
        "hot_prospects": hot[:40],
        "scatter_by_segment": dict(scatter_by_segment),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/stargazers", response_class=HTMLResponse)
async def stargazers(request: Request, gh: GitHubClient = Depends(get_github)):
    templates = request.app.state.templates
    try:
        profiles = await gh.get_recent_enriched_stargazers(n=200)
    except GitHubRateLimitError as e:
        return templates.TemplateResponse(request, "pages/stargazers.html", {
            "active": "stargazers", "rate_limit_error": True,
            "rate_limit_reset": e.reset_at, "analysis": None,
            "sample_size": 0, "company_chart_json": "{}",
            "location_chart_json": "{}", "segment_chart_json": "{}",
            "scatter_datasets_json": "[]",
        })
    analysis = analyze_enriched(profiles)

    # Chart.js data blobs (serialized once here, not in template)
    company_chart = {
        "labels": [c for c, _ in analysis["top_companies"]],
        "data": [n for _, n in analysis["top_companies"]],
    }
    location_chart = {
        "labels": [l for l, _ in analysis["top_locations"]],
        "data": [n for _, n in analysis["top_locations"]],
    }
    seg_labels = [SEGMENT_LABELS.get(k, k) for k in analysis["segment_counts"]]
    seg_data = list(analysis["segment_counts"].values())
    seg_colors = [SEGMENT_COLORS.get(k, "#9ca3af") for k in analysis["segment_counts"]]
    segment_chart = {"labels": seg_labels, "data": seg_data, "colors": seg_colors}

    scatter_datasets = [
        {
            "label": SEGMENT_LABELS.get(seg, seg),
            "data": pts,
            "backgroundColor": SEGMENT_COLORS.get(seg, "#9ca3af"),
        }
        for seg, pts in analysis["scatter_by_segment"].items()
    ]

    return templates.TemplateResponse(
        request,
        "pages/stargazers.html",
        {
            "active": "stargazers",
            "analysis": analysis,
            "company_chart_json": json.dumps(company_chart),
            "location_chart_json": json.dumps(location_chart),
            "segment_chart_json": json.dumps(segment_chart),
            "scatter_datasets_json": json.dumps(scatter_datasets),
            "sample_size": len(profiles),
        },
    )


@router.get("/stargazers/velocity", response_class=HTMLResponse)
async def stargazers_velocity(request: Request, gh: GitHubClient = Depends(get_github)):
    """HTMX lazy-loaded velocity chart — may be slow on first call (fetches all timestamps)."""
    templates = request.app.state.templates
    try:
        events = await gh.get_all_star_timestamps()
    except GitHubRateLimitError as e:
        return HTMLResponse(
            f'<div class="rounded-lg border border-orange-200 bg-orange-50 p-5 text-sm text-orange-800">'
            f'GitHub rate limit exceeded. Resets at {e.reset_at.strftime("%H:%M")}. '
            f'Add a <code>GITHUB_TOKEN</code> to <code>.env</code> for 5,000 req/hr.</div>'
        )
    velocity = compute_velocity(events)
    period_stats = velocity_period_stats(events)

    return templates.TemplateResponse(
        request,
        "partials/velocity_chart.html",
        {
            "velocity_json": json.dumps(velocity),
            "period_stats": period_stats,
            "total_stars": velocity.get("total", 0),
        },
    )
