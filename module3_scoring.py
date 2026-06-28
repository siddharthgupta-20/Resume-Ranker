"""
Module 3: Scoring Engine
------------------------
Takes outputs from Module 2 (embeddings, taxonomy) and produces
a final ranked list of top 100 candidates for a given JD.

Scoring dimensions:
    1. JD Fit Score        (50%) — semantic similarity + skill matching
    2. Hirability Score    (30%) — platform signals, responsiveness, logistics
    3. Intent & Quality    (20%) — activity, education tier, market validation

Disqualifier layer applies hard penalties before final ranking.

Usage:
    from module3_scoring import ScoringEngine
    engine = ScoringEngine(parsed_jd, jd_embedding, candidate_embeddings)
    top_100 = engine.rank(candidates)
"""

import json
import math
from datetime import date, datetime
from numpy import dot
from numpy.linalg import norm


# ── Constants ─────────────────────────────────────────────────────────────────

WEIGHTS = {
    "jd_fit":         0.50,
    "hirability":     0.30,
    "intent_quality": 0.20,
}

# Consulting firms from JD disqualifying signals
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl",
    "tech mahindra", "mphasis", "hexaware"
}

# Wrong domain keywords from JD disqualifying signals
WRONG_DOMAIN_KEYWORDS = [
    "computer vision", "cv engineer", "speech recognition",
    "robotics", "autonomous", "image processing", "object detection"
]

# Education tier score mapping
TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.40
}

# Skill proficiency score mapping
PROFICIENCY_SCORES = {
    "expert":       1.0,
    "advanced":     0.75,
    "intermediate": 0.50,
    "beginner":     0.25
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = vec_a, vec_b
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(dot(a, b) / denom)


def days_since(date_str: str) -> int:
    """Return number of days between a date string and today."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 999


def min_max_normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def safe_get(d: dict, *keys, default=0):
    """Safely get nested dict value."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d if d is not None else default


# ── Scoring Engine ────────────────────────────────────────────────────────────

class ScoringEngine:
    def __init__(
        self,
        parsed_jd: dict,
        jd_embedding: list[float],
        candidate_embeddings: dict[str, list[float]]
    ):
        self.jd = parsed_jd
        self.jd_embedding = jd_embedding
        self.candidate_embeddings = candidate_embeddings

        # Precompute JD skill sets for fast lookup
        self.required_skills_lower = {
            s.lower() for s in parsed_jd.get("required_skills", [])
        }
        self.preferred_skills_lower = {
            s.lower() for s in parsed_jd.get("preferred_skills", [])
        }
        self.locations_accepted_lower = {
            loc.lower() for loc in parsed_jd.get("locations_accepted", [])
        }

    # ── Disqualifier Layer ────────────────────────────────────────────────────

    def get_disqualifier_multiplier(self, candidate: dict) -> tuple[float, list[str]]:
        """
        Check candidate against JD disqualifying signals.

        Returns:
            (multiplier, list of triggered disqualifiers)
            multiplier = 1.0 means no penalty
            multiplier < 1.0 means penalized
        """
        flags = []
        multiplier = 1.0

        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])
        signals = candidate.get("redrob_signals", {})

        # --- Check 1: Entire career at consulting firms ---
        if career:
            companies = [job.get("company", "").lower() for job in career]
            consulting_count = sum(
                1 for c in companies
                if any(firm in c for firm in CONSULTING_FIRMS)
            )
            if consulting_count == len(companies):
                flags.append("Entire career at consulting firms")
                multiplier *= 0.2

        # --- Check 2: Wrong domain (CV/speech/robotics without NLP) ---
        headline = profile.get("headline", "").lower()
        summary = profile.get("summary", "").lower()
        current_title = profile.get("current_title", "").lower()
        combined_text = f"{headline} {summary} {current_title}"

        nlp_keywords = ["nlp", "retrieval", "search", "ranking", "recommendation", "ir ", "information retrieval"]
        has_nlp = any(kw in combined_text for kw in nlp_keywords)
        has_wrong_domain = any(kw in combined_text for kw in WRONG_DOMAIN_KEYWORDS)

        if has_wrong_domain and not has_nlp:
            flags.append("Wrong domain: CV/speech/robotics without NLP/IR")
            multiplier *= 0.3

        # --- Check 3: No recent activity (proxy for no production code) ---
        last_active_days = days_since(signals.get("last_active_date", "2000-01-01"))
        if last_active_days > 540:  # ~18 months
            flags.append("Inactive for 18+ months")
            multiplier *= 0.5

        # --- Check 4: Experience too low ---
        years_exp = profile.get("years_of_experience", 0)
        min_exp = self.jd.get("min_experience_years", 0)
        if years_exp < min_exp - 1:  # allow 1 year grace
            flags.append(f"Under-experienced: {years_exp} vs {min_exp} required")
            multiplier *= 0.6

        return multiplier, flags

    # ── Sub-scorer 1: JD Fit ──────────────────────────────────────────────────

    def score_jd_fit(self, candidate: dict) -> float:
        """
        Score how well candidate matches the JD.
        Combines:
          - Semantic similarity (embedding cosine)
          - Required skill coverage
          - Skill proficiency depth
          - Verified assessment scores
          - Experience range fit
        """
        cid = candidate["candidate_id"]
        candidate_vec = self.candidate_embeddings.get(cid)

        # --- Semantic similarity (40% of JD fit) ---
        if candidate_vec:
            semantic_score = cosine_similarity(self.jd_embedding, candidate_vec)
            # Cosine similarity for good matches is usually 0.5-0.95
            # Normalize to 0-1 assuming range 0.3 to 0.95
            semantic_score = min_max_normalize(semantic_score, 0.3, 0.95)
        else:
            semantic_score = 0.0

        # --- Required skill coverage (35% of JD fit) ---
        candidate_skills = candidate.get("skills", [])
        candidate_skill_names = {s["name"].lower() for s in candidate_skills}

        # Build proficiency map for candidate
        proficiency_map = {
            s["name"].lower(): PROFICIENCY_SCORES.get(s.get("proficiency", "beginner"), 0.25)
            for s in candidate_skills
        }

        required_hits = []
        for req_skill in self.required_skills_lower:
            # Exact match
            if req_skill in candidate_skill_names:
                prof = proficiency_map.get(req_skill, 0.5)
                required_hits.append(prof)
            else:
                # Partial match (e.g. "faiss" in "vector databases (pinecone/faiss)")
                partial = any(
                    req_skill in cand_skill or cand_skill in req_skill
                    for cand_skill in candidate_skill_names
                )
                if partial:
                    required_hits.append(0.4)  # partial credit
                else:
                    required_hits.append(0.0)

        skill_coverage = sum(required_hits) / len(self.required_skills_lower) if self.required_skills_lower else 0.0

        # --- Preferred skill bonus (10% of JD fit) ---
        preferred_hits = sum(
            1 for ps in self.preferred_skills_lower
            if ps in candidate_skill_names or any(
                ps in cs or cs in ps for cs in candidate_skill_names
            )
        )
        preferred_score = preferred_hits / len(self.preferred_skills_lower) if self.preferred_skills_lower else 0.0

        # --- Verified assessment scores (15% of JD fit) ---
        assessment_scores = safe_get(candidate, "redrob_signals", "skill_assessment_scores", default={})
        if assessment_scores:
            # Average of top 5 assessment scores, normalized to 0-1
            top_scores = sorted(assessment_scores.values(), reverse=True)[:5]
            assessment_score = sum(top_scores) / (len(top_scores) * 100)
        else:
            assessment_score = 0.3  # neutral if no assessments

        # --- Experience fit ---
        years_exp = candidate["profile"].get("years_of_experience", 0)
        min_exp = self.jd.get("min_experience_years", 0)
        max_exp = self.jd.get("max_experience_years") or min_exp + 10

        if min_exp <= years_exp <= max_exp:
            exp_score = 1.0
        elif years_exp < min_exp:
            exp_score = max(0.0, 1.0 - (min_exp - years_exp) * 0.15)
        else:
            exp_score = max(0.5, 1.0 - (years_exp - max_exp) * 0.05)

        # --- Weighted combination ---
        jd_fit = (
            0.40 * semantic_score +
            0.35 * skill_coverage +
            0.10 * preferred_score +
            0.15 * assessment_score
        )

        # Apply experience as a multiplier (not additive)
        jd_fit *= (0.7 + 0.3 * exp_score)

        return round(min(1.0, jd_fit), 4)

    # ── Sub-scorer 2: Hirability ──────────────────────────────────────────────

    def score_hirability(self, candidate: dict) -> float:
        """
        Score how easy it is to hire this candidate.
        Based purely on redrob_signals.
        """
        signals = candidate.get("redrob_signals", {})

        # --- Interview completion rate (25%) ---
        interview_rate = safe_get(signals, "interview_completion_rate", default=0.5)

        # --- Offer acceptance rate (20%) ---
        offer_rate = safe_get(signals, "offer_acceptance_rate", default=-1)
        if offer_rate == -1:
            offer_score = 0.5  # no history → neutral
        else:
            offer_score = offer_rate

        # --- Recruiter response rate (20%) ---
        response_rate = safe_get(signals, "recruiter_response_rate", default=0.5)

        # --- Response time (15%) — faster is better ---
        avg_response_hours = safe_get(signals, "avg_response_time_hours", default=48)
        # Normalize: 0 hours = 1.0, 72+ hours = 0.0
        response_time_score = max(0.0, 1.0 - (avg_response_hours / 72))

        # --- Notice period (10%) ---
        notice_days = safe_get(signals, "notice_period_days", default=90)
        preferred_notice = self.jd.get("notice_period_preference_days", 30)
        max_notice = self.jd.get("notice_period_max_days", 90)

        if notice_days <= preferred_notice:
            notice_score = 1.0
        elif notice_days <= max_notice:
            notice_score = 0.6
        else:
            notice_score = 0.2  # exceeds max acceptable

        # --- Location fit (5%) ---
        candidate_location = candidate["profile"].get("location", "").lower()
        willing_to_relocate = safe_get(signals, "willing_to_relocate", default=False)

        location_match = any(
            loc in candidate_location for loc in self.locations_accepted_lower
        )
        if location_match:
            location_score = 1.0
        elif willing_to_relocate:
            location_score = 0.7
        else:
            location_score = 0.2

        # --- Verification trust bonus (5%) ---
        verified_email = 1 if safe_get(signals, "verified_email", default=False) else 0
        verified_phone = 1 if safe_get(signals, "verified_phone", default=False) else 0
        linkedin = 1 if safe_get(signals, "linkedin_connected", default=False) else 0
        trust_score = (verified_email + verified_phone + linkedin) / 3

        # --- Weighted combination ---
        hirability = (
            0.25 * interview_rate +
            0.20 * offer_score +
            0.20 * response_rate +
            0.15 * response_time_score +
            0.10 * notice_score +
            0.05 * location_score +
            0.05 * trust_score
        )

        return round(min(1.0, hirability), 4)

    # ── Sub-scorer 3: Intent & Quality ────────────────────────────────────────

    def score_intent_quality(self, candidate: dict) -> float:
        """
        Score candidate's intent to find a job and overall quality signals.
        """
        signals = candidate.get("redrob_signals", {})
        education = candidate.get("education", [])

        # --- Open to work flag (15%) ---
        open_to_work = 1.0 if safe_get(signals, "open_to_work_flag", default=False) else 0.3

        # --- Recency of activity (20%) ---
        last_active_days = days_since(signals.get("last_active_date", "2000-01-01"))
        if last_active_days <= 7:
            recency_score = 1.0
        elif last_active_days <= 30:
            recency_score = 0.8
        elif last_active_days <= 90:
            recency_score = 0.5
        elif last_active_days <= 180:
            recency_score = 0.3
        else:
            recency_score = 0.1

        # --- Application activity (10%) ---
        apps_30d = safe_get(signals, "applications_submitted_30d", default=0)
        # Normalize: 0 = 0.0, 10+ = 1.0
        activity_score = min(1.0, apps_30d / 10)

        # --- GitHub activity (20%) ---
        github_score = safe_get(signals, "github_activity_score", default=-1)
        if github_score == -1:
            github_normalized = 0.3  # no github → neutral for AI role
        else:
            github_normalized = github_score / 100

        # --- Profile completeness (10%) ---
        completeness = safe_get(signals, "profile_completeness_score", default=50)
        completeness_score = completeness / 100

        # --- Education tier (15%) ---
        if education:
            # Take the highest tier from all education entries
            tier_vals = [
                TIER_SCORES.get(edu.get("tier", "unknown"), 0.4)
                for edu in education
            ]
            edu_score = max(tier_vals)
        else:
            edu_score = 0.3

        # --- Market validation (10%) ---
        # Saved by recruiters and search appearances = other recruiters want them too
        saved = safe_get(signals, "saved_by_recruiters_30d", default=0)
        appearances = safe_get(signals, "search_appearance_30d", default=0)
        # Normalize: saved 5+ = full score, appearances 50+ = full score
        saved_score = min(1.0, saved / 5)
        appearance_score = min(1.0, appearances / 50)
        market_score = (saved_score + appearance_score) / 2

        # --- Weighted combination ---
        intent_quality = (
            0.15 * open_to_work +
            0.20 * recency_score +
            0.10 * activity_score +
            0.20 * github_normalized +
            0.10 * completeness_score +
            0.15 * edu_score +
            0.10 * market_score
        )

        return round(min(1.0, intent_quality), 4)

    # ── Final Scorer ──────────────────────────────────────────────────────────

    def score_candidate(self, candidate: dict) -> dict:
        """
        Compute the full score breakdown for a single candidate.

        Returns:
            Dict with all sub-scores, disqualifiers, and final score.
        """
        # Disqualifier check first
        disqualifier_multiplier, disqualifier_flags = self.get_disqualifier_multiplier(candidate)

        # Three dimension scores
        jd_fit = self.score_jd_fit(candidate)
        hirability = self.score_hirability(candidate)
        intent_quality = self.score_intent_quality(candidate)

        # Weighted composite
        raw_score = (
            WEIGHTS["jd_fit"]         * jd_fit +
            WEIGHTS["hirability"]     * hirability +
            WEIGHTS["intent_quality"] * intent_quality
        )

        # Apply disqualifier penalty
        final_score = raw_score * disqualifier_multiplier

        return {
            "candidate_id": candidate["candidate_id"],
            "final_score": round(final_score, 4),
            "raw_score": round(raw_score, 4),
            "breakdown": {
                "jd_fit":         jd_fit,
                "hirability":     hirability,
                "intent_quality": intent_quality,
            },
            "disqualifier_multiplier": disqualifier_multiplier,
            "disqualifier_flags":      disqualifier_flags,
        }

    # ── Ranker ────────────────────────────────────────────────────────────────

    def rank(self, candidates: list[dict], top_n: int = 100) -> list[dict]:
        """
        Score all candidates and return top N ranked list.

        Args:
            candidates: Full candidate dataset.
            top_n: Number of top candidates to return (default 100).

        Returns:
            List of score dicts sorted by final_score descending.
        """
        print(f"Scoring {len(candidates)} candidates...")
        scored = []

        for i, candidate in enumerate(candidates):
            try:
                result = self.score_candidate(candidate)
                scored.append(result)
            except Exception as e:
                print(f"  ⚠️  Error scoring {candidate.get('candidate_id')}: {e}")

            if (i + 1) % 100 == 0:
                print(f"  Scored {i+1}/{len(candidates)}...")

        # Sort by final score descending
        scored.sort(key=lambda x: x["final_score"], reverse=True)

        print(f"\n✅ Done. Returning top {top_n} candidates.")
        return scored[:top_n]


# ── CSV Export ────────────────────────────────────────────────────────────────

def build_reasoning(candidate: dict, score_result: dict) -> str:
    """
    Build a one-line reasoning string for the submission CSV.
    Format: "<title> with <X> yrs; <N> AI core skills; response rate <R>."
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills  = candidate.get("skills", [])

    title   = profile.get("current_title", "Candidate")
    yrs_exp = profile.get("years_of_experience", 0)

    # Count AI/ML related skills
    ai_keywords = [
        "machine learning", "deep learning", "nlp", "python", "pytorch",
        "tensorflow", "embedding", "vector", "retrieval", "ranking",
        "transformer", "bert", "llm", "sklearn", "scikit", "mlops",
        "faiss", "pinecone", "weaviate", "elasticsearch", "search"
    ]
    ai_skill_count = sum(
        1 for s in skills
        if any(kw in s["name"].lower() for kw in ai_keywords)
    )

    response_rate = signals.get("recruiter_response_rate", 0.0)

    # Append disqualifier note if any
    flags = score_result.get("disqualifier_flags", [])
    flag_note = f" ⚠ {flags[0]}." if flags else ""

    return (
        f"{title} with {yrs_exp} yrs; "
        f"{ai_skill_count} AI core skills; "
        f"response rate {response_rate:.2f}."
        f"{flag_note}"
    )


def export_submission_csv(
    top_100: list[dict],
    candidates: list[dict],
    output_path: str = "submission.csv"
):
    """
    Export top 100 results to submission CSV matching the required format:
    candidate_id, rank, score, reasoning
    """
    import csv

    # Build candidate lookup map
    candidate_map = {c["candidate_id"]: c for c in candidates}

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, result in enumerate(top_100, 1):
            cid       = result["candidate_id"]
            score     = result["final_score"]
            candidate = candidate_map.get(cid, {})
            reasoning = build_reasoning(candidate, result)

            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"📄 Submission CSV saved to {output_path}")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python module3_scoring.py <parsed_jd.json> <candidates.json> <jd_embedding.json> <candidate_embeddings.json>")
        sys.exit(1)

    # Load inputs
    with open(sys.argv[1]) as f:
        parsed_jd = json.load(f)

    # Handle both .json and .jsonl
    with open(sys.argv[2]) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            candidates = json.load(f)
        else:
            candidates = [json.loads(line) for line in f if line.strip()]

    with open(sys.argv[3]) as f:
        jd_embedding = json.load(f)

    with open(sys.argv[4]) as f:
        candidate_embeddings = json.load(f)

    print(f"Loaded {len(candidates)} candidates.")

    # Run scoring
    engine = ScoringEngine(parsed_jd, jd_embedding, candidate_embeddings)
    top_100 = engine.rank(candidates, top_n=100)

    # Save full JSON output
    with open("top_100_candidates.json", "w") as f:
        json.dump(top_100, f, indent=2)
    print("📄 Full results saved to top_100_candidates.json")

    # Save submission CSV
    export_submission_csv(top_100, candidates, output_path="submission.csv")

    # Terminal preview
    print("\nTop 10 Preview:")
    print("-" * 70)
    for i, result in enumerate(top_100[:10], 1):
        flags = f" ⚠️  {result['disqualifier_flags']}" if result["disqualifier_flags"] else ""
        print(
            f"{i:>2}. {result['candidate_id']} | "
            f"Score: {result['final_score']:.4f} | "
            f"JD Fit: {result['breakdown']['jd_fit']:.2f} | "
            f"Hire: {result['breakdown']['hirability']:.2f} | "
            f"Intent: {result['breakdown']['intent_quality']:.2f}"
            f"{flags}"
        )