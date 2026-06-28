"""
app.py — Redrob AI Candidate Ranker
Full pipeline: Upload JD + Candidates → Top 100

Run with:
    streamlit run app.py
"""

import json
import csv
import io
import os
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🎯",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    .hero-title {
        font-size: 2.2rem; font-weight: 800;
        letter-spacing: -0.03em; color: #ffffff;
    }
    .hero-sub { font-size: 1rem; color: #8b90a0; margin-top: 0.3rem; }
    .step-box {
        background: #1a1d27; border: 1px solid #2a2d3a;
        border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    }
    .step-label {
        font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.12em; color: #818cf8; margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #1a1d27; border: 1px solid #2a2d3a;
        border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #818cf8; }
    .metric-label { font-size: 0.72rem; color: #8b90a0; text-transform: uppercase; letter-spacing: 0.08em; }
    hr { border-color: #2a2d3a !important; }
    [data-testid="stSidebar"] { background-color: #13151f; }
    .status-line {
        font-size: 0.85rem; color: #8b90a0; padding: 4px 0;
    }
    .status-done { color: #4ade80; }
    .status-running { color: #facc15; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_candidates(file) -> list:
    content = file.read().decode("utf-8")
    first   = content.strip()[0]
    if first == "[":
        return json.loads(content)
    else:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def to_csv_bytes(top_100: list, candidate_map: dict) -> bytes:
    from module3_scoring import build_reasoning
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, result in enumerate(top_100, 1):
        cid       = result["candidate_id"]
        candidate = candidate_map.get(cid, {})
        reasoning = build_reasoning(candidate, result)
        writer.writerow([cid, rank, f"{result['final_score']:.4f}", reasoning])
    return output.getvalue().encode("utf-8")


def results_to_dataframe(top_100: list, candidate_map: dict) -> pd.DataFrame:
    rows = []
    for r in top_100:
        cid     = r["candidate_id"]
        cand    = candidate_map.get(cid, {})
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        b       = r["breakdown"]
        flags   = r.get("disqualifier_flags", [])
        rows.append({
            "Rank":          r.get("rank", "—"),
            "Candidate ID":  cid,
            "Title":         profile.get("current_title", "—"),
            "Exp (yrs)":     profile.get("years_of_experience", "—"),
            "Final Score":   round(r["final_score"], 3),
            "JD Fit":        round(b["jd_fit"], 3),
            "Hirability":    round(b["hirability"], 3),
            "Intent":        round(b["intent_quality"], 3),
            "Location":      profile.get("location", "—"),
            "Notice (days)": signals.get("notice_period_days", "—"),
            "⚠ Flags":       " | ".join(flags) if flags else "—",
        })
    return pd.DataFrame(rows)


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Hero header
    st.markdown("""
    <div class='hero-title'>🎯 Redrob AI Candidate Ranker</div>
    <div class='hero-sub'>Upload a JD and candidate dataset → get top 100 ranked candidates</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar — weights
    with st.sidebar:
        st.markdown("### ⚙️ Scoring Weights")
        st.markdown("Weights must sum to 1.0")
        st.markdown("---")
        w_fit    = st.slider("JD Fit",          0.0, 1.0, 0.50, 0.05)
        w_hire   = st.slider("Hirability",      0.0, 1.0, 0.30, 0.05)
        w_intent = st.slider("Intent / Quality",0.0, 1.0, 0.20, 0.05)
        total    = round(w_fit + w_hire + w_intent, 2)
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total}. Adjust to 1.0.")
        else:
            st.success(f"✅ Weights sum to {total}")
        st.markdown("---")
        top_n = st.slider("Top N candidates", 10, 100, 100, 10)

    # ── Step 1 & 2: Upload files ──────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='step-label'>Step 1 — Job Description</div>", unsafe_allow_html=True)
        jd_file = st.file_uploader(
            "Upload JD (.txt or .json)",
            type=["txt", "json"],
            help="Plain text JD or a pre-parsed parsed_jd.json"
        )
        if jd_file:
            st.success(f"✅ Loaded: {jd_file.name}")

    with col2:
        st.markdown("<div class='step-label'>Step 2 — Candidate Dataset</div>", unsafe_allow_html=True)
        candidates_file = st.file_uploader(
            "Upload candidates (.json or .jsonl)",
            type=["json", "jsonl"],
            help="Full candidate dataset in JSON array or JSONL format"
        )
        if candidates_file:
            st.success(f"✅ Loaded: {candidates_file.name}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Step 3: Run pipeline ──────────────────────────────────────────────────
    st.markdown("<div class='step-label'>Step 3 — Run Full Pipeline</div>", unsafe_allow_html=True)

    run_ready = jd_file is not None and candidates_file is not None
    if not run_ready:
        st.info("👆 Upload both files above to enable ranking.")

    run_btn = st.button(
        "🚀 Run Full Pipeline → Get Top 100",
        type="primary",
        disabled=not run_ready,
        use_container_width=True
    )

    if run_btn and run_ready:

        # ── Import modules ────────────────────────────────────────────────────
        from module1_jd_parser import parse_jd, validate_parsed_jd
        from module2_taxonomy  import TaxonomyEngine
        from module3_scoring   import ScoringEngine

        import module3_scoring
        module3_scoring.WEIGHTS["jd_fit"]         = w_fit
        module3_scoring.WEIGHTS["hirability"]     = w_hire
        module3_scoring.WEIGHTS["intent_quality"] = w_intent

        # Progress UI
        progress_bar = st.progress(0)
        status_area  = st.empty()

        def update(pct, msg, done=False):
            progress_bar.progress(pct)
            icon = "✅" if done else "⏳"
            status_area.markdown(f"**{icon} {msg}**")

        start_time = time.time()

        # ── Stage 1: Parse JD ─────────────────────────────────────────────────
        update(0.05, "Stage 1/4 — Parsing Job Description...")
        jd_content = jd_file.read().decode("utf-8")

        if jd_file.name.endswith(".json"):
            parsed_jd = json.loads(jd_content)
            st.toast("JD loaded from pre-parsed JSON", icon="📄")
        else:
            try:
                parsed_jd = parse_jd(jd_content)
                valid, issues = validate_parsed_jd(parsed_jd)
                if not valid:
                    st.warning(f"JD validation issues: {issues}")
            except Exception as e:
                st.error(f"JD parsing failed: {e}")
                st.stop()

        update(0.15, "Stage 1/4 — JD parsed ✓", done=True)

        with st.expander("📋 View parsed JD", expanded=False):
            st.json(parsed_jd)

        # ── Stage 2: Load candidates ──────────────────────────────────────────
        update(0.20, "Stage 2/4 — Loading candidate dataset...")
        try:
            candidates_file.seek(0)
            candidates    = load_candidates(candidates_file)
            candidate_map = {c["candidate_id"]: c for c in candidates}
            update(0.25, f"Stage 2/4 — {len(candidates)} candidates loaded ✓", done=True)
            st.toast(f"{len(candidates)} candidates loaded", icon="👥")
        except Exception as e:
            st.error(f"Failed to load candidates: {e}")
            st.stop()

        # ── Stage 3: Taxonomy + Embeddings ────────────────────────────────────
        update(0.28, "Stage 3/4 — Building skill taxonomy (LLM)...")
        engine = TaxonomyEngine()

        # Collect skills
        all_skills = engine.collect_skills(
            jd_required=parsed_jd.get("required_skills", []),
            jd_preferred=parsed_jd.get("preferred_skills", []),
            candidate_dataset=candidates
        )
        update(0.32, f"Stage 3/4 — Found {len(all_skills)} unique skills. Building taxonomy...")

        # Build taxonomy with live batch updates
        from anthropic import Anthropic
        tax_client  = Anthropic()
        batch_size  = 100
        batches     = [all_skills[i:i+batch_size] for i in range(0, len(all_skills), batch_size)]
        taxonomy    = {}

        for idx, batch in enumerate(batches):
            batch_pct = 0.32 + (idx / len(batches)) * 0.18
            update(batch_pct, f"Stage 3/4 — Taxonomy batch {idx+1}/{len(batches)}...")

            from module2_taxonomy import TAXONOMY_PROMPT
            skills_list = "\n".join(f"- {s}" for s in batch)
            response = tax_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                messages=[{"role": "user", "content": TAXONOMY_PROMPT.format(skills_list=skills_list)}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
                raw = raw.strip()
            try:
                taxonomy.update(json.loads(raw))
            except Exception:
                pass
            if idx < len(batches) - 1:
                time.sleep(0.4)

        engine.taxonomy = taxonomy
        update(0.50, f"Stage 3/4 — Taxonomy built ({len(taxonomy)} skills). Embedding candidates...")

        # Embed JD
        jd_embedding = engine.embed_jd(parsed_jd)

        # Embed candidates with progress
        candidate_embeddings = {}
        total_cands = len(candidates)
        for i, candidate in enumerate(candidates):
            cid = candidate["candidate_id"]
            candidate_embeddings[cid] = engine.embed_candidate(candidate)
            if (i + 1) % 50 == 0 or (i + 1) == total_cands:
                emb_pct = 0.50 + ((i + 1) / total_cands) * 0.25
                update(emb_pct, f"Stage 3/4 — Embedding candidates: {i+1}/{total_cands}...")

        update(0.75, "Stage 3/4 — All embeddings ready ✓", done=True)

        # ── Stage 4: Scoring + Ranking ────────────────────────────────────────
        update(0.77, f"Stage 4/4 — Scoring {len(candidates)} candidates...")

        scoring_engine = ScoringEngine(parsed_jd, jd_embedding, candidate_embeddings)
        scored = []

        for i, candidate in enumerate(candidates):
            try:
                result = scoring_engine.score_candidate(candidate)
                scored.append(result)
            except Exception:
                pass
            if (i + 1) % 100 == 0 or (i + 1) == total_cands:
                score_pct = 0.77 + ((i + 1) / total_cands) * 0.20
                update(score_pct, f"Stage 4/4 — Scoring: {i+1}/{total_cands}...")

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_100 = scored[:top_n]
        for idx, r in enumerate(top_100, 1):
            r["rank"] = idx

        elapsed = round(time.time() - start_time, 1)
        update(1.0, f"✅ Done in {elapsed}s — Top {top_n} candidates ready!", done=True)

        st.session_state["top_100"]       = top_100
        st.session_state["candidate_map"] = candidate_map
        st.session_state["parsed_jd"]     = parsed_jd

    # ── Results ───────────────────────────────────────────────────────────────
    if "top_100" in st.session_state:
        top_100       = st.session_state["top_100"]
        candidate_map = st.session_state["candidate_map"]
        parsed_jd     = st.session_state["parsed_jd"]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='step-label'>Results</div>", unsafe_allow_html=True)

        # Summary metrics
        scores  = [r["final_score"] for r in top_100]
        flagged = sum(1 for r in top_100 if r.get("disqualifier_flags"))

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(top_100)}</div><div class='metric-label'>Ranked</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{max(scores):.3f}</div><div class='metric-label'>Top Score</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{sum(scores)/len(scores):.3f}</div><div class='metric-label'>Avg Score</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{flagged}</div><div class='metric-label'>Flagged</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Download
        csv_bytes = to_csv_bytes(top_100, candidate_map)
        st.download_button(
            label="⬇️ Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Results table
        df = results_to_dataframe(top_100, candidate_map)
        st.dataframe(
            df,
            use_container_width=True,
            height=500,
            column_config={
                "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=1, format="%.3f"),
                "JD Fit":      st.column_config.ProgressColumn("JD Fit",      min_value=0, max_value=1, format="%.3f"),
                "Hirability":  st.column_config.ProgressColumn("Hirability",  min_value=0, max_value=1, format="%.3f"),
                "Intent":      st.column_config.ProgressColumn("Intent",      min_value=0, max_value=1, format="%.3f"),
            }
        )

        # Score distribution
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='step-label'>Score Distribution across Top Candidates</div>", unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "Rank":        range(1, len(top_100) + 1),
            "Final Score": scores,
            "JD Fit":      [r["breakdown"]["jd_fit"] for r in top_100],
            "Hirability":  [r["breakdown"]["hirability"] for r in top_100],
            "Intent":      [r["breakdown"]["intent_quality"] for r in top_100],
        })
        st.line_chart(chart_df.set_index("Rank"), height=280)


if __name__ == "__main__":
    main()