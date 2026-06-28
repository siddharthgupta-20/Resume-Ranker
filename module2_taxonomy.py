"""
Module 2: Taxonomy Engine
--------------------------
1. Collects all unique skills from the parsed JD + candidate dataset
2. Sends them to an LLM to build a 3-level skill taxonomy
3. Enriches every skill string to: "skill | concept | umbrella"
4. Generates embeddings for enriched JD skills and candidate skills

Usage:
    from module2_taxonomy import TaxonomyEngine
    engine = TaxonomyEngine()
    engine.build(jd_skills, candidate_skills_list)
    enriched = engine.enrich_skill("FastAPI")
    # → "FastAPI | REST APIs | Backend Development"
"""

import json
import time
import anthropic
from sentence_transformers import SentenceTransformer

# ── Client & Model ────────────────────────────────────────────────────────────
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # fast, good quality

# ── Prompt ────────────────────────────────────────────────────────────────────
TAXONOMY_PROMPT = """
You are a technical skill taxonomy builder for a recruitment AI system.

I will give you a list of skills. For each skill, assign it to a 3-level hierarchy:
- level_3: the skill itself (keep exactly as given)
- level_2: the CONCEPT it belongs to (what capability does it represent?)
- level_1: the UMBRELLA domain (the broadest category)

Critical rules:
- Similar tools that do the same job MUST share the exact same level_2 concept name
  Example: FAISS, Pinecone, Weaviate, Qdrant  → level_2 = "Vector Databases"
  Example: FastAPI, Django REST, Flask         → level_2 = "REST APIs"
  Example: PyTorch, TensorFlow, JAX            → level_2 = "Deep Learning Frameworks"
  Example: pandas, polars, dask                → level_2 = "Data Manipulation"
  Example: AWS, GCP, Azure                     → level_2 = "Cloud Platforms"
- level_1 umbrellas must come from this fixed set:
  "AI/ML" | "ML Infrastructure" | "Backend Development" | "Data Engineering" |
  "DevOps" | "Frontend Development" | "Software Engineering" | "Data Science" |
  "NLP/IR" | "Databases" | "Security" | "Other"
- If a skill is ambiguous, assign based on its most common use in the tech industry
- Return ONLY valid JSON, no markdown, no explanation

Input skills:
{skills_list}

Return format:
{{
  "skill_name": {{
    "level_3": "skill_name",
    "level_2": "concept it belongs to",
    "level_1": "umbrella domain"
  }},
  ...
}}
"""

# ── Taxonomy Engine ───────────────────────────────────────────────────────────
class TaxonomyEngine:
    def __init__(self):
        self.taxonomy: dict = {}        # skill → {level_3, level_2, level_1}
        self.embeddings: dict = {}      # enriched_string → vector

    # ── Step 1: Collect unique skills ─────────────────────────────────────────
    def collect_skills(
        self,
        jd_required: list[str],
        jd_preferred: list[str],
        candidate_dataset: list[dict]
    ) -> list[str]:
        """
        Collect all unique skills from the JD and across all candidates.

        Args:
            jd_required: Required skills list from parsed JD.
            jd_preferred: Preferred skills list from parsed JD.
            candidate_dataset: List of candidate dicts (raw from JSON dataset).

        Returns:
            Deduplicated list of all unique skill strings.
        """
        skill_set = set()

        # From JD
        for s in jd_required + jd_preferred:
            skill_set.add(s.strip())

        # From candidates
        for candidate in candidate_dataset:
            for skill_obj in candidate.get("skills", []):
                skill_set.add(skill_obj["name"].strip())

        return list(skill_set)

    # ── Step 2: Build taxonomy via LLM ────────────────────────────────────────
    def build_taxonomy(self, all_skills: list[str], batch_size: int = 100) -> dict:
        """
        Call the LLM to build a 3-level taxonomy for all skills.
        Processes in batches to stay within token limits.

        Args:
            all_skills: Full deduplicated skill list.
            batch_size: How many skills to send per LLM call.

        Returns:
            Taxonomy dict: {skill_name: {level_3, level_2, level_1}}
        """
        taxonomy = {}
        batches = [all_skills[i:i+batch_size] for i in range(0, len(all_skills), batch_size)]

        for idx, batch in enumerate(batches):
            print(f"  Building taxonomy: batch {idx+1}/{len(batches)} ({len(batch)} skills)...")
            skills_list = "\n".join(f"- {s}" for s in batch)

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": TAXONOMY_PROMPT.format(skills_list=skills_list)
                    }
                ]
            )

            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            try:
                batch_taxonomy = json.loads(raw)
                taxonomy.update(batch_taxonomy)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Batch {idx+1} returned invalid JSON: {e}. Skipping.")

            # Avoid rate limiting
            if idx < len(batches) - 1:
                time.sleep(0.5)

        self.taxonomy = taxonomy
        return taxonomy

    # ── Step 3: Enrich a single skill string ──────────────────────────────────
    def enrich_skill(self, skill: str) -> str:
        """
        Enrich a skill name with its concept and umbrella from the taxonomy.

        Args:
            skill: Raw skill name e.g. "FastAPI"

        Returns:
            Enriched string e.g. "FastAPI | REST APIs | Backend Development"
            Falls back to skill name alone if not in taxonomy.
        """
        entry = self.taxonomy.get(skill)
        if not entry:
            # Try case-insensitive lookup
            lower_map = {k.lower(): v for k, v in self.taxonomy.items()}
            entry = lower_map.get(skill.lower())

        if not entry:
            return skill  # fallback: just the raw skill

        return f"{entry['level_3']} | {entry['level_2']} | {entry['level_1']}"

    # ── Step 4: Enrich a list of skills ───────────────────────────────────────
    def enrich_skills_list(self, skills: list[str]) -> list[str]:
        """Enrich a list of skill strings."""
        return [self.enrich_skill(s) for s in skills]

    # ── Step 5: Build enriched text block for embedding ───────────────────────
    def build_jd_text_for_embedding(self, parsed_jd: dict) -> str:
        """
        Build a rich text representation of the JD for embedding.
        Combines enriched required skills + responsibilities + domain.

        Args:
            parsed_jd: Output of Module 1 JD parser.

        Returns:
            Single string ready to be embedded.
        """
        enriched_required = self.enrich_skills_list(parsed_jd["required_skills"])
        enriched_preferred = self.enrich_skills_list(parsed_jd["preferred_skills"])

        parts = [
            f"Role: {parsed_jd['role_title']}",
            f"Domain: {parsed_jd['domain']}",
            f"Seniority: {parsed_jd['seniority']}",
            "Required Skills: " + ", ".join(enriched_required),
            "Preferred Skills: " + ", ".join(enriched_preferred),
            "Responsibilities: " + ". ".join(parsed_jd["key_responsibilities"]),
            f"Ideal Candidate: {parsed_jd['ideal_candidate_summary']}"
        ]

        return "\n".join(parts)

    def build_candidate_text_for_embedding(self, candidate: dict) -> str:
        """
        Build a rich text representation of a candidate for embedding.
        Combines enriched skills + career history descriptions + headline.

        Args:
            candidate: Single candidate dict from the dataset.

        Returns:
            Single string ready to be embedded.
        """
        # Enrich skills
        raw_skills = [s["name"] for s in candidate.get("skills", [])]
        enriched_skills = self.enrich_skills_list(raw_skills)

        # Career history descriptions
        career_texts = []
        for job in candidate.get("career_history", []):
            career_texts.append(f"{job['title']} at {job['company']}: {job['description']}")

        # Assessment scores if available
        assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
        assessment_text = ""
        if assessment_scores:
            top = sorted(assessment_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            assessment_text = "Verified Skills: " + ", ".join(f"{k}({v:.0f})" for k, v in top)

        parts = [
            f"Headline: {candidate['profile']['headline']}",
            f"Summary: {candidate['profile']['summary']}",
            f"Current Role: {candidate['profile']['current_title']} at {candidate['profile']['current_company']}",
            f"Years of Experience: {candidate['profile']['years_of_experience']}",
            "Skills: " + ", ".join(enriched_skills),
            "Career History: " + " | ".join(career_texts),
        ]

        if assessment_text:
            parts.append(assessment_text)

        return "\n".join(parts)

    # ── Step 6: Generate embeddings ───────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a text string."""
        return EMBED_MODEL.encode(text, normalize_embeddings=True).tolist()

    def embed_jd(self, parsed_jd: dict) -> list[float]:
        """Generate embedding for the full JD."""
        text = self.build_jd_text_for_embedding(parsed_jd)
        return self.embed(text)

    def embed_candidate(self, candidate: dict) -> list[float]:
        """Generate embedding for a single candidate."""
        text = self.build_candidate_text_for_embedding(candidate)
        return self.embed(text)

    def embed_all_candidates(self, candidates: list[dict]) -> dict[str, list[float]]:
        """
        Generate embeddings for all candidates in the dataset.

        Returns:
            Dict of {candidate_id: embedding_vector}
        """
        embeddings = {}
        for i, candidate in enumerate(candidates):
            cid = candidate["candidate_id"]
            embeddings[cid] = self.embed_candidate(candidate)
            if (i + 1) % 100 == 0:
                print(f"  Embedded {i+1}/{len(candidates)} candidates...")
        return embeddings

    # ── Save / Load taxonomy ──────────────────────────────────────────────────
    def save_taxonomy(self, path: str):
        """Save taxonomy to disk so you don't rebuild it every run."""
        with open(path, "w") as f:
            json.dump(self.taxonomy, f, indent=2)
        print(f"  Taxonomy saved to {path}")

    def load_taxonomy(self, path: str):
        """Load a previously saved taxonomy from disk."""
        with open(path, "r") as f:
            self.taxonomy = json.load(f)
        print(f"  Taxonomy loaded from {path} ({len(self.taxonomy)} skills)")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Example: build taxonomy from a parsed JD + candidate dataset
    if len(sys.argv) < 3:
        print("Usage: python module2_taxonomy.py <parsed_jd.json> <candidates.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        parsed_jd = json.load(f)

    with open(sys.argv[2]) as f:
        candidates = json.load(f)

    engine = TaxonomyEngine()

    print("Step 1: Collecting all unique skills...")
    all_skills = engine.collect_skills(
        jd_required=parsed_jd["required_skills"],
        jd_preferred=parsed_jd["preferred_skills"],
        candidate_dataset=candidates
    )
    print(f"  Found {len(all_skills)} unique skills.")

    print("Step 2: Building taxonomy via LLM...")
    engine.build_taxonomy(all_skills)
    engine.save_taxonomy("taxonomy.json")

    print("Step 3: Embedding JD...")
    jd_vector = engine.embed_jd(parsed_jd)
    print(f"  JD embedding dimension: {len(jd_vector)}")

    print("Step 4: Embedding all candidates...")
    candidate_embeddings = engine.embed_all_candidates(candidates)
    print(f"  Done. {len(candidate_embeddings)} candidates embedded.")

    # Save embeddings
    with open("candidate_embeddings.json", "w") as f:
        json.dump(candidate_embeddings, f)
    print("  Candidate embeddings saved to candidate_embeddings.json")

    with open("jd_embedding.json", "w") as f:
        json.dump(jd_vector, f)
    print("  JD embedding saved to jd_embedding.json")
