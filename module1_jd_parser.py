"""
Module 1: JD Parser
-------------------
Takes raw job description text and returns a structured JSON object
with all fields needed for downstream scoring and matching.

Usage:
    from module1_jd_parser import parse_jd
    parsed = parse_jd(jd_text)
"""

import json
import anthropic

# ── Client ────────────────────────────────────────────────────────────────────
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# ── Prompt ────────────────────────────────────────────────────────────────────
JD_PARSE_PROMPT = """
You are a precise JD parser for a recruitment AI system.

Extract the following from the job description and return ONLY valid JSON with no markdown backticks.

Output schema:
{
  "role_title": string,
  "seniority": "junior" | "mid" | "senior" | "lead",
  "domain": string,
  "required_skills": [string],
  "preferred_skills": [string],
  "disqualifying_signals": [string],
  "min_experience_years": number,
  "max_experience_years": number | null,
  "work_mode": "remote" | "hybrid" | "onsite" | "flexible" | null,
  "locations_accepted": [string],
  "salary_range_inr_lpa": {"min": number, "max": number} | null,
  "notice_period_preference_days": number,
  "notice_period_max_days": number,
  "industry_preference": [string],
  "company_type_preference": [string],
  "key_responsibilities": [string],
  "ideal_candidate_summary": string,
  "culture_signals": [string]
}

Field rules:
- required_skills     : explicitly must-have technical skills, be granular
- preferred_skills    : nice-to-have / bonus skills
- disqualifying_signals: explicit red flags or deal-breakers mentioned in the JD
- company_type_preference: what kind of company background is preferred (product, startup, etc.)
- culture_signals     : soft / cultural traits the company values
- If salary is not mentioned → null
- If experience is "3+ years" → min=3, max=null
- If notice period not mentioned → use 30 as default preference, 90 as max
- Return ONLY the JSON object, no explanation, no markdown
"""

# ── Core function ─────────────────────────────────────────────────────────────
def parse_jd(jd_text: str) -> dict:
    """
    Parse a raw job description string into a structured dict.

    Args:
        jd_text: Full raw text of the job description.

    Returns:
        Parsed JD as a Python dict matching the output schema above.

    Raises:
        ValueError: If the LLM returns malformed JSON.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": f"{JD_PARSE_PROMPT}\n\nJOB DESCRIPTION:\n{jd_text}"
            }
        ]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if model wraps output despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JD parser returned invalid JSON: {e}\nRaw output:\n{raw}")

    return parsed


# ── Validation helper ─────────────────────────────────────────────────────────
REQUIRED_FIELDS = [
    "role_title", "seniority", "domain",
    "required_skills", "preferred_skills", "disqualifying_signals",
    "min_experience_years", "work_mode", "locations_accepted",
    "key_responsibilities", "ideal_candidate_summary", "culture_signals"
]

def validate_parsed_jd(parsed: dict) -> tuple[bool, list[str]]:
    """
    Check that all required fields are present and non-empty.

    Returns:
        (is_valid, list_of_missing_or_empty_fields)
    """
    issues = []
    for field in REQUIRED_FIELDS:
        if field not in parsed:
            issues.append(f"Missing field: {field}")
        elif parsed[field] in [None, "", [], {}]:
            issues.append(f"Empty field: {field}")
    return (len(issues) == 0, issues)


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python module1_jd_parser.py <path_to_jd.txt>")
        print("       or pipe text: cat jd.txt | python module1_jd_parser.py -")
        sys.exit(1)

    path = sys.argv[1]
    if path == "-":
        jd_text = sys.stdin.read()
    else:
        with open(path, "r") as f:
            jd_text = f.read()

    print("Parsing JD...", file=sys.stderr)
    result = parse_jd(jd_text)

    valid, issues = validate_parsed_jd(result)
    if not valid:
        print(f"⚠️  Validation warnings: {issues}", file=sys.stderr)
    else:
        print("✅ Parsed and validated successfully.", file=sys.stderr)

    print(json.dumps(result, indent=2))
