import os
import pandas as pd
from response_engine_legacy import ResponseEngine

# -----------------------------
# Helper Functions
# -----------------------------

# Combine research areas from multiple columns into a single string
def combine_research_areas(row):
    areas = []
    # Iterate over research area columns
    for col in ["research_area_1", "research_area_2", "research_area_3", "research_area_4", "research_area_5"]:
        val = str(row.get(col, "")).strip()# Check for non-empty values
        if val not in ["", "nan", "None"]:# Add to list
            areas.append(val)
    return ", ".join(areas) # Combine into single string

# Summarize text using the Llama response engine
def summarize_text(engine, text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    prompt = f"""
Summarize the following research abstract into 3–4 clean academic sentences 
that highlight the contribution, topic, and core methods. 
Avoid unnecessary wording and keep it concise.

Abstract:
{text}
"""
    # Generate summary
    try:
        out = engine.generate_answer(prompt)
        return out.strip()
    except Exception as e:
        print(f"[ERROR] Llama summary failed: {e}")
        return None

# Build profile text for a faculty member
def build_profile(row):
    name = row["name"]
    position = row["position"]
    email = row["email"]
    office = row.get("office", "")
    research_areas = row["combined_research_areas"]
    scholar = row.get("google_scholar_link", "")
    availability = row.get("availability", "")

    # Summaries previously generated
    s1 = row.get("summary_1", "")
    s2 = row.get("summary_2", "")

    # Build papers section
    papers_section = ""
    if isinstance(s1, str) and len(s1) > 0:
        papers_section += f"\nPaper 1 Summary:\n{s1}\n"
    if isinstance(s2, str) and len(s2) > 0:
        papers_section += f"\nPaper 2 Summary:\n{s2}\n"
# Build full profile text
    profile = f"""
{name} is a {position} at Boise State University.

Research Areas:
{research_areas}

{papers_section}

Google Scholar: {scholar}
Office: {office}
Availability: {availability}
Email: {email}
"""

    return " ".join(profile.split())


# -----------------------------
# Main Pipeline
# -----------------------------
# Build enriched faculty profiles and save to CSV
def main():
    print("[INFO] Loading faculty dataset...")
    df = pd.read_csv("data/Database.csv", encoding="cp1252")


    print("[INFO] Initializing Llama Response Engine...")
    engine = ResponseEngine()

    # Add combined research areas
    print("[INFO] Combining research areas...")
    df["combined_research_areas"] = df.apply(combine_research_areas, axis=1)

    # Summaries
    summaries_1 = []
    summaries_2 = []

    print("[INFO] Summarizing abstracts...")
    for idx, row in df.iterrows():
        print(f"[INFO] Summarizing for: {row['name']}")

        abs1 = row.get("abstract_1", "")
        abs2 = row.get("abstract_2", "")

        s1 = summarize_text(engine, abs1)
        s2 = summarize_text(engine, abs2)

        summaries_1.append(s1)
        summaries_2.append(s2)

    df["summary_1"] = summaries_1
    df["summary_2"] = summaries_2

    # Build profile_text
    print("[INFO] Building profile_text...")
    df["profile_text"] = df.apply(build_profile, axis=1)

    # Save
    out_path = "data/faculty_ready.csv"
    df.to_csv(out_path, index=False)
    print(f"[DONE] Saved enriched dataset to: {out_path}")


if __name__ == "__main__":
    main()
