import os
import json
import html
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")

HTML_FILE = Path("index.html")
DATA_FILE = Path("conferences.json")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ------------------------------------------------------------
# LOAD EXISTING DATA
# ------------------------------------------------------------

def load_existing_data():
    if not DATA_FILE.exists():
        return []

    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read conferences.json: {exc}")
        return []


existing = load_existing_data()


# ------------------------------------------------------------
# DAILY RESEARCH INSTRUCTIONS
# ------------------------------------------------------------

today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

research_prompt = f"""
Today is {today}.

Act as a conference research analyst specializing in analytical chemistry,
environmental analysis and analytical instrumentation.

Search the current web for NEW conferences and MATERIAL UPDATES to known
conferences relevant to:

- environmental analysis
- environmental mass spectrometry
- PFAS analysis
- microplastics and nanoplastics
- water analysis and monitoring
- environmental analytical chemistry
- chromatography when substantially relevant to environmental analysis
- mass spectrometry when substantially relevant
- ICP-MS and elemental environmental analysis
- emerging contaminants
- environmental measurement
- laboratory analytical instrumentation when environmental applications
  are substantial

Prioritize ORIGINAL AND AUTHORITATIVE SOURCES.

Examples include:

- Pittcon
- ACS ENVR
- ACS ANYL
- SETAC
- ASMS
- AWWA
- Environmental Measurement Symposium / NEMC
- HPLC conference
- IAEAC / ISEAC
- official conference organizers
- official professional societies
- official call-for-abstract pages

SEARCH REQUIREMENTS

1. Use web search.
2. Prefer official organizer pages over aggregators.
3. Do not invent dates.
4. Do not use a deadline unless it can be supported by an authoritative
   conference or society source.
5. Look specifically for:
   - abstract submission opening dates
   - abstract deadlines
   - poster deadlines
   - oral presentation deadlines
   - late-breaking deadlines
   - session proposal deadlines
   - registration deadlines/openings
   - deadline extensions
   - conference dates
   - location changes
   - newly announced conferences
6. Include future conferences that do not yet have deadlines when they are
   highly relevant. Mark these WATCH.
7. Do not include generic predatory conference-directory listings.
8. Do not delete a valid existing conference merely because today's search
   did not rediscover it.
9. If an existing deadline has changed, use the newly verified official
   information.
10. Avoid duplicates.

CURRENT DATABASE

{json.dumps(existing, indent=2)}

Return ONLY valid JSON.

Use this exact structure:

{{
  "conferences": [
    {{
      "conference_name": "Conference name",
      "topic": "Short explanation of relevance",
      "conference_dates": "YYYY-MM-DD to YYYY-MM-DD or best verified format",
      "location": "City, State/Region, Country",
      "milestone_date": "YYYY-MM-DD or TBD",
      "milestone_type": "Abstract deadline",
      "submission_type": "Oral/poster/etc.",
      "status": "OPEN",
      "status_update": "Brief description of new or important information",
      "official_url": "https://official-source...",
      "last_checked": "{today}"
    }}
  ]
}}

Allowed status values:

OPEN
NEW
IMMINENT
WATCH
CLOSED
EXTENDED

Return a separate row when one conference has multiple important milestones.

Preserve valid existing database entries unless official evidence shows that
they need to be changed.

Order dated milestones chronologically. Put TBD milestones after dated ones.
"""


# ------------------------------------------------------------
# ASK OPENAI TO SEARCH THE WEB
# ------------------------------------------------------------

print("Searching for conference updates...")

response = client.responses.create(
    model=MODEL,
    tools=[
        {
            "type": "web_search",
            "search_context_size": "high"
        }
    ],
    input=research_prompt
)

raw = response.output_text.strip()


# ------------------------------------------------------------
# CLEAN / PARSE JSON
# ------------------------------------------------------------

def extract_json(text):
    text = text.strip()

    # Remove Markdown code fences if returned.
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: locate outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise

        return json.loads(text[start:end + 1])


result = extract_json(raw)

new_data = result.get("conferences", [])

if not isinstance(new_data, list):
    raise ValueError("OpenAI response did not contain a conferences list.")


# ------------------------------------------------------------
# VALIDATE AND DEDUPLICATE
# ------------------------------------------------------------

required_fields = [
    "conference_name",
    "topic",
    "conference_dates",
    "location",
    "milestone_date",
    "milestone_type",
    "submission_type",
    "status",
    "status_update",
    "official_url",
    "last_checked",
]

allowed_status = {
    "OPEN",
    "NEW",
    "IMMINENT",
    "WATCH",
    "CLOSED",
    "EXTENDED",
}


def normalize(value):
    return " ".join(str(value or "").strip().lower().split())


clean_data = []
seen = set()

for row in new_data:

    if not isinstance(row, dict):
        continue

    for field in required_fields:
        row.setdefault(field, "")

    # Reject rows without a conference name.
    if not normalize(row["conference_name"]):
        continue

    # Require an official-looking HTTP source.
    url = str(row["official_url"]).strip()

    if url and not url.startswith(("http://", "https://")):
        continue

    status = str(row["status"]).strip().upper()

    if status not in allowed_status:
        status = "WATCH"

    row["status"] = status
    row["last_checked"] = today

    # Conference + milestone type + milestone date forms the unique key.
    key = (
        normalize(row["conference_name"]),
        normalize(row["milestone_type"]),
        normalize(row["milestone_date"]),
    )

    if key in seen:
        continue

    seen.add(key)
    clean_data.append(row)


# ------------------------------------------------------------
# SORT BY DEADLINE
# ------------------------------------------------------------

def deadline_sort_key(row):

    value = str(row.get("milestone_date", "")).strip()

    try:
        return (0, datetime.strptime(value, "%Y-%m-%d"))
    except ValueError:
        return (1, datetime.max)


clean_data.sort(key=deadline_sort_key)


# ------------------------------------------------------------
# SAVE DATABASE
# ------------------------------------------------------------

DATA_FILE.write_text(
    json.dumps(clean_data, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
)


# ------------------------------------------------------------
# GENERATE HTML
# ------------------------------------------------------------

def esc(value):
    return html.escape(str(value or ""))


def status_badge(status):

    status = status.upper()

    css = {
        "OPEN": "open",
        "NEW": "new",
        "IMMINENT": "imminent",
        "WATCH": "watch",
        "CLOSED": "closed",
        "EXTENDED": "extended",
    }.get(status, "watch")

    return f'<span class="badge {css}">{esc(status)}</span>'


rows_html = []

for row in clean_data:

    url = esc(row["official_url"])

    if url:
        source = (
            f'<a href="{url}" target="_blank" '
            f'rel="noopener noreferrer">Official source ↗</a>'
        )
    else:
        source = "—"

    status_text = status_badge(row["status"])

    if row["status_update"]:
        status_text += f'<div class="status-note">{esc(row["status_update"])}</div>'

    rows_html.append(
        f"""
        <tr>
            <td><strong>{esc(row["conference_name"])}</strong></td>
            <td>{esc(row["topic"])}</td>
            <td>{esc(row["conference_dates"])}</td>
            <td>{esc(row["location"])}</td>
            <td class="deadline">{esc(row["milestone_date"])}</td>
            <td>{esc(row["milestone_type"])}</td>
            <td>{esc(row["submission_type"])}</td>
            <td>{status_text}</td>
            <td>{source}</td>
            <td>{esc(row["last_checked"])}</td>
        </tr>
        """
    )


table_rows = "\n".join(rows_html)

updated_display = datetime.now(timezone.utc).strftime("%B %d, %Y")


page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Environmental Analysis Conference Tracker</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, Arial, sans-serif;
    background: #f5f7fa;
    color: #1f2937;
}}

.container {{
    max-width: 1600px;
    margin: auto;
    padding: 38px 24px;
}}

h1 {{
    margin: 0 0 8px;
    font-size: 32px;
}}

.subtitle {{
    color: #667085;
    margin-bottom: 8px;
}}

.updated {{
    color: #667085;
    font-size: 14px;
    margin-bottom: 22px;
}}

.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 18px;
}}

.badge {{
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
}}

.open {{
    background: #dcfce7;
    color: #166534;
}}

.new {{
    background: #dbeafe;
    color: #1e40af;
}}

.imminent {{
    background: #fef3c7;
    color: #92400e;
}}

.watch {{
    background: #e5e7eb;
    color: #374151;
}}

.closed {{
    background: #fee2e2;
    color: #991b1b;
}}

.extended {{
    background: #ede9fe;
    color: #5b21b6;
}}

.status-note {{
    margin-top: 6px;
    font-size: 12px;
    line-height: 1.35;
}}

.table-wrapper {{
    overflow-x: auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
}}

table {{
    width: 100%;
    min-width: 1350px;
    border-collapse: collapse;
}}

th {{
    background: #111827;
    color: white;
    padding: 13px;
    text-align: left;
    font-size: 12px;
    position: sticky;
    top: 0;
}}

td {{
    padding: 13px;
    border-bottom: 1px solid #e5e7eb;
    vertical-align: top;
    font-size: 13px;
}}

tr:hover {{
    background: #f9fafb;
}}

.deadline {{
    font-weight: 700;
    white-space: nowrap;
}}

a {{
    color: #2563eb;
    text-decoration: none;
}}

a:hover {{
    text-decoration: underline;
}}

footer {{
    margin-top: 22px;
    color: #667085;
    font-size: 12px;
    line-height: 1.5;
}}
</style>
</head>

<body>

<div class="container">

<h1>Environmental Analysis Conference Tracker</h1>

<div class="subtitle">
Environmental Analysis • Mass Spectrometry • PFAS •
Microplastics &amp; Nanoplastics • Water Analysis &amp; Monitoring
</div>

<div class="updated">
Automatically updated: {updated_display}
</div>

<div class="legend">
<span class="badge new">NEW</span>
<span class="badge imminent">IMMINENT</span>
<span class="badge extended">EXTENDED</span>
<span class="badge open">OPEN</span>
<span class="badge watch">WATCH</span>
<span class="badge closed">CLOSED</span>
</div>

<div class="table-wrapper">

<table>

<thead>
<tr>
<th>Conference</th>
<th>Topic / Relevance</th>
<th>Conference Dates</th>
<th>Location</th>
<th>Milestone / Deadline</th>
<th>Milestone Type</th>
<th>Submission Type</th>
<th>Status / Update</th>
<th>Official Source</th>
<th>Last Checked</th>
</tr>
</thead>

<tbody>
{table_rows}
</tbody>

</table>

</div>

<footer>
Conference information is automatically researched using current web
sources with preference for official organizers and professional societies.
Always verify critical submission deadlines directly with the linked
conference organizer before submission.
</footer>

</div>

</body>
</html>
"""


HTML_FILE.write_text(page, encoding="utf-8")

print(f"Saved {len(clean_data)} conference milestone rows.")
print("Updated conferences.json")
print("Updated index.html")
