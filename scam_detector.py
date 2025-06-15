import re

SCAM_PATTERNS = {
    "high_pay": re.compile(r"make\s*\$\d+/day", re.IGNORECASE),
    "no_experience": re.compile(r"no\s*(experience|skills)", re.IGNORECASE),
    "upfront_pay": re.compile(r"(pay|deposit|fee)\s*upfront", re.IGNORECASE)
}

def detect_scam_keywords(text):
    return [name for name, pattern in SCAM_PATTERNS.items() if pattern.search(str(text))]