import os
import json
import random
import re
from requests_html import HTMLSession
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# -------------------------------
# CONFIGURATION
# -------------------------------
BASE_URL = "YOUR_WEBSITE"
OUTPUT_DIR = "output_prodapt"
MAX_PAGES = 60
TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT = 0.8, 0.1, 0.1

visited = set()
session = HTMLSession()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# SCRAPER
# -------------------------------
def scrape_page(url):
    try:
        r = session.get(url, timeout=30)
        r.html.render(timeout=40, sleep=2)
        soup = BeautifulSoup(r.html.html, "html.parser")

        # Extract only visible text (ignore script, style, nav)
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join([t.strip() for t in soup.stripped_strings if len(t.strip()) > 3])
        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

        return text, links
    except Exception as e:
        print(f"[error] Failed {url}: {e}")
        return "", []

# -------------------------------
# CRAWLER
# -------------------------------
def crawl_website(base_url):
    to_visit = [base_url]
    all_docs = []

    while to_visit and len(visited) < MAX_PAGES:
        url = to_visit.pop(0)
        if url in visited or urlparse(url).netloc != urlparse(base_url).netloc:
            continue

        print(f"[crawl] {url}")
        visited.add(url)
        text, links = scrape_page(url)

        if text:
            all_docs.append({"url": url, "text": text})

        for link in links:
            if link.startswith(base_url) and link not in visited:
                to_visit.append(link)

    return all_docs

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()

# -------------------------------
# QA GENERATION
# -------------------------------
def generate_qa_pairs(doc):
    """Generate multiple QA pairs per page using heuristic patterns."""
    text = doc["text"]
    url = doc["url"]
    qas = []

    # Templates for different business aspects
    templates = [
        ("What services does Prodapt offer?", "Look for keywords like 'services', 'solutions', 'offerings'."),
        ("Who are Prodapt’s clients or target industries?", "Look for 'clients', 'industries', or 'partners'."),
        ("What is Prodapt’s expertise or specialization?", "Look for 'expertise', 'focus areas', 'core strengths'."),
        ("What technologies does Prodapt use or build expertise in?", "Look for 'AI', 'cloud', 'automation', 'OSS/BSS'."),
        ("Does Prodapt mention pricing or cost-related information?", "Look for 'pricing', 'affordable', 'cost-effective'."),
        ("How does Prodapt ensure delivery or quality?", "Look for 'delivery', 'timeline', 'implementation'."),
        ("What is Prodapt’s global presence or office location?", "Look for 'headquarters', 'global', 'offices'."),
        ("What is Prodapt’s mission or vision?", "Look for 'mission', 'vision', 'values'."),
        ("What are the key partnerships mentioned by Prodapt?", "Look for 'partner', 'alliance', 'collaboration'."),
        ("Who are Prodapt’s leaders or executives?", "Look for 'CEO', 'leadership', 'team'."),
    ]

    # Automatically extract relevant context chunks
    for q, hint in templates:
        # Find text chunks containing hint-related words
        if any(kw.lower() in text.lower() for kw in hint.split()):
            context = " ".join(text.split()[:600])
            answer = context[:600]
            qas.append({"question": q, "answer": answer, "context": context, "source": url})

    # Fallback: general info
    if not qas:
        context = text[:600]
        qas.append({
            "question": f"What can you tell me about Prodapt from {url}?",
            "answer": context,
            "context": context,
            "source": url
        })

    return qas

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    print("[info] Starting crawl...")
    docs = crawl_website(BASE_URL)
    print(f"[info] Crawled {len(docs)} pages.")
    if not docs:
        print("[warn] No pages extracted.")
        return

    # Clean texts
    for d in docs:
        d["text"] = clean_text(d["text"])

    # Save raw docs
    with open(os.path.join(OUTPUT_DIR, "docs.jsonl"), "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Generate QA pairs
    qa_pairs = []
    for doc in docs:
        qa_pairs.extend(generate_qa_pairs(doc))

    random.shuffle(qa_pairs)
    print(f"[info] Generated {len(qa_pairs)} QA pairs.")

    # Split into train/valid/test
    n = len(qa_pairs)
    train = qa_pairs[:int(n * TRAIN_SPLIT)]
    valid = qa_pairs[int(n * TRAIN_SPLIT):int(n * (TRAIN_SPLIT + VALID_SPLIT))]
    test = qa_pairs[int(n * (TRAIN_SPLIT + VALID_SPLIT)):]

    for name, data in zip(["train", "valid", "test"], [train, valid, test]):
        path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[saved] {name}: {len(data)} samples -> {path}")

    print("\n Dataset generation completed successfully!")

if __name__ == "__main__":
    main()
