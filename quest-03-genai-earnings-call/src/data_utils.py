"""
Data utilities for Energy Earnings Call Analyst.

Downloads SEC filing data for energy companies via edgar-sec,
creates instruction-tuning pairs, and prepares datasets for LoRA fine-tuning.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, set_seed

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_DATA_DIR = PROJECT / "data"
DEFAULT_OUTPUT_DIR = PROJECT / "results"

# ── Energy companies to track ────────────────────────────────────
ENERGY_COMPANIES: List[Dict[str, str | int]] = [
    {"name": "Exxon Mobil", "ticker": "XOM", "cik": 34088},
    {"name": "Chevron", "ticker": "CVX", "cik": 93410},
    {"name": "ConocoPhillips", "ticker": "COP", "cik": 1163165},
    {"name": "EOG Resources", "ticker": "EOG", "cik": 821189},
    {"name": "Devon Energy", "ticker": "DVN", "cik": 1090012},  # Replaces Pioneer (404)
    {"name": "Occidental Petroleum", "ticker": "OXY", "cik": 797468},
    {"name": "Schlumberger", "ticker": "SLB", "cik": 87347},
    {
        "name": "Halliburton",
        "ticker": "HAL",
        "cik": 45012,
    },  # Replaces Baker Hughes (404)
]

# ── Financial metrics to extract (GAAP tags) ─────────────────────
FINANCIAL_METRICS = {
    "Revenue": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "Net Income": "us-gaap:NetIncomeLoss",
    "Total Assets": "us-gaap:Assets",
    "Total Liabilities": "us-gaap:Liabilities",
    "Operating Income": "us-gaap:OperatingIncomeLoss",
    "Earnings Per Share": "us-gaap:EarningsPerShareBasic",
    "Cash and Equivalents": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "Long Term Debt": "us-gaap:LongTermDebt",
    "Gross Profit": "us-gaap:GrossProfit",
    "Research and Development": "us-gaap:ResearchAndDevelopmentExpense",
}

# ── Instruction templates ────────────────────────────────────────
INSTRUCTION_TEMPLATES = [
    "What was {company}'s {metric} for {period}?",
    "Extract the {metric} for {company} in {period}.",
    "Looking at {company}'s {period} financials, what is the {metric}?",
    "Provide the {metric} figure reported by {company} for {period}.",
    "What amount did {company} report as {metric} in {period}?",
]

SUMMARY_TEMPLATES = [
    "Summarize the key financial results for {company} for {period} based on their SEC filing.",
    "Provide a financial overview of {company} for {period} including revenue, net income, and key metrics.",
    "Analyze {company}'s financial performance for {period}.",
]

# TinyLlama prompt format, for training we specify the three response field, for inference we leave it blank and the model generates the end of the sequence/response
# We do not need to specify a response format since our training uses a set format and the model will learn to match it
# {input} is for the SEC filing text excerpt that provides context for the question.
# During training, the model learns to extract the answer from the provided filing text.
# For inference, we would similarly provide the relevant filing text as context.
TINYLLAMA_PROMPT = """<|system|>
You are a financial analyst specializing in energy company earnings. Answer questions accurately based on SEC filing data.</s>
<|user|>
{instruction}

{input}</s>
<|assistant|>
{response}</s>"""


def _format_currency(value: float) -> str:
    """Format a large number into a readable currency string."""
    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"


SEC_HEADERS = {
    "User-Agent": "ML_side_quests/1.0 (allan.example@email.com)",
    "Accept-Encoding": "gzip, deflate",
}


def fetch_financial_data(
    companies: List[Dict] = None,
    metrics: Dict[str, str] = None,
    max_fiscal_years: int = 3,
    cache_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Fetch XBRL financial data for energy companies directly from SEC EDGAR API.

    Uses the public SEC company facts API (no external library required).
    Returns a list of records: {company, ticker, metric, period, value, unit}
    """
    if companies is None:
        companies = ENERGY_COMPANIES
    if metrics is None:
        metrics = FINANCIAL_METRICS

    all_records = []

    # Check cache first
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "financial_data.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("companies") == [c["ticker"] for c in companies]:
                print(
                    f"[CACHE] Loaded {len(cached['records'])} financial records from cache"
                )
                return cached["records"]

    # ── Fetch from SEC public API ─────────────────────────────
    for company in companies:
        cik_padded = str(company["cik"]).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"

        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if resp.status_code != 200:
                print(f"  [SKIP] {company['ticker']}: HTTP {resp.status_code}")
                time.sleep(0.2)
                continue

            data = resp.json()
            us_gaap = data.get("facts", {}).get("us-gaap", {})

            for metric_name, gaap_tag in metrics.items():
                tag = gaap_tag.split(":")[
                    1
                ]  # e.g. "RevenueFromContractWithCustomerExcludingAssessedTax"
                concept = us_gaap.get(tag, {})
                units = concept.get("units", {})

                for unit_name, values in units.items():
                    # Only take USD-denominated values
                    if "USD" not in unit_name and unit_name != "USD":
                        continue

                    # Sort by end date descending, take most recent
                    sorted_vals = sorted(
                        values, key=lambda v: v.get("end", ""), reverse=True
                    )
                    years_seen = 0
                    for v in sorted_vals:
                        if "end" not in v:
                            continue
                        if years_seen >= max_fiscal_years:
                            break

                        all_records.append(
                            {
                                "company": company["name"],
                                "ticker": company["ticker"],
                                "metric": metric_name,
                                "period": v["end"],
                                "fy": v.get("fy", int(v["end"][:4])),
                                "fp": v.get("fp", "FY"),
                                "value": v.get("val", 0),
                                "unit": unit_name,
                            }
                        )
                        years_seen += 1

            print(f"  [OK] {company['ticker']}: fetched facts")
            time.sleep(0.15)  # Rate limit: ~6 req/s max

        except requests.exceptions.RequestException as e:
            print(f"  [WARN] {company['ticker']}: {e}")
            time.sleep(0.2)
            continue

    if all_records:
        print(f"  [OK] Fetched {len(all_records)} records from SEC EDGAR")
    else:
        print("  [WARN]  No financial data returned from SEC API.")

    # Cache results (even if empty, to avoid hammering API)
    if cache_dir is not None:
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "companies": [c["ticker"] for c in companies],
                    "records": all_records,
                },
                f,
                indent=2,
            )

    return all_records


def _generate_mock_data(
    companies: List[Dict],
    metrics: Dict[str, str],
    max_fiscal_years: int = 3,
) -> List[Dict]:
    """Generate realistic mock financial data for development/testing."""
    rng = np.random.default_rng(42)
    records = []

    # Approximate real financial data ranges for energy companies
    company_ranges = {
        "XOM": {
            "revenue": (300e9, 400e9),
            "net_income": (20e9, 60e9),
            "assets": (350e9, 400e9),
        },
        "CVX": {
            "revenue": (150e9, 250e9),
            "net_income": (15e9, 35e9),
            "assets": (250e9, 260e9),
        },
        "COP": {
            "revenue": (50e9, 80e9),
            "net_income": (5e9, 15e9),
            "assets": (90e9, 100e9),
        },
        "EOG": {
            "revenue": (15e9, 30e9),
            "net_income": (3e9, 8e9),
            "assets": (35e9, 45e9),
        },
        "PXD": {
            "revenue": (10e9, 25e9),
            "net_income": (2e9, 6e9),
            "assets": (20e9, 30e9),
        },
        "OXY": {
            "revenue": (20e9, 40e9),
            "net_income": (2e9, 13e9),
            "assets": (70e9, 80e9),
        },
        "SLB": {
            "revenue": (25e9, 35e9),
            "net_income": (2e9, 6e9),
            "assets": (40e9, 45e9),
        },
        "BKR": {
            "revenue": (20e9, 25e9),
            "net_income": (1e9, 3e9),
            "assets": (30e9, 35e9),
        },
    }

    current_year = 2025
    for company in companies:
        ranges = company_ranges.get(company["ticker"], {})
        for fy_offset in range(max_fiscal_years):
            year = current_year - fy_offset
            for metric_name in metrics.keys():
                if metric_name == "Revenue":
                    lo, hi = ranges.get("revenue", (10e9, 50e9))
                elif metric_name == "Net Income":
                    lo, hi = ranges.get("net_income", (1e9, 10e9))
                elif metric_name == "Total Assets":
                    lo, hi = ranges.get("assets", (20e9, 100e9))
                elif metric_name == "Total Liabilities":
                    a, b = ranges.get("assets", (10e9, 60e9))
                    lo, hi = a * 0.3, b * 0.6
                elif metric_name == "Operating Income":
                    lo, hi = ranges.get("net_income", (1e9, 10e9))
                elif metric_name == "Earnings Per Share":
                    lo, hi = (3.0, 12.0)
                elif metric_name == "Cash and Equivalents":
                    lo, hi = (3e9, 30e9)
                elif metric_name == "Long Term Debt":
                    lo, hi = (5e9, 40e9)
                elif metric_name == "Gross Profit":
                    a, b = ranges.get("revenue", (10e9, 50e9))
                    lo, hi = a * 0.3, b * 0.6
                elif metric_name == "Research and Development":
                    lo, hi = (200e6, 1e9)
                else:
                    lo, hi = (1e9, 10e9)

                value = rng.uniform(lo, hi)
                records.append(
                    {
                        "company": company["name"],
                        "ticker": company["ticker"],
                        "metric": metric_name,
                        "period": f"{year}-12-31",
                        "fy": year,
                        "fp": "FY",
                        "value": round(value, 2),
                        "unit": "USD",
                    }
                )
    return records


def create_instruction_pairs(
    financial_records: List[Dict],
    max_samples: int = 500,
    seed: int = 42,
    filing_texts_by_company: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    """
    Convert financial records into instruction-response pairs.

    Each pair is: {instruction, input, response, company, ticker, metric, period}
    If filing_texts_by_company is provided, the relevant filing excerpt
    is placed in the {input} field so the model learns to extract answers
    from the document context rather than just memorizing values.
    """
    rng = np.random.default_rng(seed)
    pairs: List[Dict] = []

    # ── 1. Individual metric Q&A ────────────────────────────────
    for record in financial_records:
        company = record["company"]
        metric = record["metric"]
        period = record.get("period", "")
        value = record["value"]
        ticker = record["ticker"]

        # Format year nicely
        year_str = period[:4] if period else "the most recent fiscal year"
        period_label = f"FY{record.get('fy', year_str)}"

        # Pick a random instruction template
        template = rng.choice(INSTRUCTION_TEMPLATES)
        instruction = template.format(
            company=company, metric=metric.lower(), period=period_label
        )

        # Format the response
        formatted_value = _format_currency(value)
        response = f"For {period_label}, {company} ({ticker}) reported {metric.lower()} of {formatted_value}."

        # Pick a relevant SEC filing text excerpt as context for this Q&A pair
        input_text = ""
        if filing_texts_by_company and company in filing_texts_by_company:
            company_texts = filing_texts_by_company[company]
            if company_texts:
                input_text = rng.choice(company_texts)

        pairs.append(
            {
                "instruction": instruction,
                "input": input_text,
                "response": response,
                "company": company,
                "ticker": ticker,
                "metric": metric,
                "period": period_label,
                "value": value,
            }
        )

    # ── 2. Multi-metric summary questions ───────────────────────
    # Group records by company + fiscal year
    from collections import defaultdict

    by_company_year = defaultdict(list)
    for rec in financial_records:
        key = (rec["company"], rec["ticker"], rec.get("fy", 0))
        by_company_year[key].append(rec)

    for (company, ticker, fy), recs in by_company_year.items():
        period_label = f"FY{fy}"
        template = rng.choice(SUMMARY_TEMPLATES)
        instruction = template.format(company=company, period=period_label)

        # Build a structured summary
        metrics_dict = {r["metric"]: r["value"] for r in recs}
        lines = [f"**Financial Summary for {company} ({ticker}) — {period_label}**"]
        for metric_name in FINANCIAL_METRICS.keys():
            if metric_name in metrics_dict:
                lines.append(
                    f"- {metric_name}: {_format_currency(metrics_dict[metric_name])}"
                )

        response = "\n".join(lines)

        # Pick a relevant SEC filing text excerpt as context for this summary pair
        summary_input = ""
        if filing_texts_by_company and company in filing_texts_by_company:
            company_texts = filing_texts_by_company[company]
            if company_texts:
                summary_input = rng.choice(company_texts)

        pairs.append(
            {
                "instruction": instruction,
                "input": summary_input,
                "response": response,
                "company": company,
                "ticker": ticker,
                "metric": "summary",
                "period": period_label,
            }
        )

    # ── Shuffle and limit ────────────────────────────────────────
    rng.shuffle(pairs)
    if max_samples and len(pairs) > max_samples:
        pairs = pairs[:max_samples]
        print(
            f"  Sampled {max_samples} instruction pairs (from {len(financial_records)} records)"
        )

    return pairs


def format_for_tinyllama(pairs: List[Dict]) -> List[str]:
    """Format instruction pairs into TinyLlama's prompt format."""
    formatted = []
    for pair in pairs:
        text = TINYLLAMA_PROMPT.format(
            instruction=pair["instruction"],
            input=pair["input"],
            response=pair["response"],
        )
        formatted.append(text)
    return formatted


def tokenize_and_split(
    formatted_texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Tokenize formatted texts and split into train/validation datasets.
    """
    set_seed(seed)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # Create HF Dataset
    dataset = Dataset.from_dict({"text": formatted_texts})
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Split
    split = tokenized.train_test_split(test_size=val_ratio, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])


def download_filing_texts(
    companies: List[Dict] = None,
    form_types: Tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    max_filings_per_company: int = 2,
    cache_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Download raw SEC filing text for domain-adaptive pretraining.
    Uses the SEC submissions API (no external library required).

    Returns a list of dicts: {company, ticker, text}
    """
    if companies is None:
        companies = ENERGY_COMPANIES

    if cache_dir is not None:
        cache_path = cache_dir / "filing_texts.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

    all_texts = []

    for company in companies:
        print(f"  Downloading filings for {company['name']}...")
        cik_padded = str(company["cik"]).zfill(10)

        try:
            # ── Step 1: Get recent submissions via SEC API ──────
            sub_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
            sub_resp = requests.get(sub_url, headers=SEC_HEADERS, timeout=30)
            if sub_resp.status_code != 200:
                print(f"    [SKIP] HTTP {sub_resp.status_code}")
                time.sleep(0.2)
                continue

            sub_data = sub_resp.json()
            recent = sub_data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accession_numbers = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])
            filing_dates = recent.get("filingDate", [])

            # ── Step 2: Filter by form type ─────────────────────
            download_targets = []
            for i, form in enumerate(forms):
                if (
                    form in form_types
                    and len(download_targets) < max_filings_per_company
                ):
                    acc_no = accession_numbers[i] if i < len(accession_numbers) else ""
                    primary = primary_docs[i] if i < len(primary_docs) else ""
                    fdate = filing_dates[i] if i < len(filing_dates) else ""
                    download_targets.append(
                        {
                            "accession": acc_no,
                            "primary": primary,
                            "date": fdate,
                            "form": form,
                        }
                    )

            if not download_targets:
                print(f"    [SKIP] No matching filings found")
                time.sleep(0.15)
                continue

            # ── Step 3: Download each filing ────────────────────
            base_url = f"https://www.sec.gov/Archives/edgar/data/{company['cik']}"
            count = 0
            for target in download_targets:
                acc_clean = target["accession"].replace("-", "")
                doc_url = f"{base_url}/{acc_clean}/{target['primary']}"

                doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=60)
                if doc_resp.status_code != 200:
                    # Fallback: try the .txt version of the primary document
                    txt_name = (
                        target["primary"].rsplit(".", 1)[0] + ".txt"
                        if "." in target["primary"]
                        else target["primary"] + ".txt"
                    )
                    doc_url = f"{base_url}/{acc_clean}/{txt_name}"
                    doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=60)

                if doc_resp.status_code == 200:
                    text = doc_resp.text
                    # Clean HTML tags
                    clean = re.sub(r"<[^>]+>", " ", text)
                    clean = re.sub(r"\s+", " ", clean).strip()
                    # Only keep substantial chunks
                    if len(clean) > 1000:
                        chunks = _chunk_text(clean, chunk_size=2000)
                        for chunk in chunks:
                            all_texts.append(
                                {
                                    "company": company["name"],
                                    "ticker": company["ticker"],
                                    "text": chunk,
                                }
                            )
                        count += 1
                        print(
                            f"    [OK] {target['form']} {target['date']} ({len(chunks)} chunks)"
                        )

                time.sleep(0.2)  # Rate limit

            if count == 0:
                print(f"    [SKIP] Could not download any filing documents")

        except Exception as e:
            print(f"    [WARN]  {company['ticker']}: {e}")
            continue

    print(f"  Downloaded {len(all_texts)} text chunks total")

    if cache_dir is not None:
        with open(cache_path, "w") as f:
            json.dump(all_texts, f, indent=2)

    return all_texts


def _chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    overlap = 200
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def prepare_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_samples: int = 500,
    max_length: int = 512,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    End-to-end dataset preparation:
    1. Fetch financial data from SEC (or use mock data)
    2. Create instruction pairs
    3. Format for TinyLlama
    4. Tokenize and split
    """
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[DATA] Preparing Earnings Call Analyst Dataset")
    print("=" * 60)

    # Step 1: Fetch financial data
    print("\n1️⃣  Fetching financial data for energy companies...")
    records = fetch_financial_data(cache_dir=cache_dir)

    # Step 2: Download filing texts for context
    print("\n2️⃣  Downloading filing texts for context...")
    filing_texts_by_company: Dict[str, List[str]] = {}
    try:
        filing_texts = download_filing_texts(cache_dir=cache_dir)
        if filing_texts:
            for item in filing_texts:
                company_name = item["company"]
                if company_name not in filing_texts_by_company:
                    filing_texts_by_company[company_name] = []
                filing_texts_by_company[company_name].append(item["text"])
            print(f"   Grouped texts for {len(filing_texts_by_company)} companies")
    except Exception as e:
        print(f"  [WARN]  Could not download filing texts: {e}")

    # Step 3: Create instruction pairs
    print("\n3️⃣  Creating instruction-response pairs...")
    pairs = create_instruction_pairs(
        records,
        max_samples=max_samples,
        seed=seed,
        filing_texts_by_company=filing_texts_by_company
        if filing_texts_by_company
        else None,
    )

    # Save raw pairs for inspection
    pairs_path = output_dir / "instruction_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2, default=str)
    print(f"   Saved {len(pairs)} pairs to {pairs_path}")

    # Step 4: Save label info (company list, metrics, etc.)
    label_info = {
        "companies": [c["name"] for c in ENERGY_COMPANIES],
        "tickers": [c["ticker"] for c in ENERGY_COMPANIES],
        "metrics": list(FINANCIAL_METRICS.keys()),
        "total_records": len(records),
        "total_pairs": len(pairs),
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    label_info_path = output_dir / "label_info.json"
    with open(label_info_path, "w") as f:
        json.dump(label_info, f, indent=2)
    print(f"   Label info saved to {label_info_path}")

    print("\n[OK] Dataset preparation complete!")
    return pairs, label_info


if __name__ == "__main__":
    prepare_dataset()
