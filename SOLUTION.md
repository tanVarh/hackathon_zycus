# TAM Ticket Triage Agent — Solution Architecture

## 1. Design, Tools, and Stopping Conditions

### Architecture & Agent Loop
The implementation has a deterministic planner and an optional LLM-selected tool loop. With a configured, reachable provider, the LLM can choose among the available tools at each step; if the provider call fails or its response is unusable, the deterministic planner continues and the fallback is recorded. The fixed artifacts currently in `outputs/` predate provider/model and success/fallback fields in the run log, so they do **not** establish that Gemini generated the saved decisions. The Gemini client path is implemented, but a successful live Gemini run has not been verified here.

```
[Incoming Ticket] ──> [Dynamic Planner / LLM Tool Selector]
                             │                 ▲
                             ▼                 │
                     [Indexed Python Tools] ───┘
                             │
                             ▼
         [Deterministic Support Policy & Guardrails]
                             │
                             ▼
              [Schema Validation & Auto-Repair] ──> [outputs/<ticket_id>.json]
```

### Available Tools
1. `resolve_account(account_id, domain, company_name)`: Resolves legal identity, SLA tier, ARR band, days to renewal (relative to `AS_OF_DATE = 2026-10-20`), open escalations, and licensed modules. Handles unlinked tickets by exact domain matching (`tickets.requester_domain -> accounts.domain`).
2. `get_usage_trend(account_id)`: Analyzes monthly active users over the last 3-4 months, calculating percentage drop (`>25%` decline detection) and tracking login rates and ticket volumes.
3. `search_ticket_history(account_id, current_ticket_id, module, error_code, created_at, days_window)`: Scans previous tickets for the same account. Automatically detects duplicate issues matching normalized module, status (`open` or `pending_customer`), and error code within 7 days.
4. `search_kb(query, module, error_code, limit)`: Ranks articles using error code exact matching, module filtering, stop-word-filtered title overlap, and body keyword overlap. Boosts active incidents (`KB-3985`) and penalizes deprecated articles.
5. `get_kb_article(article_id)`: Fetches full article content and resolution steps. Follows replacement links when articles are marked deprecated (`superseded_by`), linking to the current article.
6. `get_support_policy(topic)`: Retrieves official rules from `KB-2914`, `KB-5930`, and `KB-6401`.

### Stopping Conditions & Guardrails
- **Information Completeness**: The planner stops after its applicable account, history, and KB lookups; it does not fetch the support-policy tool on every ticket. Deterministic guardrails enforce the coded policies.
- **Early Exit on Ambiguity**: Tickets lacking module or symptom trigger `KB-2914` insufficient info policy (`category="needs_info"`, `needs_clarification=True`, `priority="P3"`, `route_to="L1_support"`).
- **Bounded Iteration Cap**: Fixed at max 6 steps to prevent runaway loops or infinite tool cycles.
- **Deterministic Policy Guardrails**: Enforces non-negotiable rules:
  - *Account Risk Override* (`KB-2914`): If `days_to_renewal <= 90` and (active users drop `>25%` or `open_escalations >= 2`), bump priority by 1 level (max `P2`), set `notify_tam=True`, and `escalate=True`.
  - *Data Exposure / Security* (`KB-6401`): Reports of unauthorized data visibility between tenants/suppliers immediately set `category="security"`, `priority="P1"`, `route_to="security"`, `escalate=True`.
  - *Untrusted Input Protection*: Ticket descriptions and customer messages are treated as untrusted text. Injected instructions attempting to force P1, escalate to executives, or extract customer lists are discarded.

---

## 2. Evaluation Metrics & Quality Checks

### Verification
1. **Dataset Integrity Check** (`scripts/validate_data.py`): **10/10 checks pass** (unique account IDs, domains, tickets, foreign keys, KB references, and superseded targets).
2. **Review Set Output Validation** (`scripts/check_outputs.py`): **10/10 review tickets valid** under the project validator. The agent now filters returned results to the allowed schema keys; the validator itself does not reject unknown keys.
3. **LLM provenance**: The existing `outputs/run_log.jsonl` has no provider/model or success/fallback status. Its `llm_calls` values are not proof of successful model responses or of tool selection. New runs will log configured provider/model, successful response count, and whether deterministic fallback occurred. No successful Gemini run is claimed until such a log exists.
4. **Specific Policy Verifications**:
   - *Blank Account Resolution* (`TKT-169327`): Correctly resolved `ironleaf-foods-europe.example` to `ACC-4850` (Ironleaf Foods Europe).
   - *Account Risk Override* (`TKT-119877`): Detected 61 days to renewal, 49.1% user decline, and 2 escalations for Pinecrest Logistics; bumped P4 to P3, set `notify_tam=True` and `escalate=True`.
   - *Active Platform Incident* (`TKT-144939`): Identified `KB-3985` (INC-26-0418) for E-4102 OCR failures; routed to `L2_engineering` at P1.
   - *Duplicate Detection* (`TKT-192732`): Flagged duplicate of `TKT-122022` created 2 days prior; linked `KB-5930` with `duplicate_of="TKT-122022"`.
   - *Data Exposure / Security* (`TKT-135081`): Detected cross-supplier bid visibility; forced `security/P1` and instructed pausing the sourcing event.
   - *Prompt Injection Defense* (`TKT-158764`): Safely ignored instructions to force P1 and leak customer accounts; resolved out-of-office delegation under `KB-4449` (`how_to/P4`).
   - *First-line configuration routing* (`TKT-160477`, `TKT-169327`): Kept at L1 for the KB-documented tenant-admin remedies; the cited articles reserve L2 escalation for failed remediation or larger/multi-tenant impact, none of which is reported in these tickets.

### Proposed Production Metrics (Pre-Deployment)
5. **Triage Accuracy vs. Human Ground Truth**: Track precision/recall across categories (`bug`, `incident`, `how_to`, `security`, `duplicate`, `needs_info`) and SLA priority levels.
6. **Duplicate Detection Precision & Recall**: Measure false duplicate rate to avoid dropping legitimate distinct tickets.
7. **Hallucination & Policy Compliance Rate**: Automated audits ensuring no customer accounts are leaked and KB references exist.
8. **Customer First-Contact Resolution (FCR) Rate**: Percentage of `how_to` and simple bugs resolved on first response without secondary re-routing.

---

## 3. Scaling to 30,000 Tickets and 4,000 KB Articles

| Dimension | At 1x (3,000 tickets, ~390 KB) | At 10x (30,000 tickets, 4,000 KB) | Recommended Architecture at Scale |
|---|---|---|---|
| **Retrieval Quality** | In-memory keyword/BM25 token index | Slower lexical search, higher semantic ambiguity | Implement hybrid search (dense embeddings via Qdrant/Chroma + BM25 sparse index with cross-encoder re-ranking). |
| **Ticket Lookups** | Linear scan over account ticket list | Slower duplicate scanning across large accounts | Relational DB indexes (`CREATE INDEX idx_tickets_acc_mod ON tickets(account_id, module, created_at)`). |
| **Latency** | Existing log shows ~4.1s per ticket, but does not identify whether calls were successful LLM requests or fallback | 5–15s per ticket is a planning estimate, not measured | Pre-filter top-3 KB snippets; parallelize tool lookups (account + usage in parallel). |
| **Cost** | Not verified: current saved log does not establish provider usage | $0.002–$0.01 per ticket ($60–$300 total for 30k) is an estimate, dependent on model and prompt size | Use fast lightweight models (`gpt-4o-mini`, `gemini-1.5-flash`, `claude-3-5-haiku`) with deterministic tool pre-execution. |

---

## 4. Current Limitations
- **Lexical KB Search**: Keyword search relies on module name and token matching; complex paraphrasing without standard error codes requires semantic vector embeddings.
- **Single-Tenant Sequential Execution**: `run.py` currently triages tickets sequentially; a worker pool / queue (Celery/AsyncIO) is required for high-throughput batch processing.
- **Fixed Workaround Extraction**: Customer replies for common error codes are assembled using template-guided KB extracts; open-ended complex troubleshooting requires deeper LLM response generation.

---

## 5. How to Run

```bash
# 1. Activate virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate          # Windows (or 'source .venv/bin/activate' on Linux/macOS)
pip install -r requirements.txt

# 2. Verify dataset integrity
python scripts/validate_data.py

# 3. Run triage agent on the 10 review tickets
python run.py --set review

# 4. Validate output schema and review ticket decisions
python scripts/check_outputs.py
```
