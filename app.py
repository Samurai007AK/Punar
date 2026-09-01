"""Punar judge console -- one-command local demo UI.

Run:
    streamlit run app.py

Three tabs, zero duplicated business logic -- every tab calls straight into
the same functions the CLI and API use:

    Single case     -> punar.core.agent.run_agent      (same as `punar recover`)
    Batch results   -> punar.benchmark.run_benchmark    (same as `punar report`)
    Audit lab       -> punar.audit.AuditStore.verify_chain (same as `punar audit`)

Offline-first: no network calls. Everything here runs against the local
simulator, a local policy.json and a local SQLite file.
"""
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

# Allow running straight from a checkout without installing the package,
# same trick scripts/benchmark.py uses.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from punar.audit import AuditStore
from punar.core.agent import run_agent
from punar.core.gate import load_policy

st.set_page_config(page_title="Punar judge console", layout="wide")

POLICY_PATH = "punar/config/policy.json"
DEMO_RUN_DIR = "outputs/demo_run"
DEMO_AUDIT_DB = "outputs/demo_audit.db"
IST = ZoneInfo("Asia/Kolkata")

st.title("Punar judge console")
st.caption("Every view below calls the same functions punar.main / the API use. "
           "Nothing here is reimplemented for the demo.")

tab_case, tab_batch, tab_audit = st.tabs(["Single case", "Batch results", "Audit lab"])

# --------------------------------------------------------------------- tab 1
_BASE_CASE = {"amount_inr": 2499.0, "method": "upi", "merchant_name": "Demo Merchant",
              "language": "en", "opted_out": False, "error_code": "BAD_REQUEST_ERROR",
              "error_description": "insufficient balance in the account",
              "error_reason": "INSUFFICIENT_FUNDS"}

EXAMPLES = {
    "Insufficient funds -- recovers on the 3rd touch": {
        "case": {**_BASE_CASE, "case_id": "demo_ok", "customer_id": "cust_demo_ok"},
        "outcomes": [False, False, True], "now": None},
    "Fraud block -- never contacted (non-retriable)": {
        "case": {**_BASE_CASE, "case_id": "demo_fraud", "customer_id": "cust_demo_fraud",
                 "method": "card", "error_code": "GATEWAY_ERROR",
                 "error_description": "fraud block", "error_reason": "FRAUD"},
        "outcomes": [], "now": None},
    "Opted-out customer -- never contacted": {
        "case": {**_BASE_CASE, "case_id": "demo_optout", "customer_id": "cust_demo_optout",
                 "opted_out": True},
        "outcomes": [], "now": None},
    "Outside contact hours -- 05:40 IST, deferred": {
        "case": {**_BASE_CASE, "case_id": "demo_offhours", "customer_id": "cust_demo_offhours"},
        "outcomes": [False, False, True],
        "now": datetime(2026, 8, 29, 5, 40, tzinfo=IST)},
}

with tab_case:
    st.subheader("Run one failed payment through the agent")
    choice = st.selectbox("Example", [*EXAMPLES, "Paste your own JSON"])

    if choice == "Paste your own JSON":
        raw = st.text_area("Case JSON", height=160,
                           placeholder='{"case_id": "...", "amount_inr": 2499.0, '
                                      '"error_reason": "INSUFFICIENT_FUNDS", ...}')
        outcomes_raw = st.text_input("Scripted per-touch outcomes (comma-separated true/false)")
        case = json.loads(raw) if raw.strip() else None
        outcomes = [tok.strip().lower() in ("1", "true", "yes")
                   for tok in outcomes_raw.split(",") if tok.strip()]
        now_dt = None
    else:
        spec = EXAMPLES[choice]
        case, outcomes, now_dt = spec["case"], spec["outcomes"], spec["now"]
        st.json(case)

    if st.button("Run through the agent"):
        if case is None:
            st.error("Paste a case JSON first.")
        else:
            policy = load_policy(POLICY_PATH)
            outcome_iter = iter(outcomes)

            def simulate(_case, _intervention, _now):
                return next(outcome_iter, False)

            state = run_agent(case, policy, random.Random(42), simulate, now=now_dt)

            st.markdown(f"**diagnosis:** `{state['diagnosis']['reason']}` "
                       f"({state['diagnosis']['retriability']})  |  "
                       f"**outcome:** `{state['outcome']}` (`{state['exit_code']}`)")

            st.markdown("**decision trail**")
            for entry in state["audit"]:
                st.text(f"{entry['at']}  {entry['step']:<9} "
                       f"{json.dumps(entry['detail'], default=str)}")

            if state["touch_history"]:
                st.markdown("**actions taken**")
                for t in state["touch_history"]:
                    reach = "customer" if t.get("contacts_customer") else "internal"
                    st.text(f"#{t['round']} {t['intervention']:<28} {t['channel']:<16} "
                           f"[{reach}] {t.get('delivery_status', '')}")
            else:
                st.info("No touch was sent for this case.")

            if state["blocked_actions"]:
                st.markdown("**blocked pre-send by the policy judge**")
                for b in state["blocked_actions"]:
                    st.warning(f"{b['intervention']}: {', '.join(b['violations'])}")

# --------------------------------------------------------------------- tab 2
with tab_batch:
    st.subheader("Batch benchmark results")
    json_path = os.path.join(DEMO_RUN_DIR, "benchmark.json")

    if not os.path.exists(json_path):
        st.warning(f"No results yet. Run `python -m punar.main report --out {DEMO_RUN_DIR}` "
                   "once, or generate a quick one below.")
    if st.button("Run a quick benchmark now (250 cases x 10 seeds)"):
        from punar.benchmark import run_benchmark, to_json_safe
        from punar.charts import generate_charts
        from punar.sim.params import SimParams
        with st.spinner("Running..."):
            result = run_benchmark(seeds=10, params=SimParams())
            os.makedirs(DEMO_RUN_DIR, exist_ok=True)
            generate_charts(result, DEMO_RUN_DIR)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(to_json_safe(result), fh, indent=2, default=str)
        st.success("Done.")
        st.rerun()

    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as fh:
            result = json.load(fh)
        rows = []
        for name in result["arms"]:
            s = result["summary"][name]
            rr = s["recovery_rate"]
            rows.append({
                "arm": result["labels"].get(name, name),
                "recovery %": f"{rr['point'] * 100:.1f} "
                             f"[{rr['ci_lo'] * 100:.1f}, {rr['ci_hi'] * 100:.1f}]",
                "net revenue (INR)": f"{s['net_revenue_inr']['point']:,.0f}",
                "gross revenue (INR)": f"{s['gross_revenue_inr']['point']:,.0f}",
                "touches / recovery": f"{s['touches_per_recovery']['point']:.2f}",
                "opt-out contacts": f"{s['opt_out_violations']['point']:.0f}",
                "non-retriable contacts": f"{s['non_retriable_touches']['point']:.0f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Simulated outcomes under stated priors, not measured Razorpay results. "
                   "95% bootstrap CIs over seeds.")

        pngs = sorted(p for p in os.listdir(DEMO_RUN_DIR) if p.endswith(".png"))
        if pngs:
            st.markdown("**charts**")
            cols = st.columns(3)
            for i, name in enumerate(pngs):
                cols[i % 3].image(os.path.join(DEMO_RUN_DIR, name),
                                  caption=name.replace(".png", "").replace("_", " "))

# --------------------------------------------------------------------- tab 3
with tab_audit:
    st.subheader("Audit trail: append-only, hash-chained, tamper-evident")
    st.caption("Every state change is a new row; verify_chain() recomputes the hash "
              "chain and detects any edit, insertion, reordering or deletion.")

    if st.button("Run audit check"):
        if not os.path.exists(DEMO_AUDIT_DB):
            os.makedirs(os.path.dirname(DEMO_AUDIT_DB), exist_ok=True)
            with AuditStore(DEMO_AUDIT_DB) as store:
                for i in range(3):
                    store.append({"case_id": f"demo-{i}", "step": "diagnose",
                                 "reason": "insufficient_funds", "round": i})
        with AuditStore(DEMO_AUDIT_DB) as store:
            result = store.verify_chain()
        if result.ok:
            st.success(f"OK -- {result.rows_checked} rows verified, chain intact.")
        else:
            st.error(f"TAMPERED -- {result.problems}")

    if st.button("Simulate tampering"):
        if not os.path.exists(DEMO_AUDIT_DB):
            st.warning("Run the audit check once first to create a demo trail.")
        else:
            tampered = tempfile.mktemp(suffix=".db")
            shutil.copyfile(DEMO_AUDIT_DB, tampered)
            conn = sqlite3.connect(tampered)
            # The trigger blocks this from any normal connection -- dropping it
            # here simulates an attacker who bypasses the app and edits the file
            # directly, which is exactly the scenario hash-chaining defends against.
            conn.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
            conn.execute("UPDATE audit_log SET data = REPLACE(data, 'insufficient_funds', "
                         "'account_closed') WHERE id = (SELECT MIN(id) FROM audit_log)")
            conn.commit()
            conn.close()
            with AuditStore(tampered) as store:
                result = store.verify_chain()
            os.remove(tampered)
            if result.ok:
                st.error("Tamper not detected -- this should not happen.")
            else:
                st.warning(f"Tampering DETECTED on a copy of the trail: {result.problems}")
                st.caption("The live database was never touched: this ran against a "
                          "throwaway copy with the append-only trigger removed.")
