"""
SOC Monitoring Cockpit for Fraud Detection System
Gradio-based interface with real API integration for transaction analysis and investigation.
"""
import gradio as gr
import requests
import json
import time
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "5"))

# --- API Helpers ---

def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def api_analyze(txn: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE_URL}/api/v1/analyze", json=txn, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        logger.warning(f"API analyze failed: {e}")
        return None

def api_stats() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except:
        return {}

def api_recent() -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/recent-transactions", timeout=5)
        if r.status_code == 200:
            return r.json().get("transactions", [])
    except:
        pass
    return []

# --- Demo / Fallback Data ---

SAMPLE_TRANSACTIONS = [
    {"transaction_id": "TXN_DEMO_001", "user_id": "USR001", "transaction_amount": 45.99, "transaction_type": "Online", "timestamp": datetime.now().isoformat(), "account_balance": 1234.56, "device_type": "Mobile", "location": "New York", "merchant_category": "Grocery", "ip_address_flag": 0, "previous_fraudulent_activity": 0, "daily_transaction_count": 2, "avg_transaction_amount_7d": 52.30, "failed_transaction_count_7d": 0, "card_type": "Visa", "card_age": 365, "transaction_distance": 0.0, "authentication_method": "Biometric", "risk_score": 0.05, "is_weekend": 0},
    {"transaction_id": "TXN_DEMO_002", "user_id": "USR002", "transaction_amount": 2500.00, "transaction_type": "Online", "timestamp": datetime.now().isoformat(), "account_balance": 500.00, "device_type": "Unknown", "location": "International", "merchant_category": "Luxury Jewelry", "ip_address_flag": 1, "previous_fraudulent_activity": 1, "daily_transaction_count": 8, "avg_transaction_amount_7d": 125.00, "failed_transaction_count_7d": 5, "card_type": "Amex", "card_age": 1, "transaction_distance": 5000.0, "authentication_method": "Password", "risk_score": 0.95, "is_weekend": 1},
    {"transaction_id": "TXN_DEMO_003", "user_id": "USR003", "transaction_amount": 750.00, "transaction_type": "Online", "timestamp": datetime.now().isoformat(), "account_balance": 2000.00, "device_type": "Tablet", "location": "Chicago", "merchant_category": "Electronics", "ip_address_flag": 0, "previous_fraudulent_activity": 0, "daily_transaction_count": 1, "avg_transaction_amount_7d": 150.00, "failed_transaction_count_7d": 1, "card_type": "Mastercard", "card_age": 45, "transaction_distance": 500.0, "authentication_method": "OTP", "risk_score": 0.45, "is_weekend": 0},
    {"transaction_id": "TXN_DEMO_004", "user_id": "USR004", "transaction_amount": 12.50, "transaction_type": "POS", "timestamp": datetime.now().isoformat(), "account_balance": 3500.00, "device_type": "POS Terminal", "location": "San Francisco", "merchant_category": "Coffee Shop", "ip_address_flag": 0, "previous_fraudulent_activity": 0, "daily_transaction_count": 1, "avg_transaction_amount_7d": 15.00, "failed_transaction_count_7d": 0, "card_type": "Visa", "card_age": 720, "transaction_distance": 0.0, "authentication_method": "Chip", "risk_score": 0.01, "is_weekend": 0},
    {"transaction_id": "TXN_DEMO_005", "user_id": "USR005", "transaction_amount": 3200.00, "transaction_type": "Wire Transfer", "timestamp": datetime.now().isoformat(), "account_balance": 50000.00, "device_type": "Desktop", "location": "Miami", "merchant_category": "Real Estate", "ip_address_flag": 0, "previous_fraudulent_activity": 0, "daily_transaction_count": 1, "avg_transaction_amount_7d": 1800.00, "failed_transaction_count_7d": 0, "card_type": "Amex", "card_age": 1095, "transaction_distance": 0.0, "authentication_method": "Biometric", "risk_score": 0.25, "is_weekend": 0},
]

def demo_transactions_df() -> pd.DataFrame:
    rows = []
    for t in SAMPLE_TRANSACTIONS:
        risk = t["risk_score"]
        indicator = "CRITICAL" if risk >= 0.8 else "HIGH" if risk >= 0.6 else "MEDIUM" if risk >= 0.4 else "LOW"
        verdict = "BLOCK" if risk >= 0.8 else "REVIEW" if risk >= 0.6 else "APPROVE"
        rows.append({
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Risk": indicator,
            "Transaction ID": t["transaction_id"],
            "Amount": f"${t['transaction_amount']:.2f}",
            "Merchant": t["merchant_category"],
            "Risk Score": risk,
            "Verdict": verdict,
            "Status": "Analyzed",
            "_raw": t,
        })
    return pd.DataFrame(rows)

def _risk_style(val: float) -> str:
    if val >= 0.8:
        return "CRITICAL"
    elif val >= 0.6:
        return "HIGH"
    elif val >= 0.4:
        return "MEDIUM"
    return "LOW"

def _verdict(val: float) -> str:
    if val >= 0.8:
        return "BLOCK"
    elif val >= 0.6:
        return "REVIEW"
    return "APPROVE"

def txn_to_row(txn: dict) -> dict:
    risk = txn.get("risk_score", 0.5) if isinstance(txn.get("risk_score"), (int, float)) else float(txn.get("risk_score", 0.5))
    return {
        "Timestamp": txn.get("timestamp", txn.get("Timestamp", datetime.now().strftime("%H:%M:%S"))),
        "Risk": _risk_style(risk),
        "Transaction ID": txn.get("transaction_id", txn.get("Transaction ID", "N/A")),
        "Amount": f"${float(txn.get('transaction_amount', txn.get('Amount', 0))):.2f}",
        "Merchant": txn.get("merchant_category", txn.get("Merchant", "N/A")),
        "Risk Score": risk,
        "Verdict": _verdict(risk),
        "Status": "Analyzed",
        "_raw": txn,
    }

def build_txn_table(transactions: list) -> pd.DataFrame:
    rows = [txn_to_row(t) for t in transactions]
    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Risk", "Transaction ID", "Amount", "Merchant", "Risk Score", "Verdict", "Status"])
    return pd.DataFrame(rows)

def color_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add HTML color styling — Gradio DataFrames support cell styling via 'style' attribute."""
    return df

# --- Core UI Callbacks ---

def refresh_dashboard(state_df: pd.DataFrame, risk_filter: str, search_term: str):
    """Main refresh: fetch transactions + stats, apply filters, return all UI components."""
    connected = api_health()

    # Attempt to fetch from API
    raw_txns = api_recent() if connected else []
    if not raw_txns:
        raw_txns = [t for t in SAMPLE_TRANSACTIONS]

    full_df = build_txn_table(raw_txns)

    # Filter
    display_df = full_df.copy()
    if risk_filter and risk_filter != "All":
        if risk_filter == "Critical":
            display_df = display_df[display_df["Risk Score"] >= 0.8]
        elif risk_filter == "High":
            display_df = display_df[display_df["Risk Score"] >= 0.6]
        elif risk_filter == "Medium":
            display_df = display_df[display_df["Risk Score"] >= 0.4]

    if search_term:
        st = search_term.strip().lower()
        display_df = display_df[
            display_df["Transaction ID"].str.lower().str.contains(st) |
            display_df["Merchant"].str.lower().str.contains(st)
        ]

    # Remove _raw before display
    display_cols = [c for c in ["Timestamp", "Risk", "Transaction ID", "Amount", "Merchant", "Risk Score", "Verdict", "Status"] if c in display_df.columns]
    display_out = display_df[display_cols] if not display_df.empty else pd.DataFrame(columns=display_cols)

    # Stats
    stats = api_stats() if connected else {}
    total = stats.get("total_transactions_processed", 0) if connected else len(SAMPLE_TRANSACTIONS)
    fraud_rate_val = stats.get("fraud_detection_rate", 0.0) if connected else 0.25
    avg_time_val = stats.get("average_processing_time", 0.0) if connected else 1.2
    uptime_val = stats.get("system_uptime", 0.0) if connected else 3600.0

    uptime_str = f"{uptime_val:.0f}s" if uptime_val < 60 else f"{uptime_val/60:.1f}m"
    models_ok = sum(1 for v in stats.get("model_status", {}).values() if v == "loaded") if connected else 0

    # Critical alert
    critical_count = len(full_df[full_df["Risk"] == "CRITICAL"])
    alert_visible = critical_count > 0

    return (
        full_df,
        display_out,
        alert_visible,
        f"**Total Transactions:** {total}",
        f"**Fraud Rate:** {fraud_rate_val:.1%}" if isinstance(fraud_rate_val, float) else f"**Fraud Rate:** {fraud_rate_val}",
        f"**Avg Processing:** {avg_time_val:.3f}s" if isinstance(avg_time_val, float) else f"**Avg Processing:** {avg_time_val}",
        f"**Uptime:** {uptime_str}",
        f"**Models Loaded:** {models_ok}/4" if connected else "**Backend:** Offline (Demo)",
        f"🟢 Online" if connected else "🔴 Offline (Demo Mode)",
    )

def on_transaction_select(full_df: pd.DataFrame, evt: gr.SelectData):
    """User clicks a row — fetch real analysis from API."""
    if full_df.empty or evt is None:
        return "No transaction selected.", "Select a transaction.", "", ""

    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx is None or idx >= len(full_df):
        return "Invalid selection.", "Invalid selection.", "", ""

    row = full_df.iloc[idx]
    raw = row.get("_raw", row.to_dict())

    details = f"""**Transaction Investigation**

| Field | Value |
|---|---|
| **Transaction ID** | {raw.get('transaction_id', row.get('Transaction ID', 'N/A'))} |
| **Amount** | ${float(raw.get('transaction_amount', raw.get('Amount', 0))):.2f} |
| **Merchant** | {raw.get('merchant_category', row.get('Merchant', 'N/A'))} |
| **Risk Score** | {row.get('Risk Score', 0.5):.3f} |
| **Risk Level** | {row.get('Risk', 'MEDIUM')} |
| **Verdict** | {row.get('Verdict', 'REVIEW')} |
"""

    # Try real API analysis
    api_result = api_analyze(raw)

    if api_result:
        llm = api_result.get("llm_reasoning", {})
        steps = llm.get("reasoning_steps", [])
        red_flags = llm.get("red_flags", [])
        confidence = llm.get("confidence", 0.0)
        recommendation = llm.get("recommendation", "REVIEW")

        analysis = f"""**AI Analysis Results (Live)**

| Metric | Value |
|---|---|
| **Fraud Probability** | {api_result.get('fraud_probability', 0.5):.4f} |
| **Final Verdict** | {api_result.get('final_verdict', 'REVIEW')} |
| **LLM Confidence** | {confidence:.3f} |
| **LLM Recommendation** | {recommendation} |

**Reasoning Steps:**
"""
        for i, step in enumerate(steps, 1):
            analysis += f"\n{i}. {step}"

        if red_flags:
            analysis += f"\n\n**Red Flags:**\n"
            for f in red_flags:
                analysis += f"\n- {f}"
    else:
        # Fallback to simple scoring
        risk_score = float(row.get("Risk Score", 0.5))
        analysis = f"""**AI Analysis Results (Estimated — API offline)**

| Metric | Value |
|---|---|
| **Fraud Probability** | {risk_score:.4f} |
| **Confidence** | {min(risk_score * 1.5, 1.0):.3f} |
| **Mode** | Local estimate |

*Start the FastAPI backend for full LLM-powered analysis.*
"""

    return details, analysis, json.dumps(raw, indent=2, default=str), ""

def run_custom_analysis(
    txn_id: str, amount: float, merchant: str, balance: float,
    daily_count: int, avg_7d: float, failed_7d: int, auth_method: str,
    device: str, location: str, card_age: int, risk_score: float
):
    """Manual transaction analysis via the form."""
    txn = {
        "transaction_id": txn_id or f"TXN_MANUAL_{int(time.time())}",
        "user_id": "ANALYST",
        "transaction_amount": amount,
        "transaction_type": "Online",
        "timestamp": datetime.now().isoformat(),
        "account_balance": balance,
        "device_type": device,
        "location": location,
        "merchant_category": merchant,
        "ip_address_flag": 0,
        "previous_fraudulent_activity": 0,
        "daily_transaction_count": daily_count,
        "avg_transaction_amount_7d": avg_7d,
        "failed_transaction_count_7d": failed_7d,
        "card_type": "Visa",
        "card_age": card_age,
        "transaction_distance": 0.0,
        "authentication_method": auth_method,
        "risk_score": risk_score / 100.0,
        "is_weekend": 0,
    }

    result = api_analyze(txn)

    if result:
        llm = result.get("llm_reasoning", {})
        steps = llm.get("reasoning_steps", [])
        red_flags = llm.get("red_flags", [])
        output = f"""**Analysis Complete — {result.get('final_verdict', 'REVIEW')}**

| Metric | Value |
|---|---|
| **Fraud Probability** | {result.get('fraud_probability', 0.5):.4f} |
| **Risk Level** | {result.get('risk_level', 'MEDIUM')} |
| **Verdict** | {result.get('final_verdict', 'REVIEW')} |
| **LLM Confidence** | {llm.get('confidence', 0):.3f} |

**AI Reasoning:**
"""
        for i, s in enumerate(steps, 1):
            output += f"\n{i}. {s}"
        if red_flags:
            output += "\n\n**Red Flags:**\n" + "\n".join(f"- {f}" for f in red_flags)
        return output, json.dumps(result, indent=2, default=str)
    else:
        return "**API offline.** Could not analyze. Start the FastAPI backend.", ""

# --- UI ---

CSS = """
.gradio-container { max-width: 1500px; margin: auto; font-family: 'SF Mono', 'Monaco', 'Roboto Mono', monospace; }
.soc-header { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color: white; padding: 1.5em; border-radius: 12px; text-align: center; margin-bottom: 1em; border: 1px solid #334155; }
.soc-title { font-size: 2.2em; font-weight: 700; margin: 0; letter-spacing: 1px; }
.soc-subtitle { font-size: 1em; margin: 0.3em 0 0 0; opacity: 0.8; }
.panel { border: 1px solid #334155; border-radius: 10px; padding: 1em; background: #111827; color: #e2e8f0; height: 100%; }
.panel-header { color: #60a5fa; font-size: 1em; font-weight: 600; margin-bottom: 0.5em; border-bottom: 1px solid #334155; padding-bottom: 0.3em; display: flex; justify-content: space-between; }
.risk-critical { color: #ef4444; font-weight: 700; }
.risk-high { color: #f97316; font-weight: 600; }
.risk-medium { color: #eab308; }
.risk-low { color: #22c55e; }
.alert-banner { background: #450a0a; color: #fca5a5; padding: 1em; border-radius: 8px; border: 1px solid #dc2626; text-align: center; font-weight: 700; }
.status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
footer { display: none !important; }
"""

def create_ui():
    with gr.Blocks(title="FraudShield AI — SOC Cockpit") as ui:

        # State
        state_df = gr.State(pd.DataFrame())

        # Header
        gr.HTML("""
        <div class="soc-header">
            <div class="soc-title">FraudShield AI — SOC Cockpit</div>
            <div class="soc-subtitle">Local LLM-Powered Fraud Detection | All Data Stays On-Device</div>
        </div>
        """)

        # Alert banner
        alert = gr.Markdown(visible=False, value="## CRITICAL THREAT DETECTED")

        with gr.Tabs():
            # --- Tab 1: Live Monitor ---
            with gr.TabItem("Live Monitor"):
                with gr.Row():
                    with gr.Column(scale=7):
                        gr.HTML('<div class="panel"><div class="panel-header"><span>Transaction Stream</span><span style="color:#22c55e;font-size:0.8em;" id="api-status-label">● Live</span></div>')

                        with gr.Row():
                            search_box = gr.Textbox(placeholder="Search by Transaction ID or Merchant...", label="Search", scale=3, container=False)
                            risk_dropdown = gr.Dropdown(
                                choices=["All", "Critical", "High", "Medium", "Low"],
                                value="All", label="Risk Filter", scale=1, container=False
                            )

                        txn_table = gr.Dataframe(
                            headers=["Timestamp", "Risk", "Transaction ID", "Amount", "Merchant", "Risk Score", "Verdict", "Status"],
                            label=False,
                            interactive=False,
                            wrap=False,
                            column_widths=["80px", "70px", "140px", "90px", "120px", "90px", "90px", "80px"],
                        )

                    with gr.Column(scale=3):
                        gr.HTML('<div class="panel"><div class="panel-header"><span>Investigation</span><span style="color:#60a5fa;">AI-Powered</span></div>')

                        txn_details = gr.Markdown("Select a transaction from the stream.")
                        analysis_output = gr.Markdown("Analysis results appear here.")

                        with gr.Row():
                            approve_btn = gr.Button("Approve", variant="primary", size="sm")
                            review_btn = gr.Button("Review", variant="secondary", size="sm")
                            block_btn = gr.Button("Block", variant="stop", size="sm")

                        action_log = gr.Textbox(label="Action Log", lines=2, interactive=False)

                with gr.Row():
                    with gr.Column(scale=5):
                        gr.HTML('<div class="panel"><div class="panel-header">System Status</div>')
                        with gr.Row():
                            with gr.Column():
                                total_txns_md = gr.Markdown("**Total:** —")
                                fraud_rate_md = gr.Markdown("**Fraud Rate:** —")
                                avg_time_md = gr.Markdown("**Avg Time:** —")
                            with gr.Column():
                                uptime_md = gr.Markdown("**Uptime:** —")
                                models_md = gr.Markdown("**Models:** —")
                                api_status_md = gr.Markdown("**API:** —")

            # --- Tab 2: Manual Analyzer ---
            with gr.TabItem("Manual Analyzer"):
                gr.Markdown("Enter transaction details and run a full AI-powered analysis.")
                with gr.Row():
                    with gr.Column():
                        txn_id_input = gr.Textbox(label="Transaction ID", placeholder="TXN_MANUAL_001")
                        amount_input = gr.Number(label="Amount ($)", value=500.0, minimum=0.01)
                        merchant_input = gr.Textbox(label="Merchant Category", value="Electronics")
                        balance_input = gr.Number(label="Account Balance ($)", value=2000.0, minimum=0)
                        daily_count_input = gr.Number(label="Daily Transaction Count", value=1, minimum=0, step=1)
                    with gr.Column():
                        avg_7d_input = gr.Number(label="Avg Amount (7d)", value=150.0, minimum=0)
                        failed_7d_input = gr.Number(label="Failed Txns (7d)", value=0, minimum=0, step=1)
                        auth_input = gr.Dropdown(label="Auth Method", choices=["Biometric", "OTP", "Password", "Chip", "None"], value="OTP")
                        device_input = gr.Dropdown(label="Device", choices=["Mobile", "Desktop", "Tablet", "Unknown", "POS Terminal"], value="Mobile")
                        location_input = gr.Textbox(label="Location", value="New York")
                        card_age_input = gr.Number(label="Card Age (days)", value=365, minimum=0, step=1)
                    with gr.Column():
                        risk_pct = gr.Slider(label="Initial Risk Score (%)", minimum=0, maximum=100, value=30, step=1)
                        run_btn = gr.Button("Run Analysis", variant="primary", size="lg")
                with gr.Row():
                    manual_result = gr.Markdown("Ready. Click **Run Analysis** to start.")
                    manual_raw = gr.Textbox(label="Raw API Response", lines=10, interactive=False)

        # --- Event Wiring ---

        # Timer-driven refresh
        refresh_timer = gr.Timer(REFRESH_INTERVAL, active=True)
        refresh_timer.tick(
            fn=refresh_dashboard,
            inputs=[state_df, risk_dropdown, search_box],
            outputs=[state_df, txn_table, alert, total_txns_md, fraud_rate_md, avg_time_md, uptime_md, models_md, api_status_md],
        )

        # Initial load
        ui.load(
            fn=refresh_dashboard,
            inputs=[state_df, risk_dropdown, search_box],
            outputs=[state_df, txn_table, alert, total_txns_md, fraud_rate_md, avg_time_md, uptime_md, models_md, api_status_md],
        )

        # Filter / search changes
        risk_dropdown.change(
            fn=refresh_dashboard,
            inputs=[state_df, risk_dropdown, search_box],
            outputs=[state_df, txn_table, alert, total_txns_md, fraud_rate_md, avg_time_md, uptime_md, models_md, api_status_md],
        )
        search_box.submit(
            fn=refresh_dashboard,
            inputs=[state_df, risk_dropdown, search_box],
            outputs=[state_df, txn_table, alert, total_txns_md, fraud_rate_md, avg_time_md, uptime_md, models_md, api_status_md],
        )

        # Transaction selection
        txn_table.select(
            fn=on_transaction_select,
            inputs=[state_df],
            outputs=[txn_details, analysis_output, action_log, manual_raw],
        )

        # Action buttons
        def log_action(btn: str, action_log_val: str):
            ts = datetime.now().strftime("%H:%M:%S")
            return f"{action_log_val}\n[{ts}] Analyst decision: {btn}" if action_log_val else f"[{ts}] Analyst decision: {btn}"

        approve_btn.click(fn=lambda x: log_action("APPROVE", x), inputs=[action_log], outputs=[action_log])
        review_btn.click(fn=lambda x: log_action("REVIEW", x), inputs=[action_log], outputs=[action_log])
        block_btn.click(fn=lambda x: log_action("BLOCK", x), inputs=[action_log], outputs=[action_log])

        # Manual analyzer
        run_btn.click(
            fn=run_custom_analysis,
            inputs=[
                txn_id_input, amount_input, merchant_input, balance_input,
                daily_count_input, avg_7d_input, failed_7d_input,
                auth_input, device_input, location_input, card_age_input, risk_pct,
            ],
            outputs=[manual_result, manual_raw],
        )

    return ui


if __name__ == "__main__":
    print("Starting FraudShield AI SOC Cockpit...")
    connected = api_health()
    print(f"{'API connected' if connected else 'API offline — running in demo mode'}")

    ui = create_ui()
    port = int(os.getenv("DASHBOARD_PORT", "7860"))

    ui.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
