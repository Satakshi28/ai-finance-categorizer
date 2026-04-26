import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json
import os
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp { background: #0a0a0f; color: #e8e8f0; }

section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e;
}

.metric-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a40;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .label {
    font-size: 12px;
    color: #6b6b8a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
    margin-bottom: 8px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #e8e8f0;
}
.metric-card .value.green { color: #4ade80; }
.metric-card .value.red { color: #f87171; }
.metric-card .value.blue { color: #60a5fa; }

.alert-box {
    background: linear-gradient(135deg, #2d1515 0%, #1f0f0f 100%);
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    color: #fca5a5;
}
.alert-box .alert-title {
    font-weight: 600;
    color: #f87171;
    margin-bottom: 4px;
    font-family: 'Syne', sans-serif;
}

.category-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}

div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

.stButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e1e2e;
}
</style>
""", unsafe_allow_html=True)

# ── Category config ───────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Food & Dining": "#f59e0b",
    "Shopping": "#ec4899",
    "Transport": "#3b82f6",
    "Entertainment": "#8b5cf6",
    "Utilities & Bills": "#06b6d4",
    "Health & Medical": "#10b981",
    "Housing & Rent": "#f97316",
    "Education": "#6366f1",
    "Income": "#4ade80",
    "Travel": "#e879f9",
    "Groceries": "#84cc16",
    "Other": "#94a3b8",
}

# ── Groq categorization ──────────────────────────────────────────────────────
def categorize_transactions(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    client = Groq(api_key=api_key)

    transactions_text = ""
    for i, row in df.iterrows():
        transactions_text += f'{i}: "{row["Description"]}" ₹{row["Amount"]} ({row["Type"]})\n'

    prompt = f"""You are a financial transaction categorizer. Categorize each transaction below into one of these categories:
Food & Dining, Shopping, Transport, Entertainment, Utilities & Bills, Health & Medical, Housing & Rent, Education, Income, Travel, Groceries, Other

Rules:
- Income/Credit transactions → "Income"
- Be specific: Swiggy/Zomato/restaurants → "Food & Dining", Amazon/Myntra/Flipkart → "Shopping"
- Gym → "Health & Medical", Flight/hotel → "Travel", Petrol → "Transport"

Transactions:
{transactions_text}

Respond ONLY with a valid JSON object mapping index to category. Example:
{{"0": "Food & Dining", "1": "Income"}}
No markdown, no explanation, just the JSON."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    categories_map = json.loads(raw)

    df = df.copy()
    df["Category"] = df.index.astype(str).map(categories_map).fillna("Other")
    return df


def detect_anomalies(df: pd.DataFrame) -> list:
    anomalies = []
    debits = df[df["Type"] == "Debit"].copy()
    if debits.empty:
        return anomalies

    cat_stats = debits.groupby("Category")["Amount"].agg(["mean", "std"]).reset_index()

    for _, row in debits.iterrows():
        cat = row["Category"]
        stats = cat_stats[cat_stats["Category"] == cat]
        if stats.empty:
            continue
        mean = stats["mean"].values[0]
        std = stats["std"].values[0] if not pd.isna(stats["std"].values[0]) else 0
        threshold = mean + 2 * std if std > 0 else mean * 2
        if row["Amount"] > threshold and row["Amount"] > mean * 1.8:
            anomalies.append({
                "description": row["Description"],
                "amount": row["Amount"],
                "category": cat,
                "avg": round(mean, 2),
                "excess": round(row["Amount"] - mean, 2)
            })
    return anomalies


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 FinSight AI")
    st.markdown("<div style='color:#6b6b8a; font-size:13px; margin-bottom:24px;'>AI-Powered Transaction Intelligence</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("[Get free API key →](https://console.groq.com/keys)", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:24px;">Upload Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CSV File", type=["csv"],
        help="Columns: Date, Description, Amount, Type (Debit/Credit)")

    st.markdown("---")
    st.markdown("<div style='color:#6b6b8a; font-size:12px;'>Need a test file? Download sample below ↓</div>", unsafe_allow_html=True)

    sample_csv = """Date,Description,Amount,Type
2026-04-01,Swiggy Order #4821,450.00,Debit
2026-04-01,Salary Credit,85000.00,Credit
2026-04-02,Amazon Purchase,1299.00,Debit
2026-04-02,Uber Ride to Airport,680.00,Debit
2026-04-03,Big Bazaar Groceries,2340.00,Debit
2026-04-04,Netflix Subscription,649.00,Debit
2026-04-05,Zomato Order,320.00,Debit
2026-04-05,ATM Withdrawal,5000.00,Debit
2026-04-06,Electricity Bill BESCOM,1850.00,Debit
2026-04-07,Dominos Pizza,780.00,Debit
2026-04-08,Ola Cab,230.00,Debit
2026-04-08,Freelance Payment Received,15000.00,Credit
2026-04-09,Flipkart Electronics,8999.00,Debit
2026-04-10,Gym Membership,1500.00,Debit
2026-04-11,PVR Cinema Tickets,700.00,Debit
2026-04-12,Airtel Recharge,299.00,Debit
2026-04-13,Medical Pharmacy,560.00,Debit
2026-04-14,Starbucks Coffee,450.00,Debit
2026-04-15,Book Purchase Flipkart,399.00,Debit
2026-04-16,House Rent Transfer,18000.00,Debit
2026-04-17,Swiggy Instamart,890.00,Debit
2026-04-18,IndiGo Flight Ticket,4500.00,Debit
2026-04-19,Spotify Premium,119.00,Debit
2026-04-20,Petrol Pump,2000.00,Debit
2026-04-21,Restaurant Bill,1200.00,Debit
2026-04-22,UPI Transfer Received,3000.00,Credit
2026-04-23,Coursera Subscription,1500.00,Debit
2026-04-24,Doctor Consultation,800.00,Debit
2026-04-25,Myntra Clothes,2199.00,Debit
2026-04-26,Interest Credit,120.00,Credit"""
    st.download_button("⬇ Download Sample CSV", sample_csv, "sample_transactions.csv", "text/csv")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# FinSight AI")
st.markdown("<div style='color:#6b6b8a; margin-bottom:32px;'>Upload your bank statement · AI categorizes every transaction · Get instant spending insights</div>", unsafe_allow_html=True)

if not uploaded_file:
    # Landing state
    col1, col2, col3 = st.columns(3)
    features = [
        ("🤖", "AI Categorization", "Groq automatically tags every transaction into 12 spending categories"),
        ("📊", "Spending Dashboard", "Interactive charts — pie, bar, trend — all generated instantly"),
        ("🚨", "Anomaly Detection", "Flags unusual transactions that spike above your normal spending"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; padding:28px;">
                <div style="font-size:28px; margin-bottom:12px;">{icon}</div>
                <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700; margin-bottom:8px;">{title}</div>
                <div style="color:#6b6b8a; font-size:13px; line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload a CSV file from the sidebar to get started. Use the sample CSV if you don't have one handy.")

else:
    # Load CSV
    try:
        df = pd.read_csv(uploaded_file)
        required = {"Description", "Amount", "Type"}
        if not required.issubset(df.columns):
            st.error(f"CSV must have columns: {required}. Found: {list(df.columns)}")
            st.stop()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

    # Categorize button
    if "categorized_df" not in st.session_state:
        st.session_state.categorized_df = None

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        run = st.button("🤖 Categorize with AI")
    with col_info:
        if not api_key:
            st.warning("Enter your Groq API key in the sidebar first..")

    if run:
        if not api_key:
            st.error("API key required.")
        else:
            with st.spinner("Groq is reading your transactions..."):
                try:
                    st.session_state.categorized_df = categorize_transactions(df, api_key)
                    st.success(f"✅ Categorized {len(df)} transactions successfully!")
                except Exception as e:
                    st.error(f"API Error: {e}")

    result_df = st.session_state.categorized_df

    if result_df is not None:
        debits = result_df[result_df["Type"] == "Debit"]
        credits = result_df[result_df["Type"] == "Credit"]
        total_spent = debits["Amount"].sum()
        total_income = credits["Amount"].sum()
        savings = total_income - total_spent
        top_cat = debits.groupby("Category")["Amount"].sum().idxmax() if not debits.empty else "—"

        # Metrics row
        st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Total Spent", f"₹{total_spent:,.0f}", "red"),
            ("Total Income", f"₹{total_income:,.0f}", "green"),
            ("Net Savings", f"₹{savings:,.0f}", "green" if savings >= 0 else "red"),
            ("Top Category", top_cat, "blue"),
        ]
        for col, (label, value, color) in zip([c1, c2, c3, c4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value {color}">{value}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        st.markdown('<div class="section-header">Spending Breakdown</div>', unsafe_allow_html=True)
        cat_totals = debits.groupby("Category")["Amount"].sum().reset_index().sort_values("Amount", ascending=False)
        colors = [CATEGORY_COLORS.get(c, "#94a3b8") for c in cat_totals["Category"]]

        ch1, ch2 = st.columns(2)
        with ch1:
            fig_pie = px.pie(
                cat_totals, values="Amount", names="Category",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                hole=0.5,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8e8f0", legend=dict(font=dict(size=11)),
                margin=dict(t=20, b=20),
                showlegend=True,
            )
            fig_pie.update_traces(textfont_size=11)
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            fig_bar = px.bar(
                cat_totals, x="Amount", y="Category", orientation="h",
                color="Category", color_discrete_map=CATEGORY_COLORS,
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8e8f0", showlegend=False,
                xaxis=dict(gridcolor="#1e1e2e", title="Amount (₹)"),
                yaxis=dict(gridcolor="#1e1e2e", title=""),
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Trend chart if Date column exists
        if "Date" in result_df.columns:
            try:
                result_df["Date"] = pd.to_datetime(result_df["Date"])
                daily = debits.copy()
                daily["Date"] = pd.to_datetime(daily["Date"])
                daily_sum = daily.groupby("Date")["Amount"].sum().reset_index()
                fig_line = px.line(daily_sum, x="Date", y="Amount",
                    labels={"Amount": "Daily Spend (₹)"})
                fig_line.update_traces(line_color="#6366f1", line_width=2.5)
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e8e8f0",
                    xaxis=dict(gridcolor="#1e1e2e"),
                    yaxis=dict(gridcolor="#1e1e2e"),
                    margin=dict(t=10, b=10),
                )
                st.markdown('<div class="section-header">Daily Spending Trend</div>', unsafe_allow_html=True)
                st.plotly_chart(fig_line, use_container_width=True)
            except:
                pass

        # Anomalies
        st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)
        anomalies = detect_anomalies(result_df)
        if anomalies:
            for a in anomalies:
                st.markdown(f"""
                <div class="alert-box">
                    <div class="alert-title">⚠ Unusual Transaction Detected</div>
                    <b>{a['description']}</b> — ₹{a['amount']:,.0f} 
                    &nbsp;|&nbsp; Category: {a['category']}
                    &nbsp;|&nbsp; Your avg: ₹{a['avg']:,.0f} 
                    &nbsp;|&nbsp; <span style="color:#f87171;">+₹{a['excess']:,.0f} above average</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ No anomalies detected — your spending looks normal!")

        # Full table
        st.markdown('<div class="section-header" style="margin-top:24px;">All Transactions</div>', unsafe_allow_html=True)

        # Category filter
        all_cats = ["All"] + sorted(result_df["Category"].unique().tolist())
        selected_cat = st.selectbox("Filter by category", all_cats)
        display_df = result_df if selected_cat == "All" else result_df[result_df["Category"] == selected_cat]

        st.dataframe(
            display_df[["Date", "Description", "Amount", "Type", "Category"]].reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

        # Download
        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Categorized CSV", csv_out, "categorized_transactions.csv", "text/csv")

    else:
        # Preview uncategorized
        st.markdown('<div class="section-header">Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Loaded {len(df)} transactions. Click **Categorize with AI** above to analyze.")
