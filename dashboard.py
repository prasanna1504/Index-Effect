"""
Index Effect in Indian Equity Markets — Interactive Dashboard
Run: streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import streamlit as st
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"
DATA    = ROOT / "data"

st.set_page_config(
    page_title="Index Effect — India",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    cars    = pd.read_csv(RESULTS / "event_study_cars.csv",   parse_dates=["announcement_date","effective_date"])
    var_    = pd.read_csv(RESULTS / "var_spikes.csv",         parse_dates=["announcement_date"])
    regime  = pd.read_csv(RESULTS / "regime_analysis.csv",   parse_dates=["announcement_date","effective_date"])
    aum_ev  = pd.read_csv(RESULTS / "aum_regression.csv",    parse_dates=["announcement_date","effective_date"])
    aum_ts  = pd.read_csv(DATA    / "passive_aum.csv",       parse_dates=["date"])
    return cars, var_, regime, aum_ev, aum_ts

cars, var_, regime, aum_ev, aum_ts = load_data()

ERA_COLORS  = {"Era1_2015-2017": "#2196F3", "Era2_2018-2025": "#FF5722"}
ERA_NICE    = {"Era1_2015-2017": "Era 1 (2015–17)", "Era2_2018-2025": "Era 2 (2018–25)"}
ACT_COLORS  = {"Inclusion": "#1f77b4", "Exclusion": "#d62728"}
ALL_SYMBOLS = sorted(cars["symbol"].unique())

# ── Sidebar nav ───────────────────────────────────────────────────────────────
PAGES = [
    "🏠  Overview",
    "📈  Returns Explorer",
    "⚠️  Risk Explorer",
    "🔍  Structural Break Lab",
    "💰  AUM Regressor",
    "📋  Event Data",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Nifty 50 Reconstitutions · 2015–2025 · N=51 events")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("The Index Effect in Indian Equity Markets")
    st.markdown("**How passive AUM growth transformed Nifty 50 reconstitution dynamics (2015–2025)**")

    # KPI cards
    inc1 = cars[(cars.action=="Inclusion") & (cars.era=="Era1_2015-2017")]["car_ann_day"].mean()*100
    inc2 = cars[(cars.action=="Inclusion") & (cars.era=="Era2_2018-2025")]["car_ann_day"].mean()*100
    exc_all = cars[cars.action=="Exclusion"]["car_ann_day"].mean()*100
    n_events = len(cars)
    aum_growth = aum_ts["passive_aum_cr"].iloc[-1] / aum_ts["passive_aum_cr"].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events", n_events)
    c2.metric("Era 1 Inclusion CAR", f"{inc1:+.2f}%", "announcement day")
    c3.metric("Era 2 Inclusion CAR", f"{inc2:+.2f}%", f"{inc2-inc1:+.2f}% vs Era 1")
    c4.metric("Exclusion CAR", f"{exc_all:+.2f}%", "persistent across eras")
    c5.metric("Passive AUM Growth", f"{aum_growth:.0f}×", "2015 → 2025")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Event Timeline")
        by_date = (cars.groupby(["announcement_date","action"])["car_ann_day"]
                   .mean().reset_index())
        fig = px.scatter(
            by_date,
            x="announcement_date", y=by_date["car_ann_day"]*100,
            color="action", color_discrete_map=ACT_COLORS,
            hover_data={"announcement_date": "|%b %Y"},
            labels={"y":"Announcement-day CAR (%)", "announcement_date":"Date",
                    "car_ann_day":"CAR (%)"},
            title="Mean CAR per announcement date",
            height=350,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
        fig.add_vrect(x0="2018-01-01", x1="2025-12-31",
                      fillcolor="orange", opacity=0.05, line_width=0,
                      annotation_text="Era 2", annotation_position="top left")
        fig.update_layout(margin=dict(t=40,b=20), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Key Findings")
        st.markdown("""
**Returns (Act 1)**
- Era 1 inclusions: **+2.1%*** announcement-day CAR
- Era 2 inclusions: **–0.27%** (no longer significant)
- Structural break: Sup-F = 8.67 (critical = 8.85) — borderline
- Max F-stat at **Aug 2024**, not 2018

**Risk (Act 2)**
- VaR spikes **+40–60% above baseline** around events
- Era 1 exclusion spikes significantly larger (1.58× vs 1.25×)
- Crisis events add **+0.6×** to VaR spike for inclusions

**AUM Mechanism**
- log(AUM) coefficient: **–0.0076*** for inclusions (p=0.026)
- 10× AUM increase → **–176 bp** change in CAR[0,+1]
- Passive AUM grew **26× from ₹41K Cr → ₹11L Cr**
""")

    st.markdown("---")
    st.subheader("Passive AUM Growth")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=aum_ts["date"], y=aum_ts["passive_aum_cr"]/100000,
        fill="tozeroy", fillcolor="rgba(26,35,126,0.15)",
        line=dict(color="#1a237e", width=2),
        name="Passive AUM",
        hovertemplate="₹%{y:.1f}L Cr<extra></extra>",
    ))
    inc_dates = cars[cars.action=="Inclusion"]["announcement_date"].unique()
    exc_dates = cars[cars.action=="Exclusion"]["announcement_date"].unique()
    for dt in inc_dates:
        fig2.add_vline(x=dt, line_color=ACT_COLORS["Inclusion"], line_width=0.8, opacity=0.5)
    for dt in exc_dates:
        fig2.add_vline(x=dt, line_color=ACT_COLORS["Exclusion"], line_width=0.8, opacity=0.5)
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              line=dict(color=ACT_COLORS["Inclusion"]),
                              name="Inclusion event"))
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              line=dict(color=ACT_COLORS["Exclusion"]),
                              name="Exclusion event"))
    fig2.update_layout(
        yaxis_title="₹ Lakh Crore", height=280,
        margin=dict(t=20,b=20), legend_title="",
        yaxis_tickprefix="₹", yaxis_ticksuffix="L Cr",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Returns Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("Returns Explorer")
    st.markdown("CAR windows around Nifty 50 reconstitution announcements")

    c1, c2, c3 = st.columns(3)
    sel_action = c1.selectbox("Action", ["Both", "Inclusion", "Exclusion"])
    sel_era    = c2.multiselect("Era", list(ERA_NICE.values()),
                                default=list(ERA_NICE.values()))
    sel_window = c3.selectbox("CAR window",
                              ["car_ann_day","car_pre_drift","car_post_short",
                               "car_post_medium","car_post_long","car_eff_window"],
                              format_func=lambda x: {
                                  "car_ann_day":     "Announcement day [0,+1]",
                                  "car_pre_drift":   "Pre-event drift [–10,–1]",
                                  "car_post_short":  "Post-event short [+2,+5]",
                                  "car_post_medium": "Post-event medium [+2,+20]",
                                  "car_post_long":   "Post-event long [+2,+40]",
                                  "car_eff_window":  "Effective-date window",
                              }[x])

    era_map_rev = {v: k for k, v in ERA_NICE.items()}
    sel_era_raw = [era_map_rev[e] for e in sel_era]
    df = cars[cars["era"].isin(sel_era_raw)].copy()
    if sel_action != "Both":
        df = df[df["action"] == sel_action]

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["CAR Distribution", "Time Series", "Era Comparison"])

    with tab1:
        fig = px.histogram(
            df, x=df[sel_window]*100,
            color="action" if sel_action=="Both" else "era",
            color_discrete_map=ACT_COLORS if sel_action=="Both" else ERA_COLORS,
            barmode="overlay", nbins=30, opacity=0.7,
            labels={sel_window: "CAR (%)"},
            title=f"Distribution of {sel_window}",
        )
        fig.add_vline(x=0, line_dash="dot", line_color="black")
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats table
        grp_col = "action" if sel_action=="Both" else "era"
        summary = (df.groupby(grp_col)[sel_window]
                     .agg(N="count", Mean=lambda x: x.mean()*100,
                          Median=lambda x: x.median()*100,
                          Std=lambda x: x.std()*100)
                     .round(3).reset_index())
        summary.columns = [grp_col.title(), "N", "Mean (%)", "Median (%)", "Std (%)"]
        st.dataframe(summary, hide_index=True, use_container_width=True)

    with tab2:
        by_date = (df.groupby(["announcement_date","action"])[sel_window]
                     .mean().reset_index())
        fig = px.scatter(
            by_date,
            x="announcement_date", y=by_date[sel_window]*100,
            color="action", color_discrete_map=ACT_COLORS,
            trendline="lowess", trendline_options={"frac": 0.4},
            labels={sel_window: "CAR (%)", "announcement_date": ""},
            title=f"{sel_window} over time (LOWESS trend)",
            height=400,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        era_means = (cars[cars["action"].isin(
                        ["Inclusion","Exclusion"] if sel_action=="Both" else [sel_action]
                    )].groupby(["era","action"])[sel_window]
                     .agg(["mean","sem","count"]).reset_index())
        era_means["mean_pct"] = era_means["mean"]*100
        era_means["err_pct"]  = era_means["sem"]*100*1.96
        era_means["era_nice"] = era_means["era"].map(ERA_NICE)

        fig = px.bar(
            era_means, x="era_nice", y="mean_pct",
            color="action" if sel_action=="Both" else "era",
            color_discrete_map=ACT_COLORS if sel_action=="Both" else ERA_COLORS,
            error_y="err_pct", barmode="group",
            labels={"mean_pct": "Mean CAR (%)", "era_nice": ""},
            title=f"Mean {sel_window} by era (±95% CI)",
            height=380,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(legend_title="")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Risk Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("Risk Explorer")
    st.markdown("VaR spikes and correlation structure around reconstitution events")

    c1, c2 = st.columns(2)
    sel_action = c1.selectbox("Action", ["Both", "Inclusion", "Exclusion"])
    sel_metric = c2.selectbox("Metric", ["peak_var_ratio", "ann_day_var_ratio", "corr_jump"],
                              format_func=lambda x: {
                                  "peak_var_ratio":    "Peak VaR ratio (×baseline)",
                                  "ann_day_var_ratio": "Announcement-day VaR ratio",
                                  "corr_jump":        "Correlation jump (post − pre)",
                              }[x])

    merged = var_.merge(
        regime[["announcement_date","symbol","corr_jump","is_crisis","regime_label"]],
        on=["announcement_date","symbol"], how="left"
    )
    if sel_action != "Both":
        merged = merged[merged.action == sel_action]

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["By Era", "Crisis vs Calm", "Scatter vs CAR"])

    with tab1:
        era_grp = (merged.dropna(subset=[sel_metric])
                         .groupby(["era","action"])[sel_metric]
                         .agg(["mean","sem"]).reset_index())
        era_grp["era_nice"] = era_grp["era"].map(ERA_NICE)
        era_grp["ci95"]     = era_grp["sem"]*1.96

        fig = px.bar(
            era_grp, x="era_nice", y="mean",
            color="action" if sel_action=="Both" else "era",
            color_discrete_map=ACT_COLORS if sel_action=="Both" else ERA_COLORS,
            error_y="ci95", barmode="group",
            labels={"mean": sel_metric, "era_nice": ""},
            title=f"Mean {sel_metric} by era (±95% CI)",
        )
        if "var" in sel_metric:
            fig.add_hline(y=1, line_dash="dot", line_color="grey",
                          annotation_text="Baseline (1×)")
        fig.update_layout(legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        crisis_grp = (merged.dropna(subset=[sel_metric,"is_crisis"])
                            .groupby(["is_crisis","action"])[sel_metric]
                            .agg(["mean","sem","count"]).reset_index())
        crisis_grp["regime"] = crisis_grp["is_crisis"].map({True:"Crisis",False:"Calm"})
        crisis_grp["ci95"]   = crisis_grp["sem"]*1.96

        fig = px.bar(
            crisis_grp, x="regime", y="mean",
            color="action" if sel_action=="Both" else "regime",
            color_discrete_map=ACT_COLORS if sel_action=="Both"
                               else {"Calm":"#4CAF50","Crisis":"#F44336"},
            error_y="ci95", barmode="group",
            labels={"mean": sel_metric, "regime":""},
            title=f"Mean {sel_metric}: Crisis vs Calm",
        )
        if "var" in sel_metric:
            fig.add_hline(y=1, line_dash="dot", line_color="grey")
        fig.update_layout(legend_title="")
        st.plotly_chart(fig, use_container_width=True)

        n_crisis = int(merged["is_crisis"].sum())
        st.caption(f"Crisis events (COVID 2020, Rates 2022): N={n_crisis} | "
                   f"Calm: N={len(merged)-n_crisis}")

    with tab3:
        scatter_df = merged.merge(
            cars[["announcement_date","symbol","car_ann_day"]],
            on=["announcement_date","symbol"], how="inner"
        ).dropna(subset=[sel_metric,"car_ann_day"])

        fig = px.scatter(
            scatter_df,
            x=scatter_df["car_ann_day"]*100,
            y=sel_metric,
            color="era" if sel_action!="Both" else "action",
            color_discrete_map=ERA_COLORS if sel_action!="Both" else ACT_COLORS,
            trendline="ols",
            labels={"x":"Announcement-day CAR (%)", sel_metric: sel_metric},
            hover_data=["symbol","announcement_date"],
            title=f"{sel_metric} vs CAR (OLS trendline)",
        )
        if "var" in sel_metric:
            fig.add_hline(y=1, line_dash="dot", line_color="grey")
        fig.add_vline(x=0, line_dash="dot", line_color="grey")
        fig.update_layout(legend_title="")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Structural Break Lab
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("Structural Break Lab")
    st.markdown("Drag the break date and watch the CAR comparison update live.")

    sel_action = st.selectbox("Action", ["Inclusion", "Exclusion"])  # break lab needs a single action
    sub = cars[cars.action == sel_action].copy()
    sub = sub.sort_values("announcement_date")

    ann_dates = sorted(sub["announcement_date"].dt.date.unique())
    min_d, max_d = ann_dates[2], ann_dates[-3]   # keep ≥3 obs each side

    break_date = st.slider(
        "Break date",
        min_value=min_d, max_value=max_d,
        value=pd.Timestamp("2018-01-01").date(),
        format="YYYY-MM-DD",
        help="Slide to change the assumed structural break date"
    )

    sub["group"] = sub["announcement_date"].dt.date.apply(
        lambda d: f"Before {break_date}" if d <= break_date else f"After {break_date}"
    )

    col_l, col_r = st.columns([2, 1])

    with col_l:
        grp = sub.groupby("group")["car_ann_day"].agg(
            N="count", Mean=lambda x: x.mean()*100,
            Std=lambda x: x.std()*100, SEM=lambda x: (x.std()/np.sqrt(len(x)))*100
        ).reset_index()
        grp["ci95"] = grp["SEM"]*1.96

        fig = px.bar(grp, x="group", y="Mean", error_y="ci95",
                     color="group",
                     color_discrete_sequence=["#2196F3","#FF5722"],
                     labels={"Mean":"Mean CAR[0,+1] (%)", "group":""},
                     title=f"{sel_action}s: before vs after {break_date}",
                     height=380)
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Scatter with break line
        fig2 = px.scatter(
            sub, x="announcement_date", y=sub["car_ann_day"]*100,
            color="group", color_discrete_sequence=["#2196F3","#FF5722"],
            labels={"y":"CAR[0,+1] (%)", "announcement_date":""},
            trendline="ols",
            height=320,
        )
        fig2.add_vline(x=pd.Timestamp(break_date), line_dash="dash",
                       line_color="black", line_width=2,
                       annotation_text="Break", annotation_position="top")
        fig2.add_hline(y=0, line_dash="dot", line_color="grey")
        fig2.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.subheader("Split stats")
        for _, row in grp.iterrows():
            st.metric(row["group"], f"{row['Mean']:+.2f}%",
                      f"N={int(row['N'])}  σ={row['Std']:.2f}%")

        st.markdown("---")
        st.subheader("Wald F-statistic")
        # Compute simple Chow F-stat
        before = sub[sub["announcement_date"].dt.date <= break_date]["car_ann_day"].dropna()
        after  = sub[sub["announcement_date"].dt.date >  break_date]["car_ann_day"].dropna()

        if len(before) >= 3 and len(after) >= 3:
            n1, n2  = len(before), len(after)
            n_total = n1 + n2
            # Restricted model: pooled mean
            rss_r = ((sub["car_ann_day"] - sub["car_ann_day"].mean())**2).sum()
            # Unrestricted: separate means
            rss_u = (((before - before.mean())**2).sum() +
                     ((after  - after.mean() )**2).sum())
            f_stat = ((rss_r - rss_u) / 1) / (rss_u / (n_total - 2))
            p_approx = 1 - __import__("scipy").stats.f.cdf(f_stat, 1, n_total-2)

            critical = 8.85  # Andrews (1993) 5% for Sup-Wald
            color = "🔴" if f_stat >= critical else "🟡" if f_stat >= 6 else "🟢"
            st.metric("F-statistic", f"{f_stat:.3f}",
                      delta=f"{'REJECT' if f_stat>=critical else 'fail to reject'} at 5%")
            st.caption(f"{color} Andrews (1993) critical value = 8.85\np ≈ {p_approx:.3f}")
        else:
            st.warning("Not enough observations in one segment (need ≥3 each side).")

        st.markdown("---")
        st.caption("""
**Sup-Wald result** from `07_structural_break.py`:
- Inclusions: Sup-F = **8.67** (critical 8.85) — borderline
- Max F at **Aug 2024**, not 2018
- 2018 is economically motivated, not statistically confirmed
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AUM Regressor
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("AUM Regressor")
    st.markdown("Testing the passive mechanism directly: does announcement-day CAR compress as passive AUM grows?")

    sel_action = st.selectbox("Action", ["Both (pooled)", "Inclusion", "Exclusion"])

    if sel_action == "Both (pooled)":
        sub = aum_ev.dropna(subset=["car_ann_day","log_aum"])
    else:
        sub = aum_ev[(aum_ev.action == sel_action)].dropna(subset=["car_ann_day","log_aum"])

    col_l, col_r = st.columns([3, 1])

    with col_l:
        fig = px.scatter(
            sub,
            x="aum_lakh_cr",
            y=sub["car_ann_day"]*100,
            color="era",
            color_discrete_map=ERA_COLORS,
            symbol="action",
            hover_data=["symbol","announcement_date"],
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="black",
            labels={"aum_lakh_cr": "Passive AUM (₹ Lakh Crore)",
                    "y": "CAR[0,+1] (%)"},
            title=f"Announcement-day CAR vs Passive AUM — {sel_action}",
            height=450,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_traces(selector=dict(mode="lines"), line_width=2)
        fig.update_layout(legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("OLS Results")
        if sel_action == "Both (pooled)":
            mod = smf.ols("car_ann_day ~ log_aum + inclusion + log_aum:inclusion",
                          data=sub).fit()
            for term in mod.params.index:
                p = mod.pvalues[term]
                stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
                st.metric(term,
                          f"{mod.params[term]:+.5f}{stars}",
                          f"t={mod.tvalues[term]:+.2f}  p={p:.3f}")
        else:
            if len(sub) >= 4:
                mod = smf.ols("car_ann_day ~ log_aum", data=sub).fit()
                beta = mod.params["log_aum"]
                p    = mod.pvalues["log_aum"]
                stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""

                bp_per_10x = beta * np.log(10) * 100 * 100
                st.metric("Intercept",
                          f"{mod.params['Intercept']:+.4f}",
                          f"t={mod.tvalues['Intercept']:+.2f}")
                st.metric(f"log(AUM){stars}",
                          f"{beta:+.6f}",
                          f"t={mod.tvalues['log_aum']:+.2f}  p={p:.3f}")
                st.metric("R²", f"{mod.rsquared:.3f}")
                st.metric("N", len(sub))
                st.markdown("---")
                st.metric("Economic magnitude",
                          f"{bp_per_10x:+.0f} bp",
                          "per 10× AUM increase")
            else:
                st.warning("Not enough data for regression.")

    st.markdown("---")
    st.subheader("VaR Spike vs AUM")
    sub_var = aum_ev.copy()
    if sel_action != "Both (pooled)":
        sub_var = sub_var[sub_var.action == sel_action]

    sub_var = sub_var.dropna(subset=["peak_var_ratio","log_aum"])
    fig2 = px.scatter(
        sub_var,
        x="aum_lakh_cr", y="peak_var_ratio",
        color="era", color_discrete_map=ERA_COLORS,
        symbol="action",
        trendline="ols", trendline_scope="overall",
        trendline_color_override="black",
        hover_data=["symbol","announcement_date"],
        labels={"aum_lakh_cr":"Passive AUM (₹ Lakh Crore)",
                "peak_var_ratio":"Peak VaR ratio (×baseline)"},
        title="VaR Spike vs Passive AUM",
        height=380,
    )
    fig2.add_hline(y=1, line_dash="dot", line_color="grey",
                   annotation_text="Baseline")
    fig2.update_layout(legend_title="")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Event Data
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("Event Data")
    st.markdown(f"All **{len(regime)}** events with full metrics")

    c1, c2, c3, c4 = st.columns(4)
    filt_action = c1.multiselect("Action", ["Inclusion","Exclusion"],
                                 default=["Inclusion","Exclusion"])
    filt_era    = c2.multiselect("Era", list(ERA_NICE.values()),
                                 default=list(ERA_NICE.values()))
    filt_regime = c3.multiselect("Regime", ["Calm","Crisis"],
                                 default=["Calm","Crisis"])
    filt_sym    = c4.multiselect("Symbol", ALL_SYMBOLS, default=[])

    era_map_rev = {v: k for k, v in ERA_NICE.items()}
    df = regime.copy()
    df = df[df["action"].isin(filt_action)]
    df = df[df["era"].isin([era_map_rev[e] for e in filt_era])]
    df = df[df["regime_label"].isin(filt_regime)]
    if filt_sym:
        df = df[df["symbol"].isin(filt_sym)]

    disp_cols = ["announcement_date","effective_date","symbol","action","era",
                 "car_ann_day","car_pre_drift","car_post_short","car_post_medium",
                 "peak_var_ratio","corr_jump","regime_label","avol_ratio"]
    disp_cols = [c for c in disp_cols if c in df.columns]
    show = df[disp_cols].copy()
    show["era"] = show["era"].map(ERA_NICE)

    # Format percentages
    for col in ["car_ann_day","car_pre_drift","car_post_short","car_post_medium"]:
        if col in show.columns:
            show[col] = (show[col]*100).round(2)

    for col in ["peak_var_ratio","corr_jump","avol_ratio"]:
        if col in show.columns:
            show[col] = show[col].round(3)

    col_rename = {
        "announcement_date": "Ann. Date", "effective_date": "Eff. Date",
        "symbol": "Symbol", "action": "Action", "era": "Era",
        "car_ann_day": "CAR[0,+1] %", "car_pre_drift": "Pre-drift %",
        "car_post_short": "Post [+2,+5] %", "car_post_medium": "Post [+2,+20] %",
        "peak_var_ratio": "Peak VaR ×", "corr_jump": "Corr Jump",
        "regime_label": "Regime", "avol_ratio": "Abnorm Vol ×",
    }
    show = show.rename(columns=col_rename)
    st.dataframe(show, use_container_width=True, hide_index=True, height=500)

    st.caption(f"Showing {len(show)} of {len(regime)} events")

    # Download
    csv = show.to_csv(index=False)
    st.download_button("Download CSV", csv, "index_effect_events.csv", "text/csv")
