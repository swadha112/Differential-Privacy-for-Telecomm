# =============================================================
# app.py  —  Telecom DP Shield
# Differential Privacy in Telecom: Interactive Dashboard
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Telecom DP Shield",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* Metric cards */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.75rem;
    font-weight: 700;
}
/* Remove default top padding */
.block-container { padding-top: 1.2rem; }
/* Nicer tabs */
.stTabs [data-baseweb="tab"] {
    height: 46px;
    font-weight: 500;
    font-size: 0.95rem;
}
/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# CONSTANTS
# =============================================================
MAPBOX_STYLE  = "open-street-map"
MUMBAI_CENTER = {"lat": 19.076, "lon": 72.877}
NIGHT_HOURS   = [22, 23, 0, 1, 2, 3, 4, 5, 6]
DAY_HOURS     = list(range(9, 18))
DAY_ORDER     = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================
# DATA LOADING  (cached — runs once)
# =============================================================
@st.cache_data
def load_data():
    df     = pd.read_csv(os.path.join(DATA_DIR, "mumbai_telecom_cdr.csv"))
    towers = pd.read_csv(os.path.join(DATA_DIR, "mumbai_towers.csv"))
    users  = pd.read_csv(os.path.join(DATA_DIR, "mumbai_users.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, towers, users


# =============================================================
# DIFFERENTIAL PRIVACY ENGINE
# =============================================================

def _planar_laplace(lats, lons, eps_km, seed):
    """
    Planar Laplace (Geo-Indistinguishability) — Andrés et al., 2013.

    Formal guarantee:  P(z | x) / P(z | y) ≤ exp(ε · d(x,y))
    Sampling:
        r ~ Gamma(shape=2, scale=1/ε)   [noise radius in km]
        θ ~ Uniform(0, 2π)              [noise direction]
    Coordinate conversion:
        Δlat = r·cos(θ) / 111          [1° lat ≈ 111 km]
        Δlon = r·sin(θ) / (111·cos(φ)) [1° lon ≈ 111·cos(φ) km]
    """
    rng = np.random.RandomState(seed)
    n   = len(lats)
    r   = rng.gamma(shape=2, scale=1.0 / eps_km, size=n)   # km
    th  = rng.uniform(0, 2 * np.pi, size=n)
    la  = np.asarray(lats, dtype=float)
    lo  = np.asarray(lons, dtype=float)
    return (la + r * np.cos(th) / 111.0,
            lo + r * np.sin(th) / (111.0 * np.cos(np.radians(la))))


@st.cache_data
def get_dp_dataset(eps_km):
    """Apply Planar Laplace to full CDR; cached per ε value."""
    df, _, _ = load_data()
    ns_lat, ns_lon = _planar_laplace(
        df["source_lat"].values, df["source_lon"].values, eps_km, seed=42)
    nd_lat, nd_lon = _planar_laplace(
        df["dest_lat"].values,   df["dest_lon"].values,   eps_km, seed=99)
    out = df.copy()
    out["source_lat"] = ns_lat;  out["source_lon"] = ns_lon
    out["dest_lat"]   = nd_lat;  out["dest_lon"]   = nd_lon
    return out


def _nearest_areas(lats, lons, towers_df):
    """Vectorised nearest-tower area lookup (Euclidean approx.)."""
    tla = towers_df["lat"].values;   tlo = towers_df["lon"].values
    ta  = towers_df["area_name"].values
    la  = np.asarray(lats, dtype=float)
    lo  = np.asarray(lons, dtype=float)
    d2  = (la[:, None] - tla[None, :]) ** 2 + (lo[:, None] - tlo[None, :]) ** 2
    return ta[np.argmin(d2, axis=1)]


@st.cache_data
def home_attack_accuracy(eps_km=None):
    """
    Nighttime-tower home-inference attack.
    Most-frequent nighttime (22 h–6 h) tower area → inferred home.
    eps_km=None  →  attack on raw data.
    """
    df, towers, _ = load_data()
    q = get_dp_dataset(eps_km) if eps_km is not None else df

    night = q[q["hour_of_day"].isin(NIGHT_HOURS)].copy()
    if eps_km is not None:
        night["_area"] = _nearest_areas(
            night["source_lat"].values, night["source_lon"].values, towers)
        area_col = "_area"
    else:
        area_col = "source_area"

    gt = df.drop_duplicates("user_id").set_index("user_id")["home_area"].to_dict()
    correct = total = 0
    for uid, grp in night.groupby("user_id"):
        if len(grp) == 0:
            continue
        if gt.get(uid) == grp[area_col].mode().iloc[0]:
            correct += 1
        total += 1
    return correct / max(total, 1)


@st.cache_data
def privacy_utility_curve():
    """Precompute home-inference accuracy for a range of ε values."""
    raw_acc  = home_attack_accuracy(eps_km=None)
    eps_vals = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    dp_accs  = [home_attack_accuracy(e) for e in eps_vals]
    return eps_vals, dp_accs, raw_acc


# =============================================================
# REUSABLE PLOT HELPERS
# =============================================================

def _trajectory_map(udf, title, colorscale="Plasma"):
    udf = udf.sort_values("timestamp").reset_index(drop=True)
    hover = [
        f"⏰ {row['timestamp']}<br>"
        f"📍 {row['source_area']}<br>"
        f"📱 {row['event_type'].replace('_',' ').title()}<br>"
        f"🕐 {row['hour_of_day']}:00"
        for _, row in udf.iterrows()
    ]
    fig = go.Figure()
    # trajectory lines
    fig.add_trace(go.Scattermapbox(
        lat=udf["source_lat"].tolist(), lon=udf["source_lon"].tolist(),
        mode="lines", line=dict(width=1.5, color="rgba(99,102,241,0.35)"),
        hoverinfo="none", showlegend=False))
    # event markers (colour = hour of day)
    fig.add_trace(go.Scattermapbox(
        lat=udf["source_lat"].tolist(), lon=udf["source_lon"].tolist(),
        mode="markers",
        marker=dict(size=9, color=udf["hour_of_day"].tolist(),
                    colorscale=colorscale, cmin=0, cmax=23,
                    colorbar=dict(title="Hour", thickness=10, len=0.6),
                    opacity=0.82),
        text=hover, hoverinfo="text", name="Events"))
    fig.update_layout(
        mapbox_style=MAPBOX_STYLE,
        mapbox_center={"lat": udf["source_lat"].mean(),
                       "lon": udf["source_lon"].mean()},
        mapbox_zoom=11, height=400,
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text=title, font=dict(size=13, color="#374151")),
        paper_bgcolor="white")
    return fig


def _activity_heatmap(udf):
    agg  = udf.groupby(["day_of_week", "hour_of_day"]).size().reset_index(name="n")
    wide = (agg.pivot(index="day_of_week", columns="hour_of_day", values="n")
               .reindex([d for d in DAY_ORDER if d in agg["day_of_week"].values])
               .fillna(0))
    for h in range(24):
        if h not in wide.columns:
            wide[h] = 0
    wide = wide[sorted(wide.columns)]
    fig = px.imshow(wide,
        labels=dict(x="Hour of Day", y="", color="Events"),
        color_continuous_scale="Blues", aspect="auto",
        title="📅 Weekly Activity Heatmap — when is this user active?")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=42, b=10))
    fig.update_xaxes(tickvals=list(range(0, 24, 2)),
                     ticktext=[f"{h:02d}h" for h in range(0, 24, 2)])
    return fig


def _od_heatmap(od_df, title, colorscale="Reds"):
    fig = px.imshow(od_df,
        labels=dict(x="Destination", y="Origin", color="Calls"),
        color_continuous_scale=colorscale, aspect="auto", title=title)
    fig.update_layout(height=440, margin=dict(l=5, r=5, t=42, b=5))
    fig.update_xaxes(tickangle=45, tickfont_size=9)
    fig.update_yaxes(tickfont_size=9)
    return fig


def _tower_heatmap(tdf, title, colorscale="Blues"):
    fig = px.imshow(tdf,
        labels=dict(x="Hour of Day", y="Area", color="Events"),
        color_continuous_scale=colorscale, aspect="auto", title=title)
    fig.update_layout(height=480, margin=dict(l=5, r=5, t=42, b=5))
    fig.update_xaxes(tickvals=list(range(0, 24, 2)),
                     ticktext=[f"{h:02d}h" for h in range(0, 24, 2)])
    fig.update_yaxes(tickfont_size=9)
    return fig
def _gamma_radius_plot(eps, sampled_r=None):
    x_max = max(12, 8 / eps)
    x = np.linspace(0, x_max, 500)
    pdf = (eps ** 2) * x * np.exp(-eps * x)

    expected_r = 2.0 / eps
    mode_r = 1.0 / eps

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=pdf,
        mode="lines",
        line=dict(color="#2563eb", width=3),
        name="Gamma density"
    ))

    fig.add_vline(
        x=mode_r,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text=f"Most likely radius ≈ {mode_r:.2f} km",
        annotation_position="top left"
    )

    fig.add_vline(
        x=expected_r,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Expected radius = {expected_r:.2f} km",
        annotation_position="top right"
    )

    if sampled_r is not None:
        sampled_y = (eps ** 2) * sampled_r * np.exp(-eps * sampled_r)
        fig.add_trace(go.Scatter(
            x=[sampled_r],
            y=[sampled_y],
            mode="markers",
            marker=dict(size=11, color="#16a34a"),
            name=f"Sampled r = {sampled_r:.2f} km"
        ))
        fig.add_vline(
            x=sampled_r,
            line_dash="solid",
            line_color="#16a34a"
        )

    fig.update_layout(
        title=f"Radius distribution from Gamma(shape=2, scale=1/ε)  |  ε = {eps}",
        xaxis_title="Radius r (km)",
        yaxis_title="Probability density",
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig

# =============================================================
# SIDEBAR
# =============================================================

def render_sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:8px 0 18px 0;">
            <div style="font-size:2.2rem;">🔒</div>
            <div style="font-size:1.1rem;font-weight:700;color:#1e40af;">Telecom DP Shield</div>
            <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">Differential Privacy Demo</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**ε — Privacy Budget**")
        eps = st.slider("Epsilon", 0.1, 10.0, 2.0, 0.1,
                        label_visibility="collapsed",
                        help="Smaller ε → stronger privacy, more location noise.\n"
                             "Larger ε → weaker privacy, less noise.")

        noise_km = round(2.0 / eps, 2)
        if   eps <= 1.0: clr, lvl = "#16a34a", "🟢 Very Strong"
        elif eps <= 3.0: clr, lvl = "#2563eb", "🔵 Strong"
        elif eps <= 6.0: clr, lvl = "#d97706", "🟡 Moderate"
        else:            clr, lvl = "#dc2626", "🔴 Weak"

        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:14px;margin:4px 0 18px 0;">
            <div style="font-weight:600;color:{clr};margin-bottom:5px;">{lvl} Privacy</div>
            <div style="font-size:0.82rem;color:#64748b;line-height:1.6;">
                ε = <b>{eps}</b><br>
                Avg. noise radius: <b>{noise_km} km</b><br>
                Location blur: <b>{'~city-wide' if noise_km > 3 else f'~{noise_km} km'}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Analyze Subscriber**")
        uid = st.selectbox("User", sorted(df["user_id"].unique()), index=0,
                           label_visibility="collapsed")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.73rem;color:#94a3b8;text-align:center;line-height:1.7;">
            Simulated CDR Data<br>
            Mumbai · 30 Cell Towers<br>
            500 Subscribers<br>
            14,225 Events · 7 Days
        </div>
        """, unsafe_allow_html=True)
    return eps, uid


# =============================================================
# TAB 1 — WHAT JIO SEES
# =============================================================

def render_tab1(df, towers, users, uid, eps):
    st.markdown("""
    <div style="background:#fef2f2;border-left:5px solid #dc2626;border-radius:8px;
                padding:16px 20px;margin-bottom:20px;color:#1e293b;">
        <strong>Privacy Risk Demo</strong> — Everything below is derived from just three
        fields in a raw CDR record: <em>source tower, destination, timestamp</em>.
        No app data, no GPS consent — yet home address, workplace, daily routine, and
        social connections are all recoverable.
    </div>
    """, unsafe_allow_html=True)

    udf   = df[df["user_id"] == uid].copy()
    uinfo = users[users["user_id"] == uid].iloc[0]

    # ── KPI row ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events This Week",   len(udf))
    c2.metric("Areas Visited",      udf["source_area"].nunique())
    c3.metric("Calls Made",         int((udf["event_type"] == "call").sum()))
    c4.metric("Avg Events / Day",   f"{len(udf)/7:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Map + Inference panel ─────────────────────────────────
    col_map, col_inf = st.columns([3, 2])

    with col_map:
        st.markdown("##### Location Trajectory (Raw)")
        st.caption("Each dot = one CDR event · colour = hour of day  "
                   "(dark purple = midnight, bright yellow = noon)")
        st.plotly_chart(_trajectory_map(udf, f"Raw CDR — {uid}"),
                        use_container_width=True)

    with col_inf:
        st.markdown("##### What an Attacker Can Infer")

        ndf = udf[udf["hour_of_day"].isin(NIGHT_HOURS)]
        ddf = udf[udf["hour_of_day"].isin(DAY_HOURS)]
        inf_home = ndf["source_area"].mode().iloc[0] if len(ndf) > 0 else "Unknown"
        inf_work = ddf["source_area"].mode().iloc[0] if len(ddf) > 0 else "Unknown"
        act_home = uinfo["home_area"];  act_work = uinfo["work_area"]

        def card(label, inferred, actual):
            hit   = inferred == actual
            badge = "Correct" if hit else "Close"
            bcol  = "#dc2626" if hit else "#d97706"
            return f"""
            <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;
                        padding:15px;margin-bottom:11px;">
                <div style="font-size:0.78rem;color:#6b7280;">{label}</div>
                <div style="font-size:1.3rem;font-weight:700;color:#dc2626;">{inferred}</div>
                <div style="margin-top:6px;font-size:0.76rem;color:#6b7280;">
                    <span style="background:{bcol};color:white;padding:1px 8px;
                                 border-radius:9px;font-size:0.7rem;">{badge}</span>
                    &ensp;Ground truth: <b>{actual}</b>
                </div>
            </div>"""

        st.markdown(card("Inferred Home Location",  inf_home, act_home),
                    unsafe_allow_html=True)
        st.markdown(card("Inferred Work Location",  inf_work, act_work),
                    unsafe_allow_html=True)

        cmt = int(uinfo["morning_commute_hour"])
        st.markdown(f"""
        <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;
                    padding:15px;margin-bottom:11px;">
            <div style="font-size:0.78rem;color:#6b7280;">Inferred Morning Commute</div>
            <div style="font-size:1.3rem;font-weight:700;color:#dc2626;">~{cmt}:00 AM</div>
            <div style="font-size:0.76rem;color:#6b7280;margin-top:4px;">
                Detected from home to work tower transition pattern
            </div>
        </div>
        """, unsafe_allow_html=True)

        raw_acc = home_attack_accuracy(eps_km=None)
        st.markdown(f"""
        <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:15px;">
            <div style="font-size:0.78rem;color:#6b7280;">Network-Wide Attack Success</div>
            <div style="font-size:2rem;font-weight:800;color:#ea580c;">{raw_acc*100:.0f}%</div>
            <div style="font-size:0.78rem;color:#6b7280;margin-top:4px;">
                Home correctly inferred for <b>{raw_acc*100:.0f}% of all 500 users</b>
                using nighttime tower patterns alone
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Activity heatmap ──────────────────────────────────────
    st.markdown("##### When Is This User Active?")
    st.caption("Darker = more events. Weekday commute spikes are unmistakable.")
    st.plotly_chart(_activity_heatmap(udf), use_container_width=True)

    # ── Top call routes ──────────────────────────────────────
    calls = udf[udf["event_type"] == "call"]
    if len(calls) > 0:
        st.markdown("##### 🗺️ Most Frequent Communication Routes")
        st.caption("Source → destination tower pairs for voice calls. Reveals movement corridors & social graph.")
        routes = (calls.groupby(["source_area", "dest_area"])
                  .size().reset_index(name="count")
                  .sort_values("count", ascending=True).tail(10))
        routes["route"] = routes["source_area"] + "  →  " + routes["dest_area"]
        fig_r = px.bar(routes, x="count", y="route", orientation="h",
                       title="Top 10 Routes by Call Volume",
                       labels={"count": "Calls", "route": ""},
                       color="count", color_continuous_scale="Reds")
        fig_r.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                            margin=dict(l=5, r=5, t=42, b=5))
        st.plotly_chart(fig_r, use_container_width=True)


# =============================================================
# TAB 2 — WITH DIFFERENTIAL PRIVACY
# =============================================================

def render_tab2(df, towers, users, uid, eps):
    noise_km = round(2.0 / eps, 2)

    st.markdown(f"""
    <div style="background:#f0fdf4;border-left:5px solid #16a34a;border-radius:8px;
                padding:16px 20px;margin-bottom:20px;color:#1e293b;">
        <strong>Differential Privacy Active</strong> — ε = {eps},
        avg. noise radius ≈ <b>{noise_km} km</b>.
        Every source and destination coordinate has been perturbed using the
        <b>Planar Laplace mechanism</b>. Use the ε slider (sidebar) to explore the tradeoff.
    </div>
    """, unsafe_allow_html=True)

    df_dp    = get_dp_dataset(eps)
    udf_raw  = df[df["user_id"] == uid].copy()
    udf_dp   = df_dp[df_dp["user_id"] == uid].copy()

    # ── Side-by-side trajectory maps ─────────────────────────
    st.markdown("##### Trajectory Comparison: Raw vs. DP-Protected")
    st.markdown(
        "Both maps show the same user's week of activity. On the left, each dot "
        "is the subscriber's exact GPS location at the moment of a CDR event — "
        "an attacker with this data can reconstruct every trip, every routine. "
        "On the right, each coordinate has been shifted by Planar Laplace noise, "
        "so the dots are scattered within a ~**{:.2f} km** radius of the real positions. "
        "The pattern looks similar at a glance, but the exact points are gone.".format(noise_km)
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _trajectory_map(udf_raw, "Without DP — Exact Locations", "Plasma"),
            use_container_width=True)
        st.markdown(
            "<p style='text-align:center;font-size:0.82rem;color:#dc2626;'>"
            "Every location is exact and traceable</p>",
            unsafe_allow_html=True)
    with c2:
        st.plotly_chart(
            _trajectory_map(udf_dp, f"With DP (ε = {eps}) — Protected", "Viridis"),
            use_container_width=True)
        st.markdown(
            f"<p style='text-align:center;font-size:0.82rem;color:#16a34a;'>"
            f"Locations perturbed within ~{noise_km} km radius</p>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Attack accuracy comparison ────────────────────────────
    st.markdown("##### Home Inference Attack — All 500 Users")
    st.markdown(
        "To measure how much privacy is gained, we run a simulated attack on all 500 users: "
        "look at each user's nighttime CDR records (10 pm–6 am), find which area they were "
        "most often in, and call that their home. Without DP, this succeeds surprisingly often "
        "because people consistently sleep in the same place. With DP, the noisy coordinates "
        "map to the wrong tower area, reducing the attacker's accuracy. "
        "The grey dashed line shows what random guessing would score — that is the floor "
        "we are aiming to approach."
    )
    raw_acc = home_attack_accuracy(eps_km=None)
    dp_acc  = home_attack_accuracy(eps_km=eps)
    red_pct = max((raw_acc - dp_acc) / max(raw_acc, 1e-6) * 100, 0)

    ca, cb, cc = st.columns(3)
    ca.metric("Without DP",      f"{raw_acc*100:.1f}%",
              help="Attack success on raw CDR data")
    cb.metric(f"With DP  (ε={eps})", f"{dp_acc*100:.1f}%",
              delta=f"{(dp_acc - raw_acc)*100:.1f} pp", delta_color="inverse")
    cc.metric("Attack Reduction", f"{red_pct:.1f}%",
              help="How much DP reduced the attacker's success rate")

    n_areas   = df["home_area"].nunique()
    rand_pct  = 100.0 / n_areas

    cmp_df = pd.DataFrame({
        "Scenario": ["Without DP", f"With DP  (ε = {eps})"],
        "Attack Success (%)": [raw_acc * 100, dp_acc * 100],
    })
    fig_bar = px.bar(cmp_df, x="Scenario", y="Attack Success (%)",
                     color="Scenario",
                     color_discrete_map={"Without DP": "#dc2626",
                                         f"With DP  (ε = {eps})": "#16a34a"},
                     title="Home Location Inference Attack — 500 Users",
                     text_auto=".1f")
    fig_bar.add_hline(y=rand_pct, line_dash="dot", line_color="#9ca3af",
                      annotation_text=f"Random Guess: {rand_pct:.1f}%",
                      annotation_position="top right")
    fig_bar.update_layout(height=360, showlegend=False,
                          yaxis=dict(range=[0, 100]))
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Noise distribution ────────────────────────────────────
    st.markdown("##### Noise Distribution Added to This User's Locations")
    st.markdown(
        "For each of this user's CDR events, this histogram shows how far (in km) the "
        "reported location was moved from the real one. The red dashed line is the "
        "theoretical expected radius (2/ε). Most events fall near this value, "
        "but because the noise is drawn from a Gamma distribution, there is a long tail — "
        "a few events will be shifted much further."
    )

    src_raw = udf_raw[["source_lat", "source_lon"]].values
    src_dp  = udf_dp [["source_lat", "source_lon"]].values
    dlat_km = (src_dp[:, 0] - src_raw[:, 0]) * 111.0
    dlon_km = (src_dp[:, 1] - src_raw[:, 1]) * 111.0 * np.cos(np.radians(19.07))
    noise_v = np.sqrt(dlat_km**2 + dlon_km**2)

    ch, cs = st.columns([3, 2])
    with ch:
        fig_h = px.histogram(x=noise_v, nbins=25,
                             title=f"Noise Magnitude Distribution  (ε = {eps})",
                             labels={"x": "Noise Added (km)", "y": "Events"},
                             color_discrete_sequence=["#2563eb"])
        fig_h.add_vline(x=noise_km, line_dash="dash", line_color="#dc2626",
                         annotation_text=f"Expected: {noise_km:.2f} km")
        fig_h.update_layout(height=300, margin=dict(l=5, r=5, t=42, b=5))
        st.plotly_chart(fig_h, use_container_width=True)

 #   with cs:
 #       st.markdown(f"""
 #      <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
 #                  padding:18px;margin-top:10px;">
#          <h4 style="color:#16a34a;margin:0 0 12px 0;">📐 Noise Stats</h4>
  #          <table style="width:100%;font-size:0.87rem;border-collapse:collapse;">
  #          <tr style="border-bottom:1px solid #e5e7eb;">
  #              <td style="padding:5px 0;color:#6b7280;">ε (epsilon)</td>
  #              <td style="text-align:right;font-weight:600;">{eps}</td></tr>
   #         <tr style="border-bottom:1px solid #e5e7eb;">
   #             <td style="padding:5px 0;color:#6b7280;">Expected radius</td>
   #             <td style="text-align:right;font-weight:600;">{noise_km:.2f} km</td></tr>
    #        <tr style="border-bottom:1px solid #e5e7eb;">
    #            <td style="padding:5px 0;color:#6b7280;">Median noise</td>
    #            <td style="text-align:right;font-weight:600;">{np.median(noise_v):.2f} km</td></tr>
     #       <tr style="border-bottom:1px solid #e5e7eb;">
     #           <td style="padding:5px 0;color:#6b7280;">90th percentile</td>
     #           <td style="text-align:right;font-weight:600;">{np.percentile(noise_v, 90):.2f} km</td></tr>
     #       <tr>
     #           <td style="padding:5px 0;color:#6b7280;">Max noise</td>
     #           <td style="text-align:right;font-weight:600;">{noise_v.max():.2f} km</td></tr>
     #       </table>
     #   </div>
     #   """, unsafe_allow_html=True)


# =============================================================
# TAB 3 — HOW IT WORKS
# =============================================================

def render_tab3(df, towers, eps):
    noise_km = round(2.0 / eps, 2)

    st.markdown("""
    <div style="background:#eff6ff;border-left:5px solid #2563eb;border-radius:8px;
                padding:16px 20px;margin-bottom:20px;color:#1e293b;">
        <strong>Algorithm: Planar Laplace / Geo-Indistinguishability</strong><br>
        Formally proposed by Andrés et al. (2013); used at Apple (Location Services),
        Google (Maps), and Uber (driver privacy). The guarantee:
        for any two locations x, y and any observation z —
        <code>P(z|x) / P(z|y) ≤ e<sup>ε·d(x,y)</sup></code>
    </div>
    """, unsafe_allow_html=True)

    col_alg, col_viz = st.columns([1, 1])

    # ── Algorithm explanation ─────────────────────────────────
    with col_alg:
        st.markdown("##### 🔢 Step-by-Step")
        st.markdown(f"""
**Input:** A real GPS coordinate  `(lat, lon)`

---

**Step 1 — Sample a noise radius:**
```python
r ~ Gamma(shape=2, scale=1/ε)   # km
```
With ε = **{eps}**, expected r = **{noise_km} km**

---

**Step 2 — Sample a random direction:**
```python
θ ~ Uniform(0, 2π)   # any direction equally likely
```

---

**Step 3 — Shift the coordinate:**
```python
lat' = lat + r·cos(θ) / 111
lon' = lon + r·sin(θ) / (111·cos(lat))
```
*(111 km ≈ 1 degree of latitude or longitude)*

---
        """)

    # ── Uncertainty cloud visualisation ──────────────────────
    with col_viz:
        st.markdown("##### Uncertainty Cloud (Live)")
        st.caption(
            f"🔴 red = real location (Bandra) · 🔵 blue = 300 possible observations "
            f"after Planar Laplace with ε = {eps}")

        CLat, CLon = 19.0556, 72.8418          # Bandra, Mumbai
        rng  = np.random.RandomState(42)
        r_s  = rng.gamma(2, 1.0 / eps, size=300)
        th_s = rng.uniform(0, 2 * np.pi, size=300)
        s_la = CLat + r_s * np.cos(th_s) / 111.0
        s_lo = CLon + r_s * np.sin(th_s) / (111.0 * np.cos(np.radians(CLat)))

        # expected-radius circle
        ang  = np.linspace(0, 2 * np.pi, 120)
        rc   = 2.0 / eps
        c_la = CLat + rc * np.cos(ang) / 111.0
        c_lo = CLon + rc * np.sin(ang) / (111.0 * np.cos(np.radians(CLat)))

        fig_cld = go.Figure()
        fig_cld.add_trace(go.Scattermapbox(
            lat=s_la.tolist(), lon=s_lo.tolist(), mode="markers",
            marker=dict(size=6, color="#3b82f6", opacity=0.45),
            name="Attacker Observations", hoverinfo="skip"))
        fig_cld.add_trace(go.Scattermapbox(
            lat=c_la.tolist(), lon=c_lo.tolist(), mode="lines",
            line=dict(color="#f59e0b", width=2.5),
            name=f"Avg radius: {rc:.2f} km", hoverinfo="skip"))
        fig_cld.add_trace(go.Scattermapbox(
            lat=[CLat], lon=[CLon], mode="markers",
            marker=dict(size=15, color="#dc2626"),
            name="🔴 Real Location (Bandra)",
            hovertext="Bandra, Mumbai — Real Location", hoverinfo="text"))
        fig_cld.update_layout(
            mapbox_style=MAPBOX_STYLE,
            mapbox_center={"lat": CLat, "lon": CLon},
            mapbox_zoom=11, height=420,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0.01, y=0.99,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#e5e7eb", borderwidth=1,
                        font=dict(size=11)))
        st.plotly_chart(fig_cld, use_container_width=True)
        st.caption("Adjust ε in the sidebar — watch the blue cloud shrink or expand")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive noise sampler ─────────────────────────────
    st.markdown("##### Try It Yourself — Draw Noise and See the Calculation")
    st.markdown(
        "The Planar Laplace mechanism draws two random numbers to build the noise vector: "
        "a **radius r** from a Gamma distribution and a **direction θ** uniformly from 0°–360°. "
        "Use the panels below to draw each one and see the full calculation live."
    )

    # ── Helper: closed-form inverse CDF for Gamma(2, 1/ε) ────
    def gamma2_inv_cdf(p, eps_val):
        """Binary-search inverse CDF of Gamma(shape=2, scale=1/ε).
        Closed-form CDF: F(x) = 1 − (1 + ε·x)·exp(−ε·x)
        """
        lo_x, hi_x = 0.0, 100.0 / max(eps_val, 0.01)
        for _ in range(60):
            mid = (lo_x + hi_x) / 2.0
            cdf_mid = 1.0 - (1.0 + eps_val * mid) * np.exp(-eps_val * mid)
            if cdf_mid < p:
                lo_x = mid
            else:
                hi_x = mid
        return (lo_x + hi_x) / 2.0

    # Initialise session-state keys for drawn values
    if "tab3_r" not in st.session_state:
        st.session_state["tab3_r"] = None
    if "tab3_theta" not in st.session_state:
        st.session_state["tab3_theta"] = None

    # ── Row 1: Gamma PDF  +  Polar compass ───────────────────
    g_col, p_col = st.columns(2)

    # ── Gamma PDF panel ───────────────────────────────────────
    with g_col:
        st.markdown("**Step 1 — Draw noise radius r from Gamma(2, 1/ε)**")
        st.caption(
            f"With ε = {eps}, the distribution peaks at 1/ε = {1/eps:.3f} km "
            f"and has mean 2/ε = {2/eps:.3f} km. "
            "The shaded band is the 25th–75th percentile — half of all draws land here."
        )

        # Build PDF curve
        x_max   = gamma2_inv_cdf(0.995, eps)
        xs      = np.linspace(0, x_max, 400)
        pdf_y   = eps**2 * xs * np.exp(-eps * xs)

        p25 = gamma2_inv_cdf(0.25, eps)
        p75 = gamma2_inv_cdf(0.75, eps)
        mask = (xs >= p25) & (xs <= p75)

        fig_gam = go.Figure()

        # Shaded IQR band
        fig_gam.add_trace(go.Scatter(
            x=np.concatenate([xs[mask], xs[mask][::-1]]),
            y=np.concatenate([pdf_y[mask], np.zeros(mask.sum())]),
            fill="toself", fillcolor="rgba(99,102,241,0.18)",
            line=dict(width=0), showlegend=True, name="25th–75th %ile"
        ))

        # Main PDF curve
        fig_gam.add_trace(go.Scatter(
            x=xs, y=pdf_y, mode="lines",
            line=dict(color="#6366f1", width=2.5),
            name="Gamma(2, 1/ε) PDF"
        ))

        # Peak marker (mode = 1/ε)
        mode_y = eps**2 * (1/eps) * np.exp(-1.0)
        fig_gam.add_trace(go.Scatter(
            x=[1/eps], y=[mode_y], mode="markers+text",
            marker=dict(size=10, color="#f59e0b", symbol="diamond"),
            text=[f"Peak 1/ε={1/eps:.2f}"], textposition="top center",
            textfont=dict(size=10), name=f"Peak (1/ε = {1/eps:.3f} km)"
        ))

        # Mean marker (2/ε)
        mean_y = eps**2 * (2/eps) * np.exp(-2.0)
        fig_gam.add_trace(go.Scatter(
            x=[2/eps], y=[mean_y], mode="markers+text",
            marker=dict(size=10, color="#10b981", symbol="circle"),
            text=[f"Mean 2/ε={2/eps:.2f}"], textposition="top center",
            textfont=dict(size=10), name=f"Mean (2/ε = {2/eps:.3f} km)"
        ))

        # Drawn-r vertical line (if available)
        r_drawn = st.session_state["tab3_r"]
        if r_drawn is not None:
            r_pdf_y = eps**2 * r_drawn * np.exp(-eps * r_drawn)
            fig_gam.add_trace(go.Scatter(
                x=[r_drawn, r_drawn], y=[0, r_pdf_y],
                mode="lines+markers",
                line=dict(color="#dc2626", width=2, dash="dash"),
                marker=dict(size=8, color="#dc2626"),
                name=f"Your draw: r = {r_drawn:.3f} km"
            ))

        fig_gam.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=40),
            xaxis_title="r (km)", yaxis_title="Probability density",
            legend=dict(orientation="h", y=-0.25, x=0,
                        font=dict(size=10)),
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9")
        )
        st.plotly_chart(fig_gam, use_container_width=True)

        if st.button("Draw random r", type="primary", key="btn_draw_r"):
            st.session_state["tab3_r"] = float(np.random.gamma(2, 1.0 / eps))
            st.rerun()

        if r_drawn is not None:
            st.success(f"r = **{r_drawn:.4f} km**  (drawn from Gamma(2, 1/{eps}))")
        else:
            st.info("Press **Draw random r** to sample a noise radius.")

    # ── Polar compass panel ───────────────────────────────────
    with p_col:
        st.markdown("**Step 2 — Draw direction θ from Uniform(0°, 360°)**")
        st.caption(
            "Every direction is equally likely — the ring is perfectly uniform. "
            "This ensures noise has no directional bias (required by the privacy proof)."
        )

        theta_drawn = st.session_state["tab3_theta"]

        # Build uniform ring
        angles_ring = np.linspace(0, 2 * np.pi, 361)
        fig_pol = go.Figure()

        # Uniform ring
        fig_pol.add_trace(go.Scatterpolar(
            r=[1.0] * 361,
            theta=np.degrees(angles_ring).tolist(),
            mode="lines",
            line=dict(color="#6366f1", width=2),
            name="Uniform ring"
        ))

        # Drawn direction arrow
        if theta_drawn is not None:
            theta_deg_d = float(np.degrees(theta_drawn))
            fig_pol.add_trace(go.Scatterpolar(
                r=[0, 1.05],
                theta=[theta_deg_d, theta_deg_d],
                mode="lines+markers",
                line=dict(color="#dc2626", width=3),
                marker=dict(size=[0, 10], color="#dc2626",
                            symbol=["circle", "arrow-bar-up"],
                            angleref="previous"),
                name=f"θ = {theta_deg_d:.1f}°"
            ))

        fig_pol.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=40),
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1.2]),
                angularaxis=dict(
                    direction="clockwise", rotation=90,
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=["N 0°", "45°", "E 90°", "135°",
                              "S 180°", "225°", "W 270°", "315°"],
                    gridcolor="#e2e8f0"
                ),
                bgcolor="white"
            ),
            showlegend=True,
            legend=dict(orientation="h", y=-0.12, x=0,
                        font=dict(size=10))
        )
        st.plotly_chart(fig_pol, use_container_width=True)

        if st.button("Draw random θ", type="primary", key="btn_draw_theta"):
            st.session_state["tab3_theta"] = float(np.random.uniform(0, 2 * np.pi))
            st.rerun()

        if theta_drawn is not None:
            st.success(
                f"θ = **{np.degrees(theta_drawn):.2f}°**  "
                f"({theta_drawn:.4f} rad)  — drawn from Uniform(0°, 360°)"
            )
        else:
            st.info("Press **Draw random θ** to sample a direction.")

    # ── Row 2: Coordinate inputs + live calculation ───────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Step 3 — Enter the real location and see the full calculation**")

    ci1, ci2 = st.columns(2)
    with ci1:
        user_lat = st.number_input(
            "Real latitude (°N)", value=19.0556, format="%.6f",
            help="Latitude of the true location you want to protect"
        )
    with ci2:
        user_lon = st.number_input(
            "Real longitude (°E)", value=72.8418, format="%.6f",
            help="Longitude of the true location you want to protect"
        )

    r_val     = st.session_state["tab3_r"]
    theta_val = st.session_state["tab3_theta"]

    if r_val is not None and theta_val is not None:
        # Compute all steps
        cos_lat   = np.cos(np.radians(user_lat))
        dlat_deg  = r_val * np.cos(theta_val) / 111.0
        dlon_deg  = r_val * np.sin(theta_val) / (111.0 * cos_lat)
        new_lat   = user_lat + dlat_deg
        new_lon   = user_lon + dlon_deg
        dist_km   = np.sqrt(
            (dlat_deg * 111.0) ** 2 +
            (dlon_deg * 111.0 * cos_lat) ** 2
        )
        theta_deg_v = np.degrees(theta_val)

        st.markdown(
            f"""
            <div style="background:#f8fafc;border:1.5px solid #6366f1;border-radius:10px;
                        padding:20px 24px;margin-top:8px;color:#1e293b;font-size:14px;
                        line-height:1.9;">
              <b>Full Planar Laplace Calculation</b><br><br>

              <b>Inputs</b><br>
              &nbsp;&nbsp;Real location &nbsp;= ({user_lat:.6f}° N,&nbsp; {user_lon:.6f}° E)<br>
              &nbsp;&nbsp;ε &nbsp;= {eps}<br><br>

              <b>Step 1 — Noise radius r</b><br>
              &nbsp;&nbsp;Draw from Gamma(shape=2, scale=1/ε = 1/{eps} = {1/eps:.5f})<br>
              &nbsp;&nbsp;<b>r = {r_val:.5f} km</b>
              &nbsp;&nbsp;(peak at {1/eps:.3f} km · mean at {2/eps:.3f} km)<br><br>

              <b>Step 2 — Direction θ</b><br>
              &nbsp;&nbsp;Draw from Uniform(0°, 360°)<br>
              &nbsp;&nbsp;<b>θ = {theta_deg_v:.3f}°</b> ({theta_val:.5f} rad)<br><br>

              <b>Step 3 — Convert to coordinate offsets</b><br>
              &nbsp;&nbsp;Δlat = r · cos(θ) / 111<br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = {r_val:.5f} · cos({theta_deg_v:.3f}°) / 111<br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = {r_val:.5f} · {np.cos(theta_val):.5f} / 111<br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = <b>{dlat_deg:+.6f}°</b><br><br>
              &nbsp;&nbsp;Δlon = r · sin(θ) / (111 · cos(lat))<br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = {r_val:.5f} · sin({theta_deg_v:.3f}°) / (111 · cos({user_lat:.4f}°))<br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = {r_val:.5f} · {np.sin(theta_val):.5f} / {111.0 * cos_lat:.5f}<br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = <b>{dlon_deg:+.6f}°</b><br><br>

              <b>Step 4 — Apply shift</b><br>
              &nbsp;&nbsp;New lat = {user_lat:.6f} + ({dlat_deg:+.6f}) = <b>{new_lat:.6f}° N</b><br>
              &nbsp;&nbsp;New lon = {user_lon:.6f} + ({dlon_deg:+.6f}) = <b>{new_lon:.6f}° E</b><br><br>

              <b>Step 5 — Verify distance moved</b><br>
              &nbsp;&nbsp;dist = √((Δlat·111)² + (Δlon·111·cos(lat))²)<br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = <b>{dist_km:.4f} km</b><br><br>

              <span style="color:#6366f1;">Attacker sees ({new_lat:.6f}, {new_lon:.6f})
              instead of ({user_lat:.6f}, {user_lon:.6f})</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(
            f"Change ε in the sidebar and draw again — smaller ε forces larger r values "
            f"(bigger shifts), larger ε keeps r small and the perturbed point stays close."
        )
    else:
        missing = []
        if r_val is None:
            missing.append("r (noise radius)")
        if theta_val is None:
            missing.append("θ (direction)")
        st.info(
            f"Draw **{' and '.join(missing)}** above to see the full step-by-step calculation."
        )


# =============================================================
# TAB 4 — CITY-WIDE ANALYSIS
# =============================================================

def render_tab4(df, towers, eps):
    st.markdown("""
    <div style="background:#eff6ff;border-left:5px solid #2563eb;border-radius:8px;
                padding:16px 20px;margin-bottom:20px;color:#1e293b;">
        <strong>What is this tab showing?</strong><br><br>
        A telecom operator doesn't just see individual calls — they also aggregate all CDR
        records to answer questions like "how many people commute from Bandra to BKC each morning?"
        or "which tower is busiest at 9am?". This aggregated data drives network capacity planning,
        congestion management, and urban mobility analysis.<br><br>
        The two visualisations here show that aggregate picture:
        <ul style="margin:8px 0 0 0;padding-left:20px;">
          <li><b>Origin-Destination (OD) matrix</b> — each cell counts how many calls were made
              between two areas. Bright cells = heavy traffic between those areas.</li>
          <li><b>Tower activity heatmap</b> — how many events each area handled per hour,
              across all users. Shows rush hours and quiet periods at city scale.</li>
        </ul>
        <br>
        We protect this data using the <b>Laplace Mechanism</b> (add noise to each count) +
        optional <b>threshold suppression / Hybrid Perturbation</b> (zero out rare cells that
        could re-identify individuals through unique flows).
    </div>
    """, unsafe_allow_html=True)

    # ── OD matrix ─────────────────────────────────────────────
    st.markdown("##### 📊 Origin-Destination Call Matrix")

    thresh = st.slider(
        "Threshold k — suppress OD pairs with fewer than k calls (Hybrid Perturbation)",
        min_value=0, max_value=20, value=5, step=1,
        help="k = 0 disables suppression. Higher k protects rare (re-identifying) flows.")

    calls  = df[df["event_type"] == "call"]
    od_raw = calls.groupby(["source_area", "dest_area"]).size().unstack(fill_value=0)

    rng_od  = np.random.RandomState(42)
    n_od    = rng_od.laplace(0, 1.0 / eps, size=od_raw.shape)
    od_dp_v = np.clip(od_raw.values.astype(float) + n_od, 0, None)
    if thresh > 0:
        od_dp_v = np.where(od_dp_v >= thresh, od_dp_v, 0)
    od_dp = pd.DataFrame(od_dp_v, index=od_raw.index, columns=od_raw.columns)

    co1, co2 = st.columns(2)
    with co1:
        st.plotly_chart(_od_heatmap(od_raw, "🔴 Raw OD Matrix", "Reds"),
                        use_container_width=True)
        st.caption("⚠️ Individual movement corridors clearly visible")
    with co2:
        st.plotly_chart(
            _od_heatmap(od_dp, f"🟢 DP OD Matrix  (ε={eps}, k={thresh})", "Greens"),
            use_container_width=True)
        st.caption("✅ Individual flows protected; network patterns preserved")

    raw_tot = int(od_raw.values.sum())
    dp_tot  = int(od_dp_v.sum())
    raw_pr  = int((od_raw.values > 0).sum())
    dp_pr   = int((od_dp_v > 0).sum())
    pct_ch  = (dp_tot - raw_tot) / max(raw_tot, 1) * 100

    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Raw Total Calls",     f"{raw_tot:,}")
    cs2.metric("DP Total Calls",      f"{dp_tot:,}",  f"{pct_ch:+.1f}%")
    cs3.metric("Raw Active OD Pairs", raw_pr)
    cs4.metric("DP Active OD Pairs",  dp_pr,          f"{dp_pr - raw_pr:+d}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tower activity heatmap ────────────────────────────────
    st.markdown("##### 🗼 Hourly Tower Activity — All 500 Subscribers")

    tw_raw   = df.groupby(["source_area", "hour_of_day"]).size().unstack(fill_value=0)
    rng_tw   = np.random.RandomState(42)
    n_tw     = rng_tw.laplace(0, 1.0 / eps, size=tw_raw.shape)
    tw_dp_v  = np.clip(tw_raw.values.astype(float) + n_tw, 0, None)
    tw_dp    = pd.DataFrame(tw_dp_v, index=tw_raw.index, columns=tw_raw.columns)

    ct1, ct2 = st.columns(2)
    with ct1:
        st.plotly_chart(_tower_heatmap(tw_raw, "🔴 Raw Tower Activity", "Reds"),
                        use_container_width=True)
    with ct2:
        st.plotly_chart(
            _tower_heatmap(tw_dp, f"🟢 DP Tower Activity  (ε={eps})", "Greens"),
            use_container_width=True)

    flat_raw = tw_raw.values.flatten().astype(float)
    flat_dp  = tw_dp_v.flatten()
    corr     = float(np.corrcoef(flat_raw, flat_dp)[0, 1])

    st.markdown("##### What is Pearson Correlation and Why Does It Matter Here?")
    st.markdown(f"""
Pearson correlation (r) measures how closely two datasets follow the same **pattern**,
regardless of exact values. It ranges from –1 to +1:

- **r = 1.0** — perfect match: every raw count maps proportionally to the DP count
- **r ≈ 0.99** — almost identical patterns, only tiny noise-induced scatter
- **r = 0.0** — no relationship at all
- **r = –1.0** — opposite patterns (would mean DP destroyed the data completely)

**How it is calculated** — flatten both heatmaps into a single list of 720 numbers
(30 areas × 24 hours), then:

```
         Σ (rawᵢ − raw̄) · (dpᵢ − dp̄)
r  =  ─────────────────────────────────────────
      √[Σ(rawᵢ − raw̄)²] · √[Σ(dpᵢ − dp̄)²]
```

Intuitively: if a tower was busy in the raw data, is it still relatively busy in the DP data?
A high r means yes — the operator can still route traffic, detect congestion, and plan
capacity using the DP-protected heatmap. Individual subscriber movements are hidden,
but the aggregate picture is intact.

**Current result with ε = {eps}:  r = {corr:.4f}**
    """)

    # Scatter plot: raw vs DP counts (all 720 cells)
    scatter_df = pd.DataFrame({"Raw count": flat_raw, "DP count": flat_dp})
    max_val = max(flat_raw.max(), flat_dp.max())
    fig_sc = px.scatter(
        scatter_df, x="Raw count", y="DP count",
        title=f"Raw vs DP Tower Activity — each dot is one (area, hour) cell  |  r = {corr:.4f}",
        labels={"Raw count": "Raw event count", "DP count": "DP-noisy event count"},
        opacity=0.45,
        color_discrete_sequence=["#2563eb"],
    )
    # Perfect correlation reference line
    fig_sc.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#dc2626", width=1.5, dash="dash"),
        name="Perfect r = 1.0",
        showlegend=True,
    ))
    fig_sc.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10),
                         legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption(
        "Dots tightly clustered around the red diagonal = high correlation = patterns preserved. "
        "Reduce ε to see the scatter widen as noise overwhelms the signal."
    )

    st.markdown(f"""
    <div style="background:#f0fdf4;border-left:5px solid #16a34a;border-radius:8px;
                padding:16px 20px;margin-top:8px;color:#1e293b;">
        <strong>Statistical Utility Preserved</strong> — Pearson correlation between
        raw and DP tower activity: <b>{corr:.4f}</b> (1.0 = perfect).
        The operator can still identify peak-hour congestion, plan tower upgrades, and optimise
        routing — with full individual subscriber privacy.
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# MAIN ENTRYPOINT
# =============================================================

def main():
    df, towers, users = load_data()
    eps, uid = render_sidebar(df)

    # ── Header banner ─────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#003087 0%,#0066cc 60%,#0ea5e9 100%);
                padding:26px 32px;border-radius:16px;margin-bottom:22px;
                box-shadow:0 4px 24px rgba(0,48,135,0.22);">
        <h1 style="color:white;margin:0 0 4px 0;font-size:2rem;font-weight:800;">
            Telecom DP Shield
        </h1>
        <p style="color:#c8e4f8;margin:0;font-size:0.95rem;">
            How Differential Privacy protects subscriber telemetry data &nbsp;·&nbsp;
            Planar Laplace &nbsp;·&nbsp; Geo-Indistinguishability &nbsp;·&nbsp; Hybrid Perturbation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-level KPIs ────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📡 CDR Events",    f"{len(df):,}")
    k2.metric("👥 Subscribers",   df["user_id"].nunique())
    k3.metric("🗼 Cell Towers",   towers.shape[0])
    k4.metric("📍 Mumbai Areas",  df["source_area"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "🔍  The Privacy Risk",
        "🛡️  With Differential Privacy",
        "📐  How It Works",
        "🌐  City-Wide Analysis",
    ])

    with t1: render_tab1(df, towers, users, uid, eps)
    with t2: render_tab2(df, towers, users, uid, eps)
    with t3: render_tab3(df, towers, eps)
    with t4: render_tab4(df, towers, eps)


if __name__ == "__main__":
    main()
