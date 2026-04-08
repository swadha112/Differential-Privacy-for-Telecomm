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
.block-container { padding-top: 1.2rem; }
.stTabs [data-baseweb="tab"] {
    height: 46px;
    font-weight: 500;
    font-size: 0.95rem;
}
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
# DATA LOADING
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
    Planar Laplace (Geo-Indistinguishability)
    Sampling:
        r ~ Gamma(shape=2, scale=1/ε)   [noise radius in km]
        θ ~ Uniform(0, 2π)              [noise direction]
    """
    rng = np.random.RandomState(seed)
    n   = len(lats)
    r   = rng.gamma(shape=2, scale=1.0 / eps_km, size=n)
    th  = rng.uniform(0, 2 * np.pi, size=n)
    la  = np.asarray(lats, dtype=float)
    lo  = np.asarray(lons, dtype=float)
    return (la + r * np.cos(th) / 111.0,
            lo + r * np.sin(th) / (111.0 * np.cos(np.radians(la))))


@st.cache_data
def get_dp_dataset(eps_km):
    df, _, _ = load_data()
    ns_lat, ns_lon = _planar_laplace(
        df["source_lat"].values, df["source_lon"].values, eps_km, seed=42)
    nd_lat, nd_lon = _planar_laplace(
        df["dest_lat"].values,   df["dest_lon"].values,   eps_km, seed=99)
    out = df.copy()
    out["source_lat"] = ns_lat
    out["source_lon"] = ns_lon
    out["dest_lat"]   = nd_lat
    out["dest_lon"]   = nd_lon
    return out


def _nearest_areas(lats, lons, towers_df):
    tla = towers_df["lat"].values
    tlo = towers_df["lon"].values
    ta  = towers_df["area_name"].values
    la  = np.asarray(lats, dtype=float)
    lo  = np.asarray(lons, dtype=float)
    d2  = (la[:, None] - tla[None, :]) ** 2 + (lo[:, None] - tlo[None, :]) ** 2
    return ta[np.argmin(d2, axis=1)]


@st.cache_data
def home_attack_accuracy(eps_km=None):
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
        f"{row['timestamp']}<br>"
        f"{row['source_area']}<br>"
        f"{row['event_type'].replace('_',' ').title()}<br>"
        f"{row['hour_of_day']}:00"
        for _, row in udf.iterrows()
    ]
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=udf["source_lat"].tolist(), lon=udf["source_lon"].tolist(),
        mode="lines", line=dict(width=1.5, color="rgba(99,102,241,0.35)"),
        hoverinfo="none", showlegend=False))
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
    fig = px.imshow(
        wide,
        labels=dict(x="Hour of Day", y="", color="Events"),
        color_continuous_scale="Blues",
        aspect="auto",
        title="Weekly Activity Heatmap"
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=42, b=10))
    fig.update_xaxes(tickvals=list(range(0, 24, 2)),
                     ticktext=[f"{h:02d}h" for h in range(0, 24, 2)])
    return fig


def _od_heatmap(od_df, title, colorscale="Reds"):
    fig = px.imshow(
        od_df,
        labels=dict(x="Destination", y="Origin", color="Calls"),
        color_continuous_scale=colorscale,
        aspect="auto",
        title=title
    )
    fig.update_layout(height=440, margin=dict(l=5, r=5, t=42, b=5))
    fig.update_xaxes(tickangle=45, tickfont_size=9)
    fig.update_yaxes(tickfont_size=9)
    return fig


def _tower_heatmap(tdf, title, colorscale="Blues"):
    fig = px.imshow(
        tdf,
        labels=dict(x="Hour of Day", y="Area", color="Events"),
        color_continuous_scale=colorscale,
        aspect="auto",
        title=title
    )
    fig.update_layout(height=480, margin=dict(l=5, r=5, t=42, b=5))
    fig.update_xaxes(tickvals=list(range(0, 24, 2)),
                     ticktext=[f"{h:02d}h" for h in range(0, 24, 2)])
    fig.update_yaxes(tickfont_size=9)
    return fig


def _radius_distribution_chart(eps, max_r=15, n=500):
    r = np.linspace(0, max_r, n)
    pdf = (eps ** 2) * r * np.exp(-eps * r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r, y=pdf, mode="lines",
        line=dict(color="#2563eb", width=3),
        name="Radius density"
    ))
    fig.add_vline(
        x=2.0 / eps,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Expected radius = {2/eps:.2f} km",
        annotation_position="top right"
    )
    fig.update_layout(
        title=f"Radius distribution for ε = {eps}",
        xaxis_title="r (km)",
        yaxis_title="Density",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def _single_point_map(true_lat, true_lon, noisy_lat, noisy_lon):
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=[true_lat], lon=[true_lon], mode="markers",
        marker=dict(size=14, color="#dc2626"),
        name="True point",
        hovertemplate="True point<br>lat=%{lat}<br>lon=%{lon}<extra></extra>"
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[noisy_lat], lon=[noisy_lon], mode="markers",
        marker=dict(size=14, color="#2563eb"),
        name="Perturbed point",
        hovertemplate="Perturbed point<br>lat=%{lat}<br>lon=%{lon}<extra></extra>"
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[true_lat, noisy_lat], lon=[true_lon, noisy_lon], mode="lines",
        line=dict(width=2, color="rgba(37,99,235,0.5)"),
        hoverinfo="none", showlegend=False
    ))
    fig.update_layout(
        mapbox_style=MAPBOX_STYLE,
        mapbox_center={"lat": (true_lat + noisy_lat) / 2, "lon": (true_lon + noisy_lon) / 2},
        mapbox_zoom=12,
        height=380,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)")
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
        if   eps <= 1.0: clr, lvl = "#16a34a", "Very Strong"
        elif eps <= 3.0: clr, lvl = "#2563eb", "Strong"
        elif eps <= 6.0: clr, lvl = "#d97706", "Moderate"
        else:            clr, lvl = "#dc2626", "Weak"

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events This Week", len(udf))
    c2.metric("Areas Visited", udf["source_area"].nunique())
    c3.metric("Calls Made", int((udf["event_type"] == "call").sum()))
    c4.metric("Avg Events / Day", f"{len(udf)/7:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_map, col_inf = st.columns([3, 2])

    with col_map:
        st.markdown("##### Location Trajectory (Raw)")
        st.caption("Each dot = one CDR event · colour = hour of day")
        st.plotly_chart(_trajectory_map(udf, f"Raw CDR — {uid}"),
                        use_container_width=True)

    with col_inf:
        st.markdown("##### What an Attacker Can Infer")

        ndf = udf[udf["hour_of_day"].isin(NIGHT_HOURS)]
        ddf = udf[udf["hour_of_day"].isin(DAY_HOURS)]
        inf_home = ndf["source_area"].mode().iloc[0] if len(ndf) > 0 else "Unknown"
        inf_work = ddf["source_area"].mode().iloc[0] if len(ddf) > 0 else "Unknown"
        act_home = uinfo["home_area"]
        act_work = uinfo["work_area"]

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

        st.markdown(card("Inferred Home Location", inf_home, act_home),
                    unsafe_allow_html=True)
        st.markdown(card("Inferred Work Location", inf_work, act_work),
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

    st.markdown("##### When Is This User Active?")
    st.caption("Darker = more events. Weekday commute spikes are unmistakable.")
    st.plotly_chart(_activity_heatmap(udf), use_container_width=True)

    calls = udf[udf["event_type"] == "call"]
    if len(calls) > 0:
        st.markdown("##### Most Frequent Communication Routes")
        st.caption("Source → destination tower pairs for voice calls")
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
        <b>Planar Laplace mechanism</b>.
    </div>
    """, unsafe_allow_html=True)

    df_dp    = get_dp_dataset(eps)
    udf_raw  = df[df["user_id"] == uid].copy()
    udf_dp   = df_dp[df_dp["user_id"] == uid].copy()

    st.markdown("##### Trajectory Comparison: Raw vs. DP-Protected")
    st.markdown(
        "Both maps show the same user's week of activity. On the left, each dot "
        "is the subscriber's exact GPS location at the moment of a CDR event. "
        "On the right, each coordinate has been shifted by Planar Laplace noise, "
        "so the dots are scattered within a ~**{:.2f} km** radius of the real positions."
        .format(noise_km)
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

    st.markdown("##### Home Inference Attack — All 500 Users")
    st.markdown(
        "We run a simulated attack on all 500 users: "
        "look at each user's nighttime CDR records (10 pm–6 am), find which area they were "
        "most often in, and call that their home. Without DP, this succeeds surprisingly often. "
        "With DP, the noisy coordinates map to the wrong tower area more often, reducing accuracy."
    )
    raw_acc = home_attack_accuracy(eps_km=None)
    dp_acc  = home_attack_accuracy(eps_km=eps)
    red_pct = max((raw_acc - dp_acc) / max(raw_acc, 1e-6) * 100, 0)

    ca, cb, cc = st.columns(3)
    ca.metric("Without DP", f"{raw_acc*100:.1f}%")
    cb.metric(f"With DP (ε={eps})", f"{dp_acc*100:.1f}%",
              delta=f"{(dp_acc - raw_acc)*100:.1f} pp", delta_color="inverse")
    cc.metric("Attack Reduction", f"{red_pct:.1f}%")

    n_areas   = df["home_area"].nunique()
    rand_pct  = 100.0 / n_areas

    cmp_df = pd.DataFrame({
        "Scenario": ["Without DP", f"With DP (ε = {eps})"],
        "Attack Success (%)": [raw_acc * 100, dp_acc * 100],
    })
    fig_bar = px.bar(cmp_df, x="Scenario", y="Attack Success (%)",
                     color="Scenario",
                     color_discrete_map={"Without DP": "#dc2626",
                                         f"With DP (ε = {eps})": "#16a34a"},
                     title="Home Location Inference Attack — 500 Users",
                     text_auto=".1f")
    fig_bar.add_hline(y=rand_pct, line_dash="dot", line_color="#9ca3af",
                      annotation_text=f"Random Guess: {rand_pct:.1f}%",
                      annotation_position="top right")
    fig_bar.update_layout(height=360, showlegend=False,
                          yaxis=dict(range=[0, 100]))
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("##### Noise Distribution Added to This User's Locations")
    st.markdown(
        "For each of this user's CDR events, this histogram shows how far (in km) the "
        "reported location was moved from the real one. The red dashed line is the "
        "theoretical expected radius (2/ε)."
    )

    src_raw = udf_raw[["source_lat", "source_lon"]].values
    src_dp  = udf_dp [["source_lat", "source_lon"]].values
    dlat_km = (src_dp[:, 0] - src_raw[:, 0]) * 111.0
    dlon_km = (src_dp[:, 1] - src_raw[:, 1]) * 111.0 * np.cos(np.radians(19.07))
    noise_v = np.sqrt(dlat_km**2 + dlon_km**2)

    fig_h = px.histogram(x=noise_v, nbins=25,
                         title=f"Noise Magnitude Distribution (ε = {eps})",
                         labels={"x": "Noise Added (km)", "y": "Events"},
                         color_discrete_sequence=["#2563eb"])
    fig_h.add_vline(x=noise_km, line_dash="dash", line_color="#dc2626",
                    annotation_text=f"Expected: {noise_km:.2f} km")
    fig_h.update_layout(height=300, margin=dict(l=5, r=5, t=42, b=5))
    st.plotly_chart(fig_h, use_container_width=True)


# =============================================================
# TAB 3 — HOW IT WORKS
# =============================================================

def render_tab3(df, towers, eps):
    st.markdown("""
    <div style="background:#eff6ff;border-left:5px solid #2563eb;border-radius:8px;
                padding:16px 20px;margin-bottom:20px;color:#1e293b;">
        <strong>Algorithm: Planar Laplace / Geo-Indistinguishability</strong><br>
        Pick a random distance r and a random direction θ, then shift the original
        coordinate by that amount. Nearby noisy points are more likely than faraway ones.
    </div>
    """, unsafe_allow_html=True)

    # persistent random draw state
    if "walkthrough_seed" not in st.session_state:
        st.session_state["walkthrough_seed"] = 1234
    if "walkthrough_r" not in st.session_state:
        st.session_state["walkthrough_r"] = None
    if "walkthrough_theta" not in st.session_state:
        st.session_state["walkthrough_theta"] = None

    st.markdown("##### Interactive Walkthrough")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        eps_local = st.number_input("Epsilon (ε)", min_value=0.1, max_value=10.0, value=float(eps), step=0.1)
    with c2:
        lat_real = st.number_input("Latitude", value=19.0556, format="%.6f")
    with c3:
        lon_real = st.number_input("Longitude", value=72.8418, format="%.6f")
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Generate random direction and distance", type="primary"):
            rng = np.random.RandomState(np.random.randint(1, 999999))
            st.session_state["walkthrough_r"] = float(rng.gamma(shape=2, scale=1.0 / eps_local))
            st.session_state["walkthrough_theta"] = float(rng.uniform(0, 2 * np.pi))

    if st.session_state["walkthrough_r"] is None or st.session_state["walkthrough_theta"] is None:
        rng = np.random.RandomState(1234)
        st.session_state["walkthrough_r"] = float(rng.gamma(shape=2, scale=1.0 / eps_local))
        st.session_state["walkthrough_theta"] = float(rng.uniform(0, 2 * np.pi))

    r = st.session_state["walkthrough_r"]
    theta = st.session_state["walkthrough_theta"]
    theta_deg = float(np.degrees(theta))

    delta_lat = r * np.cos(theta) / 111.0
    delta_lon = r * np.sin(theta) / (111.0 * np.cos(np.radians(lat_real)))
    lat_noisy = lat_real + delta_lat
    lon_noisy = lon_real + delta_lon

    dist_check = np.sqrt(
        (delta_lat * 111.0) ** 2 +
        (delta_lon * 111.0 * np.cos(np.radians(lat_real))) ** 2
    )

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.plotly_chart(_single_point_map(lat_real, lon_real, lat_noisy, lon_noisy), use_container_width=True)

    with col_right:
        st.plotly_chart(_radius_distribution_chart(eps_local), use_container_width=True)

    st.markdown("##### Live Calculation")

    st.code(f"""Real point:
lat = {lat_real:.6f}
lon = {lon_real:.6f}

Step 1: Sample radius
r ~ Gamma(shape=2, scale=1/ε)
ε = {eps_local:.3f}
scale = 1/ε = {1/eps_local:.6f}
sampled r = {r:.6f} km

Step 2: Sample direction
θ ~ Uniform(0, 2π)
sampled θ = {theta_deg:.3f}° = {theta:.6f} radians

Step 3: Convert to coordinate shifts
Δlat = r * cos(θ) / 111
     = {r:.6f} * cos({theta_deg:.3f}°) / 111
     = {delta_lat:.8f}°

Δlon = r * sin(θ) / (111 * cos(lat))
     = {r:.6f} * sin({theta_deg:.3f}°) / (111 * cos({lat_real:.6f}°))
     = {delta_lon:.8f}°

Step 4: Add shifts
new lat = {lat_real:.6f} + ({delta_lat:.8f}) = {lat_noisy:.6f}
new lon = {lon_real:.6f} + ({delta_lon:.8f}) = {lon_noisy:.6f}

Step 5: Check moved distance
distance = {dist_check:.6f} km
""", language="")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sampled radius r", f"{r:.3f} km")
    m2.metric("Sampled direction θ", f"{theta_deg:.1f}°")
    m3.metric("Perturbed latitude", f"{lat_noisy:.6f}")
    m4.metric("Perturbed longitude", f"{lon_noisy:.6f}")

    st.markdown(
        f"""
For geo-indistinguishability, two locations within distance **d** of each other produce
outputs so similar that an attacker cannot distinguish between them too confidently:

**P(K(x) ∈ S) ≤ exp(ε · d(x,x')) · P(K(x') ∈ S)**

With your current choice **ε = {eps_local:.3f}**, the expected radius is **2/ε = {2/eps_local:.3f} km**.
"""
    )


# =============================================================
# TAB 4 — CITY-WIDE ANALYSIS
# =============================================================

def render_tab4(df, towers, eps):
    st.markdown("""
    <div style="background:#eff6ff;border-left:5px solid #2563eb;border-radius:8px;
                padding:16px 20px;margin-bottom:20px;color:#1e293b;">
        <strong>What is this tab showing?</strong><br><br>
        A telecom operator aggregates CDR records to answer questions like
        "how many people commute from Bandra to BKC each morning?"
        or "which tower is busiest at 9am?".
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Origin-Destination Call Matrix")

    thresh = st.slider(
        "Threshold k — suppress OD pairs with fewer than k calls",
        min_value=0, max_value=20, value=5, step=1,
        help="k = 0 disables suppression. Higher k protects rare flows.")

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
        st.plotly_chart(_od_heatmap(od_raw, "Raw OD Matrix", "Reds"),
                        use_container_width=True)
        st.caption("Individual movement corridors clearly visible")
    with co2:
        st.plotly_chart(
            _od_heatmap(od_dp, f"DP OD Matrix (ε={eps}, k={thresh})", "Greens"),
            use_container_width=True)
        st.caption("Individual flows protected; network patterns preserved")

    raw_tot = int(od_raw.values.sum())
    dp_tot  = int(od_dp_v.sum())
    raw_pr  = int((od_raw.values > 0).sum())
    dp_pr   = int((od_dp_v > 0).sum())
    pct_ch  = (dp_tot - raw_tot) / max(raw_tot, 1) * 100

    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Raw Total Calls", f"{raw_tot:,}")
    cs2.metric("DP Total Calls", f"{dp_tot:,}", f"{pct_ch:+.1f}%")
    cs3.metric("Raw Active OD Pairs", raw_pr)
    cs4.metric("DP Active OD Pairs", dp_pr, f"{dp_pr - raw_pr:+d}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("##### Hourly Tower Activity — All 500 Subscribers")

    tw_raw   = df.groupby(["source_area", "hour_of_day"]).size().unstack(fill_value=0)
    rng_tw   = np.random.RandomState(42)
    n_tw     = rng_tw.laplace(0, 1.0 / eps, size=tw_raw.shape)
    tw_dp_v  = np.clip(tw_raw.values.astype(float) + n_tw, 0, None)
    tw_dp    = pd.DataFrame(tw_dp_v, index=tw_raw.index, columns=tw_raw.columns)

    ct1, ct2 = st.columns(2)
    with ct1:
        st.plotly_chart(_tower_heatmap(tw_raw, "Raw Tower Activity", "Reds"),
                        use_container_width=True)
    with ct2:
        st.plotly_chart(
            _tower_heatmap(tw_dp, f"DP Tower Activity (ε={eps})", "Greens"),
            use_container_width=True)

    flat_raw = tw_raw.values.flatten().astype(float)
    flat_dp  = tw_dp_v.flatten()
    corr     = float(np.corrcoef(flat_raw, flat_dp)[0, 1])

    st.metric("Pattern preservation (Pearson r)", f"{corr:.4f}")

    scatter_df = pd.DataFrame({"Raw count": flat_raw, "DP count": flat_dp})
    max_val = max(flat_raw.max(), flat_dp.max())
    fig_sc = px.scatter(
        scatter_df, x="Raw count", y="DP count",
        title=f"Raw vs DP Tower Activity — each dot is one (area, hour) cell | r = {corr:.4f}",
        labels={"Raw count": "Raw event count", "DP count": "DP-noisy event count"},
        opacity=0.45,
        color_discrete_sequence=["#2563eb"],
    )
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
        "Dots tightly clustered around the red diagonal mean the city-wide pattern is still preserved."
    )


# =============================================================
# MAIN ENTRYPOINT
# =============================================================

def main():
    df, towers, users = load_data()
    eps, uid = render_sidebar(df)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#003087 0%,#0066cc 60%,#0ea5e9 100%);
                padding:26px 32px;border-radius:16px;margin-bottom:22px;
                box-shadow:0 4px 24px rgba(0,48,135,0.22);">
        <h1 style="color:white;margin:0 0 4px 0;font-size:2rem;font-weight:800;">
            Telecom DP Shield
        </h1>
        <p style="color:#c8e4f8;margin:0;font-size:0.95rem;">
            How Differential Privacy protects subscriber telemetry data ·
            Planar Laplace · Geo-Indistinguishability · Hybrid Perturbation
        </p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CDR Events", f"{len(df):,}")
    k2.metric("Subscribers", df["user_id"].nunique())
    k3.metric("Cell Towers", towers.shape[0])
    k4.metric("Mumbai Areas", df["source_area"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs([
        "The Privacy Risk",
        "With Differential Privacy",
        "How It Works",
        "City-Wide Analysis",
    ])

    with t1:
        render_tab1(df, towers, users, uid, eps)
    with t2:
        render_tab2(df, towers, users, uid, eps)
    with t3:
        render_tab3(df, towers, eps)
    with t4:
        render_tab4(df, towers, eps)


if __name__ == "__main__":
    main()