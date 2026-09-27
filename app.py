from __future__ import annotations

from dataclasses import asdict

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from floodguard.config import SETTINGS
from floodguard.models import Point
from floodguard.orchestrator import copilot
from floodguard.providers import configured_status, call_gemini, call_muse, call_nemotron
from floodguard.simulation import simulate

st.set_page_config(page_title="FloodGuard AI", page_icon="🌊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1500px; padding-top: 1rem;}
      .fg-title {font-size:2.4rem;font-weight:750;letter-spacing:-.045em;}
      .fg-sub {color:#667085;margin-top:-8px;margin-bottom:16px;}
      .chip {display:inline-block;padding:5px 9px;border-radius:999px;border:1px solid #e5e7eb;background:#fafafa;margin-right:5px;font-size:.82rem;}
      .evidence {padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa;margin-bottom:7px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="fg-title">FloodGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="fg-sub">From flood impact to safer decisions · flood simulation · real roads · multi-model AI</div>',
    unsafe_allow_html=True,
)

for key, default in {
    "point_a": None,
    "point_b": None,
    "storm_center": None,
    "result": None,
    "hazard": None,
    "ai_trace": None,
    "selection_mode": "Point A",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("1 · Scenario")
    rain_mode = st.selectbox(
        "Rainfall input",
        ["Custom scenario", "30 mm", "70 mm", "100 mm", "120 mm", "Live Open-Meteo + scenario"],
    )
    if rain_mode == "Custom scenario":
        rainfall = st.slider("Scenario rainfall (mm)", 5, 180, 70, 5)
        use_live = False
    elif rain_mode == "Live Open-Meteo + scenario":
        rainfall = st.slider("Scenario rainfall (mm)", 5, 180, 70, 5)
        use_live = True
    else:
        rainfall = int(rain_mode.split()[0])
        use_live = False
    duration = st.slider("Rain duration (min)", 10, 180, 60, 10)
    storm_radius = st.slider("Flood / storm area radius (km)", 1.0, 10.0, 3.0, 0.5)
    objective = st.selectbox("Routing objective", ["Balanced", "Safety", "Speed"])

    st.header("2 · Map selection")
    selection_mode = st.radio("What does the next map click set?", ["Point A", "Point B", "Flood area center"], index=0)
    st.session_state.selection_mode = selection_mode
    st.caption("Choose a mode, then click the map. A blue marker is A, purple is B, and the orange circle is the simulated flood area.")

    if st.button("Use A/B midpoint as flood center", use_container_width=True):
        if st.session_state.point_a and st.session_state.point_b:
            a, b = st.session_state.point_a, st.session_state.point_b
            st.session_state.storm_center = Point((a.lat + b.lat) / 2, (a.lon + b.lon) / 2)
            st.session_state.result = None
            st.session_state.hazard = None
            st.rerun()

    st.divider()
    st.header("3 · AI stack")
    for status in configured_status():
        icon = "🟢" if status.configured else "⚪"
        st.write(f"{icon} **{status.provider}**")
        st.caption(status.name + " · " + status.message)
    st.caption("Pipeline: Analyze → Cross-check → Synthesize. Models never replace deterministic geographic/routing evidence.")
    if st.button("Test AI stack", use_container_width=True):
        diagnostics = {}
        try:
            _, ms = call_nemotron("Reply with exactly READY", role="primary")
            diagnostics["Nemotron Primary"] = f"ONLINE · {ms:.0f} ms"
        except Exception as exc:
            diagnostics["Nemotron Primary"] = f"ERROR · {exc}"

        if SETTINGS.enable_gemini and SETTINGS.gemini_api_key:
            try:
                _, ms = call_gemini('{"agreement":"CONSENSUS","analysis":"READY"}')
                diagnostics["Gemini Second Opinion"] = f"ONLINE · {ms:.0f} ms"
            except Exception as exc:
                diagnostics["Gemini Second Opinion"] = f"ERROR · {exc}"
        else:
            diagnostics["Gemini Second Opinion"] = "SKIPPED · disabled or not configured"

        if SETTINGS.enable_muse and SETTINGS.nvidia_api_key:
            try:
                _, ms = call_muse("Reply with exactly READY")
                diagnostics["Muse Glimmer"] = f"ONLINE · {ms:.0f} ms"
            except Exception as exc:
                diagnostics["Muse Glimmer"] = f"ERROR · {exc}"
        else:
            diagnostics["Muse Glimmer"] = "SKIPPED · optional / disabled"

        try:
            _, ms = call_nemotron("Return exactly {\"agreement\":\"CONSENSUS\",\"answer\":\"READY\"}", role="synthesis")
            diagnostics["Nemotron Super Final"] = f"ONLINE · {ms:.0f} ms"
        except Exception as exc:
            diagnostics["Nemotron Super Final"] = f"ERROR · {exc}"
        st.session_state["ai_diagnostics"] = diagnostics
    if st.session_state.get("ai_diagnostics"):
        for name, value in st.session_state["ai_diagnostics"].items():
            st.write(f"**{name}:** {value}")

# Keep the map centered on the selected operational area.
points = [p for p in [st.session_state.point_a, st.session_state.point_b, st.session_state.storm_center] if p]
if points:
    center = Point(sum(p.lat for p in points) / len(points), sum(p.lon for p in points) / len(points))
    base = [center.lat, center.lon]
else:
    base = [36.8065, 10.1815]

m = folium.Map(location=base, zoom_start=12, tiles="OpenStreetMap", control_scale=True)

if st.session_state.point_a:
    folium.Marker([st.session_state.point_a.lat, st.session_state.point_a.lon], tooltip="Point A", icon=folium.Icon(color="blue")).add_to(m)
if st.session_state.point_b:
    folium.Marker([st.session_state.point_b.lat, st.session_state.point_b.lon], tooltip="Point B", icon=folium.Icon(color="purple")).add_to(m)
if st.session_state.storm_center:
    folium.Circle(
        [st.session_state.storm_center.lat, st.session_state.storm_center.lon],
        radius=storm_radius * 1000,
        color="#f97316",
        fill=True,
        fill_color="#f97316",
        fill_opacity=0.10,
        tooltip=f"Simulated flood area · {storm_radius:.1f} km radius",
    ).add_to(m)

haz = st.session_state.hazard
if haz is not None:
    for yi in range(len(haz.lats) - 1):
        for xi in range(len(haz.lons) - 1):
            v = float(haz.hazard[yi, xi])
            if v < 0.22:
                continue
            fill = "#fde047" if v < 0.55 else "#fb923c" if v < 0.75 else "#dc2626"
            folium.Rectangle(
                [[haz.lats[yi], haz.lons[xi]], [haz.lats[yi + 1], haz.lons[xi + 1]]],
                color=None,
                fill=True,
                fill_color=fill,
                fill_opacity=min(0.48, 0.05 + v * 0.42),
                tooltip=f"Simulated hazard index {v:.0%}",
            ).add_to(m)

    # Explicit "safest area" marker: the lowest-hazard cell in the operational grid,
    # computed from the same hazard field (not a separate guess). Only shown when the
    # zone is meaningfully safer than the flooded area, so it doesn't clutter a dry run.
    min_idx = np.unravel_index(int(np.argmin(haz.hazard)), haz.hazard.shape)
    safe_v = float(haz.hazard[min_idx])
    safe_lat = float(haz.lats[min_idx[0]])
    safe_lon = float(haz.lons[min_idx[1]])
    if safe_v < 0.20:
        folium.Circle(
            [safe_lat, safe_lon],
            radius=350,
            color="#16a34a",
            fill=True,
            fill_color="#16a34a",
            fill_opacity=0.22,
            tooltip=f"Safest area in the grid · hazard index {safe_v:.0%}",
        ).add_to(m)
        folium.Marker(
            [safe_lat, safe_lon],
            tooltip="Safest area (lowest simulated hazard)",
            icon=folium.Icon(color="green", icon="ok-sign"),
        ).add_to(m)

if st.session_state.result:
    result = st.session_state.result
    for idx, route in enumerate(result.route_candidates):
        pts = [[p.lat, p.lon] for p in route.geometry]
        if len(pts) > 1:
            folium.PolyLine(
                pts,
                weight=7 if idx == result.recommended_route_index else 3,
                opacity=0.95 if idx == result.recommended_route_index else 0.35,
                tooltip=("RECOMMENDED FLOOD-AWARE ROUTE" if idx == result.recommended_route_index else f"Candidate {idx + 1} · {route.flood_exposure:.0%} exposure"),
            ).add_to(m)

st.info("Map workflow: select **Point A**, click; select **Point B**, click; select **Flood area center**, click; then run the simulation.")
map_data = st_folium(m, height=680, use_container_width=True, returned_objects=["last_clicked"])
if map_data and map_data.get("last_clicked"):
    clicked = Point(float(map_data["last_clicked"]["lat"]), float(map_data["last_clicked"]["lng"]))
    mode = st.session_state.selection_mode
    if mode == "Point A":
        st.session_state.point_a = clicked
    elif mode == "Point B":
        st.session_state.point_b = clicked
    else:
        st.session_state.storm_center = clicked
    st.session_state.result = None
    st.session_state.hazard = None
    st.rerun()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Point A", "SET" if st.session_state.point_a else "—")
c2.metric("Point B", "SET" if st.session_state.point_b else "—")
c3.metric("Flood area", "SET" if st.session_state.storm_center else "MIDPOINT")
c4.metric("Scenario", f"{rainfall} mm")

b1, b2, b3 = st.columns([1, 1, 2])
with b1:
    run = st.button("Run FloodGuard Simulation", type="primary", use_container_width=True)
with b2:
    clear = st.button("Clear selection", use_container_width=True)
with b3:
    st.caption("The flood area is an explicit input. It is never silently generated at a random location.")

if clear:
    for k in ("point_a", "point_b", "storm_center", "result", "hazard", "ai_trace"):
        st.session_state[k] = None
    st.rerun()

if run:
    if not st.session_state.point_a or not st.session_state.point_b:
        st.error("Set both Point A and Point B on the map first.")
    else:
        center = st.session_state.storm_center or Point(
            (st.session_state.point_a.lat + st.session_state.point_b.lat) / 2,
            (st.session_state.point_a.lon + st.session_state.point_b.lon) / 2,
        )
        with st.spinner("Building terrain-aware hazard field, routing real roads, and scoring flood impact…"):
            result, hazard = simulate(
                st.session_state.point_a,
                st.session_state.point_b,
                rainfall,
                duration_min=duration,
                use_live_weather=use_live,
                storm_center=center,
                storm_radius_km=storm_radius,
                objective=objective,
            )
        st.session_state.result = result
        st.session_state.hazard = hazard
        st.session_state.ai_trace = None
        st.success(f"Simulation complete · {result.run_id}")

result = st.session_state.result
if result:
    st.divider()
    st.subheader("Flood impact evidence")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Affected area", f"{result.affected_fraction:.0%}")
    k2.metric("High-risk", f"{result.road_high_fraction:.0%}")
    k3.metric("Severe proxy", f"{result.road_blocked_fraction:.0%}")
    k4.metric("Exposure index", f"{result.exposure_index:.0%}")
    k5.metric("Max hazard", f"{result.max_hazard:.0%}")

    st.write(f"**Flood area:** {result.storm_radius_km:.1f} km radius centered at `{result.storm_center.lat:.5f}, {result.storm_center.lon:.5f}`")
    st.write(f"**Effective rainfall forcing:** {result.effective_rainfall_mm:.1f} mm · objective: **{objective}**")

    if result.live_weather:
        st.caption(
            f"LIVE Open-Meteo observation at flood center · {result.live_weather.observed_at} · "
            f"rain {result.live_weather.rain_mm} mm · precipitation {result.live_weather.precipitation_mm} mm · "
            f"temperature {result.live_weather.temperature_c} °C · wind {result.live_weather.wind_kmh} km/h"
        )
    elif use_live:
        st.warning("Live Open-Meteo was requested but unavailable; the selected scenario still ran.")

    st.subheader("Movement decision")
    if result.recommended_route_index is not None:
        route = result.route_candidates[result.recommended_route_index]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Route", "Flood-aware")
        r2.metric("Distance", f"{route.distance_m / 1000:.2f} km")
        r3.metric("ETA", f"{route.duration_s / 60:.0f} min")
        r4.metric("Flood exposure", f"{route.flood_exposure:.0%}")

        names = [step.get("name") for leg in route.raw.get("legs", []) for step in leg.get("steps", []) if step.get("name")]
        if names:
            st.write("**Road sequence:** " + " → ".join(dict.fromkeys(names) if names else names))
        st.success("This route uses real road geometry returned by the routing engine and is selected from the simulated flood-risk evidence.")
    elif result.route_candidates:
        st.warning("Real road candidates were found, but none passes the prototype severe-flood blocking constraint.")
    else:
        st.error("No real road route was returned by the routing providers. Flood simulation remains available, but the app will not invent a route.")

    st.subheader("What changed?")
    for e in result.evidence:
        st.markdown(f'<div class="evidence">{e}</div>', unsafe_allow_html=True)

    with st.expander("All route candidates"):
        for i, r in enumerate(result.route_candidates):
            label = "RECOMMENDED" if i == result.recommended_route_index else f"Candidate {i + 1}"
            st.write(f"**{label}** · {r.summary} · {r.distance_m / 1000:.2f} km · {r.duration_s / 60:.0f} min")

    st.subheader("AI Emergency Copilot")
    st.caption("Analyze → cross-check → synthesize. AI explains computed FloodGuard evidence; it does not invent routes or geographic facts.")
    if "preset_question" not in st.session_state:
        st.session_state.preset_question = ""
    p1, p2, p3 = st.columns(3)
    if p1.button("Which route should I take?", use_container_width=True):
        st.session_state.preset_question = "Which route should I take?"
    if p2.button("Why did the route change?", use_container_width=True):
        st.session_state.preset_question = "Why did the route change?"
    if p3.button("What happens at 120 mm?", use_container_width=True):
        st.session_state.preset_question = "What happens at 120 mm?"
    question = st.text_input(
        "Ask about this exact run",
        value=st.session_state.preset_question,
        placeholder="Why did the route change at 100 mm? Which route should I take exactly?",
    )
    if st.button("Ask FloodGuard AI", type="primary") and question:
        state = {
            "run_id": result.run_id,
            "scenario": {
                "rainfall_mm": result.scenario_mm,
                "effective_rainfall_mm": result.effective_rainfall_mm,
                "duration_min": result.duration_min,
                "storm_center": asdict(result.storm_center),
                "storm_radius_km": result.storm_radius_km,
            },
            "points": {"A": asdict(result.a), "B": asdict(result.b)},
            "simulation": {
                "affected_fraction": result.affected_fraction,
                "high_risk_fraction": result.road_high_fraction,
                "severe_fraction": result.road_blocked_fraction,
                "exposure_index": result.exposure_index,
                "terrain_source": result.terrain_source,
            },
            "routes": [
                {
                    "index": i + 1,
                    "distance_km": r.distance_m / 1000,
                    "eta_min": r.duration_s / 60,
                    "flood_exposure": r.flood_exposure,
                    "blocked_fraction": r.blocked_fraction,
                    "summary": r.summary,
                    "road_names": [
                        step.get("name") for leg in r.raw.get("legs", []) for step in leg.get("steps", []) if step.get("name")
                    ],
                }
                for i, r in enumerate(result.route_candidates)
            ],
            "recommended_route_index": (result.recommended_route_index + 1) if result.recommended_route_index is not None else None,
            "live_weather": asdict(result.live_weather) if result.live_weather else None,
            "evidence": result.evidence,
        }
        with st.spinner("Nemotron analyzing → Gemini/Muse cross-checking → Nemotron Super synthesizing the final answer…"):
            answer = copilot(question, state)
        st.session_state.ai_trace = answer

    if st.session_state.ai_trace:
        answer = st.session_state.ai_trace
        agreement = answer.get("agreement", "UNASSESSED")
        if agreement == "CONSENSUS":
            st.success("Model cross-check: CONSENSUS")
        elif agreement == "PARTIAL_AGREEMENT":
            st.warning("Model cross-check: PARTIAL AGREEMENT")
        elif agreement == "DISAGREEMENT":
            st.warning("Model cross-check: DISAGREEMENT — final answer grounded in deterministic evidence")
        else:
            st.info(f"Model cross-check: {agreement}")
        st.markdown(answer["answer"])
        with st.expander("Model panel · analyze → cross-check → synthesize"):
            st.write("**Pipeline:** Nemotron Primary → Gemini Second Opinion + optional Muse → Nemotron Super Final Synthesis")
            for provider, text in answer["responses"].items():
                latency = answer["latencies"].get(provider)
                st.write(f"### {provider} · {latency:.0f} ms" if latency is not None else f"### {provider}")
                st.write(text)
            st.write("### Operator-facing final synthesis · Nemotron Super")
            st.write(answer["answer"])
            if answer["errors"]:
                st.write("### Provider status / errors")
                st.json(answer["errors"])

st.divider()
st.caption("FloodGuard AI is a decision-support prototype. Its flood hazard is a simulated index, not a calibrated flood forecast or certified emergency-navigation system.")
