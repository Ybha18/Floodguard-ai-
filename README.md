# FloodGuard AI

**From Flood Impact to Safer Decisions**

FloodGuard AI is a flood-impact simulation and decision-support prototype that connects rainfall, terrain, a localized hazard field, real-road routing, live weather evidence, and a multi-model AI copilot in one workflow.

## Product workflow

```text
Point A + Point B + Flood Area + Rainfall + Duration
                         ↓
                FloodGuard Simulation
              Rainfall + terrain + location
                         ↓
                 Spatial Hazard Field
                         ↓
                    Road Impact
                         ↓
                 Real-road Routing
                         ↓
                  AI decision loop
```

The operator can choose Point A, Point B, an explicit flood-area center, rainfall (including 30 / 70 / 100 / 120 mm presets), duration, and a routing objective. The simulation then evaluates real road candidates using the computed hazard evidence.

## AI architecture

The copilot follows the architecture used in the FloodGuard presentation:

```text
                    VERIFIED FLOODGUARD EVIDENCE
                               ↓
                    Nemotron Primary Analysis
                               ↓
              ┌────────────────┴────────────────┐
              ↓                                 ↓
     Gemini Flash Second Opinion      Muse Glimmer (optional)
              └────────────────┬────────────────┘
                               ↓
                    Nemotron Super 120B
                 Cross-check + Final Synthesis
                               ↓
                     Operator-facing answer
```

The roles are intentionally different:

- **Nemotron 3.5 Lightning** — primary evidence analysis / reasoning sub-agent.
- **Gemini 2.5 Flash** — independent second opinion. It does not produce the final operator answer.
- **Muse Glimmer 30B** — optional additional model review, especially useful when multimodal evidence is added later.
- **Nemotron 3 Super 120B A12B** — final cross-check and synthesis. Its JSON result becomes the single operator-facing answer.

FloodGuard's deterministic simulation and routing engines remain the source of truth for coordinates, road geometry, route distances, route candidates, hazard scores, rainfall values, and weather observations. The models explain and cross-check those facts; they do not invent geographic truth.

## Nemotron reasoning handling

Nemotron reasoning is enabled explicitly with the NVIDIA-documented `chat_template_kwargs.enable_thinking` control. The primary analysis has a large enough generation budget to leave room for visible output, while the final Nemotron Super synthesis also has a generous output budget.

Raw reasoning traces are never used as an answer fallback. If a model returns an empty visible `content` field, FloodGuard records that as a provider error instead of exposing `reasoning_content` or a truncated `<think>` trace.

## Structured output

The Gemini second-opinion call requests JSON using the Gemini API's structured-output configuration (`responseMimeType: application/json` + `responseSchema`). The final Nemotron Super agent is prompted to return:

```json
{"agreement":"CONSENSUS|PARTIAL_AGREEMENT|DISAGREEMENT","answer":"..."}
```

The parser accepts plain JSON, fenced JSON, or JSON wrapped in other text, but a malformed final synthesis is never silently promoted to the operator answer.

## Simulation and routing

The flood area is an explicit operator input rather than a random location. The hazard engine creates a spatial hazard index from rainfall, terrain and the selected location/radius. The engine is intentionally described as a **prototype hazard heuristic**, not a calibrated hydrodynamic flood-depth forecast.

FloodGuard requests real-road geometry from OSRM / OpenStreetMap routing services. It evaluates a direct route and real detour candidates using the simulated flood evidence. If no route is returned, the app shows that state rather than fabricating a path.

## Live weather

The `Live Open-Meteo + scenario` option retrieves current weather information at the selected flood-area center. The observation timestamp, precipitation/rain values, temperature and wind are surfaced as evidence and the current rain/precipitation value can raise the effective rainfall forcing used by the simulation.

## Copilot questions

The copilot is intended to answer questions such as:

- Which route should I take exactly?
- Why did the route change?
- What happens at 100 mm?
- Which road should I avoid?
- Which shelter is safest?

The copilot must only answer from the evidence supplied for the current run (or explicit multi-scenario evidence when available). It must not invent missing scenario results, shelter data, road names, distances, coordinates, or weather.

## Responsible AI / limitations

FloodGuard is **decision support, not autonomous emergency control**.

- The flood model is a prototype spatial hazard index, not a certified flood forecast.
- Route choices come from real routing services, not language-model generation.
- AI answers are grounded in deterministic FloodGuard evidence.
- A provider outage or rate limit can degrade the AI cross-check chain; the UI exposes provider errors.
- Free model access is rate-limited and can change at the provider level.

## Environment

Create `.env` from `.env.example`:

```env
NVIDIA_API_KEY=
NVIDIA_PRIMARY_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_SYNTHESIS_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_REASONER_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_VISION_MODEL=meta/muse-glimmer-30b
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
ENABLE_GEMINI=true
ENABLE_MUSE=true

OSRM_URLS=https://router.project-osrm.org;https://routing.openstreetmap.de/routed-car
OPEN_METEO_URL=https://api.open-meteo.com/v1
OPEN_METEO_ELEVATION_URL=https://api.open-meteo.com/v1/elevation
```

`NVIDIA_REASONER_MODEL` is kept as a backward-compatible alias. New deployments should use `NVIDIA_SYNTHESIS_MODEL` for the final Nemotron Super role.

## Windows run

```powershell
cd "C:\path\to\FloodGuard AI"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
notepad .env
python -m streamlit run app.py
```

## Tests

```powershell
python -m compileall -q app.py floodguard tests
python -m pytest -q
```

The tests mock the external AI providers where appropriate. They validate the deterministic simulation, reasoning cleanup, Nemotron request parameters, Gemini structured-output handling, JSON extraction, and the complete **Nemotron → Gemini/Muse → Nemotron Super** orchestration path.
