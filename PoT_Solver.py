# streamlit app: 5x5 Weighted Path Solver (clickable graphic)
# pip install plotly streamlit-plotly-events

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

GridPos = Tuple[int, int]  # (row, col)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ====== Grid size & visual sizing ======
N = 5
CELL_PX = 70     # pixel size of each cell (visual only)
GAP_PX  = 6      # pixel gap between cells (visual only)

# ====== Internal value mapping (used by solver) ======
LOW_VAL  = 1
MED_VAL  = 3
HIGH_VAL = 5
DEFAULT_VAL = 1

# ====== Color/state mapping (display only) ======
# We'll encode per-cell "state" as ints for a discrete Heatmap:
# 0 empty, 1 low, 2 med, 3 high, 4 obstacle, 5 start, 6 end
STATE_EMPTY    = 0
STATE_LOW      = 1
STATE_MED      = 2
STATE_HIGH     = 3
STATE_OBS      = 4
STATE_START    = 5
STATE_END      = 6

STATE_TO_LABEL = {
    STATE_EMPTY: "Empty",
    STATE_LOW:   "Low",
    STATE_MED:   "Med",
    STATE_HIGH:  "High",
    STATE_OBS:   "Obstacle",
    STATE_START: "Start",
    STATE_END:   "End",
}

# Colors (hex) for each state (matching your vibe)
STATE_COLORS = {
    STATE_EMPTY: "#FFFFFF",   # white
    STATE_LOW:   "#32CD32",   # limegreen
    STATE_MED:   "#1E90FF",   # dodgerblue
    STATE_HIGH:  "#FFD700",   # gold
    STATE_OBS:   "#FF4D4D",   # red
    STATE_START: "#00BFFF",   # deepskyblue
    STATE_END:   "#800080",   # purple
}

# Build a discrete Plotly colorscale from the mapping above
# We place flat bands around each integer value (0..6).
def make_discrete_colorscale():
    keys = list(range(0, 7))
    cols = [STATE_COLORS[k] for k in keys]
    # map ints 0..6 onto 0..1 domain
    steps = [(k / 6.0) for k in keys]
    cs = []
    for i, s in enumerate(steps):
        col = cols[i]
        # tiny epsilon band to keep each value flat
        left = max(0.0, s - 1e-6)
        right = min(1.0, s + 1e-6)
        cs.append([left, col])
        cs.append([right, col])
    return cs

DISCRETE_CS = make_discrete_colorscale()

# =========================
# Core grid/solver helpers
# =========================
def in_bounds(r: int, c: int, n: int = N) -> bool:
    return 0 <= r < n and 0 <= c < n

def neighbors(pos: GridPos, n: int = N):
    r, c = pos
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc, n):
            yield (nr, nc)

def reachable(n: int, start: GridPos, end: GridPos,
              obstacles: Set[GridPos], visited: Set[GridPos]) -> bool:
    blocked = (obstacles | visited) - {start}
    if start in blocked or end in blocked:
        return False
    from collections import deque
    q = deque([start])
    seen = {start}
    while q:
        p = q.popleft()
        if p == end:
            return True
        for nb in neighbors(p, n):
            if nb not in seen and nb not in blocked:
                seen.add(nb)
                q.append(nb)
    return False

def reachable_cells(n: int, start: GridPos,
                    obstacles: Set[GridPos], visited: Set[GridPos]) -> Set[GridPos]:
    blocked = (obstacles | visited) - {start}
    if start in blocked:
        return set()
    q = deque([start])
    seen = {start}
    while q:
        p = q.popleft()
        for nb in neighbors(p, n):
            if nb not in seen and nb not in blocked:
                seen.add(nb)
                q.append(nb)
    return seen

def solve_max_value_path(
    n: int,
    start: GridPos,
    end: GridPos,
    obstacles: Set[GridPos],
    values: Dict[GridPos, int],
) -> Optional[Tuple[List[GridPos], int]]:
    if start == end:
        return ([start], values.get(start, DEFAULT_VAL)) if (start not in obstacles) else None
    if start in obstacles or end in obstacles:
        return None

    def cell_value(cell: GridPos) -> int:
        return values.get(cell, DEFAULT_VAL)

    if not reachable(n, start, end, obstacles, set()):
        return None

    free_cells = {(r, c) for r in range(n) for c in range(n)} - obstacles
    degree = {cell: sum(((cell[0]+dr, cell[1]+dc) in free_cells) for dr, dc in DIRS) for cell in free_cells}

    best_path: List[GridPos] = []
    best_val: int = -10**12
    visited: Set[GridPos] = {start}

    def optimistic_bound(cur: GridPos, cur_sum: int) -> int:
        rcells = reachable_cells(n, cur, obstacles, visited)
        return cur_sum + sum(cell_value(c) for c in rcells if c != cur)

    def manhattan(a: GridPos, b: GridPos) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def dfs(cur: GridPos, path: List[GridPos], cur_sum: int):
        nonlocal best_path, best_val
        if not reachable(n, cur, end, obstacles, visited):
            return
        if optimistic_bound(cur, cur_sum) < best_val:
            return

        if cur == end:
            if (cur_sum > best_val) or (cur_sum == best_val and len(path) > len(best_path)):
                best_val = cur_sum
                best_path = path.copy()
            return

        nxt = [nb for nb in neighbors(cur, n) if nb not in obstacles and nb not in visited]
        nxt.sort(key=lambda x: (-cell_value(x), degree.get(x, 0), manhattan(x, end)))
        for nb in nxt:
            visited.add(nb)
            path.append(nb)
            dfs(nb, path, cur_sum + cell_value(nb))
            path.pop()
            visited.remove(nb)

    dfs(start, [start], cell_value(start))
    if not best_path:
        return None
    return best_path, best_val

# =================
# Session defaults
# =================
if "start" not in st.session_state:
    st.session_state.start: GridPos = (4, 1)
if "end" not in st.session_state:
    st.session_state.end: GridPos = (0, 3)
if "obstacles" not in st.session_state:
    st.session_state.obstacles: Set[GridPos] = set()
if "cell_states" not in st.session_state:
    # 2D grid of states for display only
    st.session_state.cell_states = [[STATE_EMPTY for _ in range(N)] for _ in range(N)]
if "solution" not in st.session_state:
    st.session_state.solution = None
if "tool" not in st.session_state:
    st.session_state.tool = "Start"

# Initialize state grid with start/end markers (keep them unique)
def enforce_unique_start_end():
    # Clear any other start/end before setting
    sr, sc = st.session_state.start
    er, ec = st.session_state.end
    for r in range(N):
        for c in range(N):
            if (r, c) == (sr, sc):
                st.session_state.cell_states[r][c] = STATE_START
            elif (r, c) == (er, ec):
                st.session_state.cell_states[r][c] = STATE_END
            else:
                # don't override non-start/end states here
                if st.session_state.cell_states[r][c] in (STATE_START, STATE_END):
                    st.session_state.cell_states[r][c] = STATE_EMPTY

enforce_unique_start_end()

# =============
# Streamlit UI
# =============
st.set_page_config(page_title="5x5 Weighted Path Solver (Clickable Graphic)", page_icon="🧩", layout="centered")

st.title("5×5 Weighted Path Solver (Clickable Graphic)")
st.caption("Pick a tool, click squares on the grid to configure. Press **Solve** to draw the max-value path.")

# Toolbar
col_tool, col_actions = st.columns([1.2, 1])
with col_tool:
    tool = st.radio(
        "Tool",
        ["Start", "End", "Obstacle", "Low", "Med", "High", "Erase"],
        index=["Start", "End", "Obstacle", "Low", "Med", "High", "Erase"].index(st.session_state.tool),
        horizontal=True,
    )
    st.session_state.tool = tool

with col_actions:
    if st.button("Solve", use_container_width=True):
        # Build obstacles and values from cell_states
        obstacles: Set[GridPos] = set()
        values: Dict[GridPos, int] = {}
        for r in range(N):
            for c in range(N):
                state = st.session_state.cell_states[r][c]
                if state == STATE_OBS:
                    obstacles.add((r, c))
                elif state == STATE_LOW:
                    values[(r, c)] = LOW_VAL
                elif state == STATE_MED:
                    values[(r, c)] = MED_VAL
                elif state == STATE_HIGH:
                    values[(r, c)] = HIGH_VAL
                # empty/start/end default to 1 unless explicitly set above

        res = solve_max_value_path(
            N, st.session_state.start, st.session_state.end, obstacles, values
        )
        st.session_state.solution = res

    c1, c2, c3 = st.columns(3)
    if c1.button("Clear Weights"):
        for r in range(N):
            for c in range(N):
                if st.session_state.cell_states[r][c] in (STATE_LOW, STATE_MED, STATE_HIGH):
                    st.session_state.cell_states[r][c] = STATE_EMPTY
        st.session_state.solution = None
    if c2.button("Clear Obstacles"):
        for r in range(N):
            for c in range(N):
                if st.session_state.cell_states[r][c] == STATE_OBS:
                    st.session_state.cell_states[r][c] = STATE_EMPTY
        st.session_state.solution = None
    if c3.button("Clear All (keep S/E)"):
        for r in range(N):
            for c in range(N):
                st.session_state.cell_states[r][c] = STATE_EMPTY
        enforce_unique_start_end()
        st.session_state.solution = None

st.divider()

# =====================
# Click handling + Plot
# =====================
# Build a Z matrix of display states
z = np.array(st.session_state.cell_states, dtype=int)

fig = go.Figure(
    data=go.Heatmap(
        z=z,
        zmin=0, zmax=6,
        colorscale=DISCRETE_CS,
        showscale=False,
        x=np.arange(N),              # columns
        y=np.arange(N),              # rows
        hovertemplate="(%{y}, %{x})<extra></extra>",
    )
)

# Make grid lines to look like cells
fig.update_layout(
    width=N*CELL_PX + (N-1)*GAP_PX + 40,
    height=N*CELL_PX + (N-1)*GAP_PX + 40,
    margin=dict(l=10, r=10, t=10, b=10),
)

# Put origin at top-left for “matrix” feel
fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=list(range(N)), ticktext=[str(i) for i in range(N)])
fig.update_xaxes(tickmode="array", tickvals=list(range(N)), ticktext=[str(i) for i in range(N)])

# Draw solved path as arrows, if available
fig.update_layout(annotations=[])  # clear
if st.session_state.solution:
    path, total = st.session_state.solution
    if path:
        # Add line segments as arrow annotations between cell centers
        anns = []
        for i in range(len(path)-1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            # centers at (x=c+0.5, y=r+0.5)
            anns.append(dict(
                x=c2+0.5, y=r2+0.5, xref="x", yref="y",
                ax=c1+0.5, ay=r1+0.5, axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=3, opacity=0.9, arrowcolor="black"
            ))
        fig.update_layout(annotations=anns)

# Render and capture click
clicks = plotly_events(
    fig, click_event=True, hover_event=False, select_event=False, override_height=None, override_width=None, key="grid"
)

# If user clicked a cell, update its state according to tool
if clicks:
    c = int(round(clicks[0]["x"]))
    r = int(round(clicks[0]["y"]))
    st.session_state.solution = None  # invalidate on edit

    if st.session_state.tool == "Obstacle":
        # toggle obstacle (but not on S/E)
        if (r, c) not in (st.session_state.start, st.session_state.end):
            st.session_state.cell_states[r][c] = (
                STATE_EMPTY if st.session_state.cell_states[r][c] == STATE_OBS else STATE_OBS
            )

    elif st.session_state.tool in ("Low", "Med", "High"):
        if (r, c) not in (st.session_state.start, st.session_state.end):
            state_map = {"Low": STATE_LOW, "Med": STATE_MED, "High": STATE_HIGH}
            st.session_state.cell_states[r][c] = state_map[st.session_state.tool]

    elif st.session_state.tool == "Erase":
        if (r, c) not in (st.session_state.start, st.session_state.end):
            st.session_state.cell_states[r][c] = STATE_EMPTY

    elif st.session_state.tool == "Start":
        # clear old start, set new
        oldr, oldc = st.session_state.start
        st.session_state.cell_states[oldr][oldc] = STATE_EMPTY
        st.session_state.start = (r, c)
        st.session_state.cell_states[r][c] = STATE_START
        # cannot have start also be end
        if st.session_state.end == (r, c):
            st.session_state.end = None

    elif st.session_state.tool == "End":
        oldr, oldc = st.session_state.end
        if st.session_state.end is not None:
            st.session_state.cell_states[oldr][oldc] = STATE_EMPTY
        st.session_state.end = (r, c)
        st.session_state.cell_states[r][c] = STATE_END
        if st.session_state.start == (r, c):
            st.session_state.start = None

# Info panel
st.divider()
if st.session_state.solution is None:
    st.info("Click squares to configure, then press **Solve**.")
else:
    path, total = st.session_state.solution
    if not path:
        st.error("No path found.")
    else:
        st.success(f"Max value: {total} | Path length: {len(path)}")

with st.expander("Show configuration (for debugging)"):
    st.write(f"Start: {st.session_state.start}")
    st.write(f"End: {st.session_state.end}")
    obs = sorted([(r, c) for r in range(N) for c in range(N) if st.session_state.cell_states[r][c] == STATE_OBS])
    st.write(f"Obstacles: {obs}")
