# streamlit app: 5x5 Weighted Path Solver (Pure Streamlit, CSS Grid of forms + SVG path overlay)

from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import streamlit as st

# ------------------------
# Config / constants
# ------------------------
st.set_page_config(page_title="5x5 Weighted Path Solver", page_icon="🧩", layout="centered")

GridPos = Tuple[int, int]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N = 5  # grid size

# Visual sizing for the grid
CELL = 70      # px per cell
GAP  = 6       # px between cells
GRID_W = N * CELL + (N - 1) * GAP

# Colors by state
COLORS = {
    "empty": "white",
    "low":   "#32CD32",  # limegreen
    "med":   "#1E90FF",  # dodgerblue
    "high":  "#FFD700",  # gold
    "obs":   "#FF4D4D",  # red
    "start": "#00BFFF",  # deepskyblue
    "end":   "#800080",  # purple
}
TOOLS = ["Start", "End", "Obstacle", "Low", "Med", "High", "Erase"]

# ------------------------
# Solver helpers
# ------------------------
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
        return ([start], values.get(start, 1)) if (start not in obstacles) else None
    if start in obstacles or end in obstacles:
        return None

    def cell_value(cell: GridPos) -> int:
        return values.get(cell, 1)

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

# ------------------------
# Session state
# ------------------------
if "grid" not in st.session_state:
    st.session_state.grid = [["empty" for _ in range(N)] for _ in range(N)]
if "start" not in st.session_state:
    st.session_state.start = (4, 1)
    st.session_state.grid[4][1] = "start"
if "end" not in st.session_state:
    st.session_state.end = (0, 3)
    st.session_state.grid[0][3] = "end"
if "tool" not in st.session_state:
    st.session_state.tool = "Start"
if "solution" not in st.session_state:
    st.session_state.solution = None

# ------------------------
# Toolbar
# ------------------------
st.title("5×5 Weighted Path Solver (Tight Squares • Pure Streamlit)")
st.caption("Click squares to set Start/End/Obstacle/Weights. Press Solve to draw the path.")

col_tool, col_actions = st.columns([1.5, 1])
with col_tool:
    st.session_state.tool = st.radio("Tool", TOOLS, index=TOOLS.index(st.session_state.tool), horizontal=True)
with col_actions:
    if st.button("Solve", use_container_width=True):
        obs: Set[GridPos] = set()
        vals: Dict[GridPos, int] = {}
        for r in range(N):
            for c in range(N):
                s = st.session_state.grid[r][c]
                if s == "obs":
                    obs.add((r, c))
                elif s == "low":
                    vals[(r, c)] = 1
                elif s == "med":
                    vals[(r, c)] = 3
                elif s == "high":
                    vals[(r, c)] = 5
        st.session_state.solution = solve_max_value_path(N, st.session_state.start, st.session_state.end, obs, vals)

    c1, c2, c3 = st.columns(3)
    if c1.button("Clear Weights"):
        for r in range(N):
            for c in range(N):
                if st.session_state.grid[r][c] in ("low", "med", "high"):
                    st.session_state.grid[r][c] = "empty"
        st.session_state.solution = None
    if c2.button("Clear Obstacles"):
        for r in range(N):
            for c in range(N):
                if st.session_state.grid[r][c] == "obs":
                    st.session_state.grid[r][c] = "empty"
        st.session_state.solution = None
    if c3.button("Clear All (keep S/E)"):
        st.session_state.grid = [["empty" for _ in range(N)] for _ in range(N)]
        sr, sc = st.session_state.start
        er, ec = st.session_state.end
        st.session_state.grid[sr][sc] = "start"
        st.session_state.grid[er][ec] = "end"
        st.session_state.solution = None

st.divider()

# ------------------------
# Apply tool on click
# ------------------------
def apply_tool(r: int, c: int):
    st.session_state.solution = None
    g = st.session_state.grid
    t = st.session_state.tool
    if t == "Erase":
        if g[r][c] not in ("start", "end"):
            g[r][c] = "empty"
    elif t == "Obstacle":
        if g[r][c] not in ("start", "end"):
            g[r][c] = "obs" if g[r][c] != "obs" else "empty"
    elif t in ("Low", "Med", "High"):
        if g[r][c] not in ("start", "end"):
            g[r][c] = t.lower()
    elif t == "Start":
        oldr, oldc = st.session_state.start
        g[oldr][oldc] = "empty"
        g[r][c] = "start"
        st.session_state.start = (r, c)
    elif t == "End":
        oldr, oldc = st.session_state.end
        g[oldr][oldc] = "empty"
        g[r][c] = "end"
        st.session_state.end = (r, c)

# ------------------------
# CSS: tight CSS Grid of forms; buttons fill each cell
# ------------------------
st.markdown(f"""
<style>
/* center page to grid width so columns/forms don't stretch */
section.main > div.block-container {{
  max-width: {GRID_W + 32}px !important;
}}

/* Grid wrapper: N columns of fixed CELL px with GAP px between */
#gridwrap {{
  display: grid;
  grid-template-columns: repeat({N}, {CELL}px);
  grid-auto-rows: {CELL}px;
  gap: {GAP}px;
  width: {GRID_W}px;
  margin: 0 auto;
}}

/* Each Streamlit form behaves like a square tile */
#gridwrap form {{
  margin: 0 !important;
  padding: 0 !important;
  width: {CELL}px !important;
  height: {CELL}px !important;
}}

/* Make the submit button fill the tile and color it */
#gridwrap form [data-testid="baseButton-secondary"],
#gridwrap form button {{
  width: 100% !important;
  height: 100% !important;
  min-width: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 1px solid rgba(0,0,0,0.25) !important;
  border-radius: 8px !important;
  line-height: 1 !important;
  font-weight: 700 !important;
  color: #102A43 !important; /* label color for S/E letters if any */
}}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Grid: one <form> per square (no columns)
# ------------------------
st.markdown('<div id="gridwrap">', unsafe_allow_html=True)

# Helper: robust rerun that works across Streamlit versions
def _rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

for r in range(N):
    for c in range(N):
        state = st.session_state.grid[r][c]
        label = "S" if state == "start" else ("E" if state == "end" else " ")
        bg = COLORS[state]  # Inline style for background color per tile
        with st.form(key=f"cell-{r}-{c}"):
            clicked = st.form_submit_button(
                label,
                help=f"({r},{c}) {state}",
                kwargs={"style": f"background-color:{bg};"}
            )
            if clicked:
                apply_tool(r, c)
                _rerun()  # <-- immediate visual update on the same click

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ------------------------
# Path overlay SVG (same visual size as grid)
# ------------------------
def center_of(rc: GridPos):
    rr, cc = rc
    x = cc * (CELL + GAP) + CELL / 2
    y = rr * (CELL + GAP) + CELL / 2
    return x, y

if st.session_state.solution:
    path, total = st.session_state.solution
    rects = []
    for rr in range(N):
        for cc in range(N):
            fill = COLORS[st.session_state.grid[rr][cc]]
            rects.append(
                f'<rect x="{cc*(CELL+GAP)}" y="{rr*(CELL+GAP)}" width="{CELL}" height="{CELL}" '
                f'rx="6" ry="6" stroke="gray" fill="{fill}" />'
            )
    lines = []
    for i in range(len(path)-1):
        x1, y1 = center_of(path[i])
        x2, y2 = center_of(path[i+1])
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="3" marker-end="url(#arrow)" />'
        )
    svg = f"""
    <svg width="{GRID_W}" height="{GRID_W}" viewBox="0 0 {GRID_W} {GRID_W}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="black" />
        </marker>
      </defs>
      {''.join(rects)}
      {''.join(lines)}
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)
    st.success(f"Max value: {total} | Path length: {len(path)}")
else:
    st.info("Click tiles to configure, then press **Solve**.")
