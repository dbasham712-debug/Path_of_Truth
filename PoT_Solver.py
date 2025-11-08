# streamlit app: 5x5 Weighted Path Solver (Pure Streamlit; Clickable SVG)

from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import streamlit as st
from streamlit.components.v1 import html as html_component

# ==========================
# Config & basic constants
# ==========================
st.set_page_config(page_title="5x5 Weighted Path Solver", page_icon="🧩", layout="centered")

GridPos = Tuple[int, int]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N = 5  # grid size

# Visual sizing (SVG only)
CELL = 70
GAP = 6
W = N * (CELL + GAP) - GAP
H = W

# Color palette by state
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


# ==========================
# Solver helpers
# ==========================
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


# ==========================
# Session state defaults
# ==========================
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


# ==========================
# UI: toolbar
# ==========================
st.title("5×5 Weighted Path Solver (Clickable SVG • Pure Streamlit)")
st.caption("Pick a tool, click squares to configure. Press **Solve** to draw the path.")

col_tool, col_actions = st.columns([1.5, 1])
with col_tool:
    st.session_state.tool = st.radio("Tool", TOOLS, index=TOOLS.index(st.session_state.tool), horizontal=True)
with col_actions:
    if st.button("Solve", use_container_width=True):
        # derive obstacles/values from colors
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


# ==========================
# Click handling + SVG
# ==========================
def apply_tool(r: int, c: int):
    """Apply current tool to cell (r, c)."""
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


def center_of(rc: GridPos):
    r, c = rc
    x = c * (CELL + GAP) + CELL / 2
    y = r * (CELL + GAP) + CELL / 2
    return x, y


def build_svg_markup(path: Optional[List[GridPos]]):
    # Cells
    rects = []
    texts = []
    for r in range(N):
        for c in range(N):
            fill = COLORS[st.session_state.grid[r][c]]
            rects.append(
                f'<rect x="{c*(CELL+GAP)}" y="{r*(CELL+GAP)}" width="{CELL}" height="{CELL}" '
                f'rx="6" ry="6" stroke="gray" fill="{fill}" onclick="cellClick({r},{c})" />'
            )
            # Optional labels for Start/End
            if st.session_state.grid[r][c] == "start":
                x, y = center_of((r, c))
                texts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="20" fill="#00334d">S</text>')
            elif st.session_state.grid[r][c] == "end":
                x, y = center_of((r, c))
                texts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="20" fill="#3f005f">E</text>')

    # Path arrows (if solved)
    lines = []
    if path:
        for i in range(len(path) - 1):
            x1, y1 = center_of(path[i])
            x2, y2 = center_of(path[i + 1])
            lines.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="3" marker-end="url(#arrow)" />'
            )

    # Pure SVG (scripts added in outer HTML shell)
    return f"""
    <svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="black" />
        </marker>
      </defs>
      {''.join(rects)}
      {''.join(lines)}
      {''.join(texts)}
    </svg>
    """


def render_clickable_svg(path: Optional[List[GridPos]]):
    """Render SVG via components.html with proper Streamlit component APIs.
       IMPORTANT: All literal braces in JS are doubled to escape f-string parsing."""
    svg = build_svg_markup(path)
    html = f"""
    <div id="svg-host">{svg}</div>
    <script>
      // Notify Streamlit we're ready & set height
      function ready() {{
        if (window.Streamlit && window.Streamlit.setComponentReady) {{
          window.Streamlit.setComponentReady();
        }}
        if (window.Streamlit && window.Streamlit.setFrameHeight) {{
          window.Streamlit.setFrameHeight({H} + 2);
        }}
      }}
      // Called by <rect onclick="cellClick(r,c)">
      function cellClick(r, c) {{
        if (window.Streamlit && window.Streamlit.setComponentValue) {{
          window.Streamlit.setComponentValue({{row: r, col: c}});
        }}
      }}
      window.cellClick = cellClick;
      if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", ready);
      }} else {{
        ready();
      }}
    </script>
    """
    # Returns last value sent via setComponentValue (or None)
    return html_component(html, height=H+10, scrolling=False, key="grid_svg")


# Draw component and capture click result
path = st.session_state.solution[0] if st.session_state.solution else None
clicked = render_clickable_svg(path)

# If a cell was clicked, apply tool and rerun to reflect change immediately
if clicked and isinstance(clicked, dict) and "row" in clicked and "col" in clicked:
    apply_tool(int(clicked["row"]), int(clicked["col"]))
    st.experimental_rerun()

st.divider()
if st.session_state.solution is None:
    st.info("Click squares to configure, then press **Solve**.")
else:
    p, total = st.session_state.solution
    if not p:
        st.error("No path found.")
    else:
        st.success(f"Max value: {total} | Path length: {len(p)}")
