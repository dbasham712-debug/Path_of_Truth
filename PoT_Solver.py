# streamlit app: 5x5 Weighted Path Solver (Pure Streamlit SVG version)

from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import streamlit as st

GridPos = Tuple[int, int]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N = 5  # grid size


# ===================== Core Solver =====================
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


# ===================== Streamlit UI =====================
st.set_page_config(page_title="5x5 Weighted Path Solver", page_icon="🧩", layout="centered")

# --- Cell states ---
COLORS = {
    "empty": "white",
    "low": "#32CD32",
    "med": "#1E90FF",
    "high": "#FFD700",
    "obs": "#FF4D4D",
    "start": "#00BFFF",
    "end": "#800080",
}

TOOLS = ["Start", "End", "Obstacle", "Low", "Med", "High", "Erase"]

# --- Session state ---
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

# --- Toolbar ---
col_tool, col_actions = st.columns([1.5, 1])
with col_tool:
    st.session_state.tool = st.radio("Tool", TOOLS, index=TOOLS.index(st.session_state.tool), horizontal=True)
with col_actions:
    if st.button("Solve", use_container_width=True):
        obs = set()
        values = {}
        for r in range(N):
            for c in range(N):
                state = st.session_state.grid[r][c]
                if state == "obs":
                    obs.add((r, c))
                elif state == "low":
                    values[(r, c)] = 1
                elif state == "med":
                    values[(r, c)] = 3
                elif state == "high":
                    values[(r, c)] = 5
        s, e = st.session_state.start, st.session_state.end
        st.session_state.solution = solve_max_value_path(N, s, e, obs, values)
    if st.button("Clear All"):
        st.session_state.grid = [["empty" for _ in range(N)] for _ in range(N)]
        st.session_state.solution = None
        st.session_state.grid[st.session_state.start[0]][st.session_state.start[1]] = "start"
        st.session_state.grid[st.session_state.end[0]][st.session_state.end[1]] = "end"

st.divider()


# --- Click handling ---
def handle_click(r, c):
    st.session_state.solution = None
    tool = st.session_state.tool
    g = st.session_state.grid
    if tool == "Erase":
        if g[r][c] not in ("start", "end"):
            g[r][c] = "empty"
    elif tool == "Obstacle":
        if g[r][c] != "start" and g[r][c] != "end":
            g[r][c] = "obs" if g[r][c] != "obs" else "empty"
    elif tool in ("Low", "Med", "High"):
        if g[r][c] not in ("start", "end"):
            g[r][c] = tool.lower()
    elif tool == "Start":
        oldr, oldc = st.session_state.start
        g[oldr][oldc] = "empty"
        g[r][c] = "start"
        st.session_state.start = (r, c)
    elif tool == "End":
        oldr, oldc = st.session_state.end
        g[oldr][oldc] = "empty"
        g[r][c] = "end"
        st.session_state.end = (r, c)


# --- Render SVG grid ---
CELL = 70
GAP = 6
W = N * (CELL + GAP) - GAP
H = W

def center_of(rc):
    r, c = rc
    x = c * (CELL + GAP) + CELL / 2
    y = r * (CELL + GAP) + CELL / 2
    return x, y

def make_svg(path=None):
    rects = []
    for r in range(N):
        for c in range(N):
            color = COLORS[st.session_state.grid[r][c]]
            rects.append(f'<rect x="{c*(CELL+GAP)}" y="{r*(CELL+GAP)}" width="{CELL}" height="{CELL}" '
                         f'stroke="gray" fill="{color}" rx="6" ry="6" onclick="sendClick({r},{c})" />')

    lines = []
    if path:
        for i in range(len(path)-1):
            x1, y1 = center_of(path[i])
            x2, y2 = center_of(path[i+1])
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                         f'stroke="black" stroke-width="3" marker-end="url(#arrow)" />')

    svg = f"""
    <svg id="gridSVG" width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="black" />
        </marker>
        <script type="text/ecmascript"><![CDATA[
            function sendClick(r,c){{
                const streamlitEvent = new CustomEvent("cellclick", {{detail: {{row:r,col:c}}}});
                window.parent.postMessage({{isStreamlitMessage:true,type:"cellClick",row:r,col:c}}, "*");
            }}
        ]]></script>
      </defs>
      {''.join(rects)}
      {''.join(lines)}
    </svg>
    """
    return svg

# --- Handle simulated click events from front-end ---
clicked = st.session_state.get("clicked_cell")
if clicked:
    r, c = clicked
    handle_click(r, c)
    st.session_state.clicked_cell = None

# --- Draw the SVG ---
path = None
if st.session_state.solution:
    path, total = st.session_state.solution
st.markdown(make_svg(path), unsafe_allow_html=True)

# --- Inject a listener to capture clicks ---
st.markdown("""
<script>
window.addEventListener("message", (e) => {
    if (e.data && e.data.type === "cellClick") {
        const index = e.data.row + "," + e.data.col;
        window.parent.postMessage({isStreamlitMessage:true, type:"streamlit:setComponentValue", key:"clicked_cell", value:[e.data.row, e.data.col]}, "*");
        window.parent.postMessage({isStreamlitMessage:true, type:"streamlit:rerun"}, "*");
    }
});
</script>
""", unsafe_allow_html=True)

st.divider()
if st.session_state.solution is None:
    st.info("Click cells to configure, then press **Solve**.")
else:
    path, total = st.session_state.solution
    if not path:
        st.error("No path found.")
    else:
        st.success(f"Max value: {total} | Path length: {len(path)}")
