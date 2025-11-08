# streamlit app: 5x5 Weighted Path Solver (click-to-setup)

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st

GridPos = Tuple[int, int]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N = 5  # grid size


# ------------------------
# Core grid/solver helpers
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
    visited: Set[GridPos] = set([start])

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


# --------------
# Streamlit UI
# --------------
st.set_page_config(page_title="5x5 Weighted Path Solver", page_icon="🧩", layout="centered")

# --- square buttons (CSS) ---
st.markdown("""
<style>
div[data-testid="stButton"] > button {
  width: 100% !important;
  aspect-ratio: 1 / 1 !important;
  height: auto !important;
  min-height: 0 !important;
  padding: 0 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
</style>
""", unsafe_allow_html=True)

# Init session state
if "start" not in st.session_state:
    st.session_state.start: GridPos = (4, 1)
if "end" not in st.session_state:
    st.session_state.end: GridPos = (0, 3)
if "obstacles" not in st.session_state:
    st.session_state.obstacles: Set[GridPos] = set()
if "cell_values" not in st.session_state:
    st.session_state.cell_values: Dict[GridPos, int] = {}
if "solution" not in st.session_state:
    st.session_state.solution = None
if "tool" not in st.session_state:
    st.session_state.tool = "Start"

st.title("5×5 Weighted Path Solver")
st.caption("Select a tool, click cells to configure, then hit **Solve**.")

# Toolbar
col_tool, col_vals, col_actions = st.columns([1.1, 1.2, 1.2])
with col_tool:
    tool = st.radio(
    "Tool",
    ["Start", "End", "Obstacle", "Low", "Med", "High", "Mythical", "Erase"],
    index=["Start", "End", "Obstacle", "Low", "Med", "High", "Mythical", "Erase"].index(st.session_state.tool),
    )
    st.session_state.tool = tool

with col_vals:
    low_val  = st.number_input("Low",  min_value=0, max_value=999, value=2,  step=1)
    med_val  = st.number_input("Med",  min_value=0, max_value=999, value=4,  step=1)
    high_val = st.number_input("High", min_value=0, max_value=999, value=10,  step=1)
    myth_val = st.number_input("Mythical", min_value=0, max_value=999, value=30, step=1)


with col_actions:
    if st.button("Solve", use_container_width=True):
        s, e = st.session_state.start, st.session_state.end
        res = solve_max_value_path(N, s, e, st.session_state.obstacles, st.session_state.cell_values)
        st.session_state.solution = res
    c1, c2, c3 = st.columns(3)
    if c1.button("Clear Values"):
        st.session_state.cell_values = {}
        st.session_state.solution = None
    if c2.button("Clear Obstacles"):
        st.session_state.obstacles = set()
        st.session_state.solution = None
    if c3.button("Clear All (keep S/E)"):
        st.session_state.obstacles = set()
        st.session_state.cell_values = {}
        st.session_state.solution = None

st.divider()

# Path index lookup
path_index = {}
if st.session_state.solution:
    path, total = st.session_state.solution
    for i, cell in enumerate(path, start=1):
        path_index[cell] = i

# Click handler
def click_cell(r: int, c: int):
    st.session_state.solution = None
    pos = (r, c)
    tool = st.session_state.tool

    if tool == "Obstacle":
        if pos not in (st.session_state.start, st.session_state.end):
            if pos in st.session_state.obstacles:
                st.session_state.obstacles.remove(pos)
            else:
                st.session_state.obstacles.add(pos)
        st.rerun()

    elif tool == "Start":
        st.session_state.obstacles.discard(pos)
        if pos == st.session_state.end:
            st.session_state.end = None
        st.session_state.start = pos
        st.rerun()

    elif tool == "End":
        st.session_state.obstacles.discard(pos)
        if pos == st.session_state.start:
            st.session_state.start = None
        st.session_state.end = pos
        st.rerun()

    elif tool in ("Low", "Med", "High", "Mythical"):
        v = {"Low": low_val, "Med": med_val, "High": high_val, "Mythical": myth_val}[tool]
        st.session_state.cell_values[pos] = int(v)
        st.rerun()


    elif tool == "Erase":
        st.session_state.cell_values.pop(pos, None)
        st.session_state.obstacles.discard(pos)
        st.rerun()


# Grid
for r in range(N):
    cols = st.columns(N, gap="small")
    for c in range(N):
        pos = (r, c)
        is_start = (pos == st.session_state.start)
        is_end = (pos == st.session_state.end)
        is_ob = (pos in st.session_state.obstacles)
        val = st.session_state.cell_values.get(pos)
        on_path = path_index.get(pos)

        # Color-coded label
        if on_path:
            main_text = f"{on_path}"
            color_token = ""
        elif is_ob:
            main_text, color_token = "X", "🟥"
        elif is_start:
            main_text, color_token = "S", "🟦"
        elif is_end:
            main_text, color_token = "E", "🟪"
        else:
            if val == low_val:
                color_token = "🟩"
            elif val == med_val:
                color_token = "🟦"
            elif val == high_val:
                color_token = "🟨"
            elif val == myth_val:
                color_token = "🟪"  
            else:
                color_token = "⬜"
            main_text = str(val if val is not None else 1)

        label = f"{color_token} {main_text}".strip()
        help_txt = []
        if is_start: help_txt.append("Start")
        if is_end: help_txt.append("End")
        if is_ob: help_txt.append("Obstacle")
        if val is not None: help_txt.append(f"Value={val}")
        else: help_txt.append("Value=1")
        if on_path: help_txt.append(f"Path idx={on_path}")

        if cols[c].button(label, key=f"cell-{r}-{c}", help=", ".join(help_txt), use_container_width=True):
            click_cell(r, c)
# --- SVG arrows for the solved path (place this right after the grid loop) ---
if st.session_state.solution:
    path, total = st.session_state.solution
    if path:
        # SVG canvas settings
        cell_px = 60       # pixel size per cell (only affects the SVG, not buttons)
        gap_px = 6         # gap to visually match st.columns gap a bit
        W = N * (cell_px + gap_px) - gap_px
        H = W

        def center_of(rc):
            r, c = rc
            x = c * (cell_px + gap_px) + cell_px / 2
            y = r * (cell_px + gap_px) + cell_px / 2
            return x, y

        # Build SVG paths between consecutive cells
        lines = []
        for i in range(len(path) - 1):
            x1, y1 = center_of(path[i])
            x2, y2 = center_of(path[i + 1])
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                         f'stroke="black" stroke-width="3" marker-end="url(#arrow)" />')

        cell_rects = []
        for r in range(N):
            for c in range(N):
                pos = (r, c)
                if pos in st.session_state.obstacles:
                    color = "red"
                elif pos == st.session_state.start:
                    color = "deepskyblue"
                elif pos == st.session_state.end:
                    color = "purple"
                elif st.session_state.cell_values.get(pos) == low_val:
                    color = "limegreen"
                elif st.session_state.cell_values.get(pos) == med_val:
                    color = "dodgerblue"
                elif st.session_state.cell_values.get(pos) == high_val:
                    color = "gold"
                elif st.session_state.cell_values.get(pos) == myth_val:
                    color = "purple"
                else:
                    color = "white"


                cell_rects.append(
                    f'<rect x="{c * (cell_px + gap_px)}" y="{r * (cell_px + gap_px)}" '
                    f'width="{cell_px}" height="{cell_px}" fill="{color}" stroke="gray" />'
                )

        svg = f"""
        <svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="black" />
            </marker>
          </defs>
          <!-- cells -->
          {''.join(cell_rects)}
          <!-- path with arrows -->
          {''.join(lines)}
        </svg>
        """
        st.markdown(svg, unsafe_allow_html=True)

st.divider()
if st.session_state.solution is None:
    st.info("Click cells to configure, then press **Solve**.")
else:
    path, total = st.session_state.solution
    if not path:
        st.error("No path found.")
    else:
        st.success(f"Max value: {total} | Path length: {len(path)}")

with st.expander("Show configuration"):
    st.write(f"Start: {st.session_state.start}")
    st.write(f"End: {st.session_state.end}")
    st.write(f"Obstacles: {sorted(list(st.session_state.obstacles))}")
    st.json({str(k): v for k, v in st.session_state.cell_values.items()})


