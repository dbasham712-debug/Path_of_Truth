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

# === Pixel-perfect grid sizing ===
SQUARE_PX = 70   # cell size in px
GAP_PX    = 6    # gap between cells in px
N = 5            # grid width

st.markdown(f"""
<style>
/* Make every row in the grid a fixed CSS grid instead of flex columns */
#cellgrid [data-testid="stHorizontalBlock"] {{
  display: grid !important;
  grid-template-columns: repeat({{N}}, {{SQUARE_PX}}px) !important;
  grid-auto-rows: {{SQUARE_PX}}px !important;
  gap: {{GAP_PX}}px !important;              /* exact gutter you want */
  justify-content: start !important;         /* pack tightly from left */
  align-items: start !important;
  padding: 0 !important;
  margin: 0 !important;
}}

/* Ensure Streamlit column wrappers don't add size/spacing */
#cellgrid [data-testid="column"] {{
  width: {{SQUARE_PX}}px !important;
  max-width: {{SQUARE_PX}}px !important;
  padding: 0 !important;
  margin: 0 !important;
  flex: 0 0 {{SQUARE_PX}}px !important;
}}

/* True square buttons that fill the grid cells, no extra margins */
#cellgrid [data-testid="stButton"] > button {{
  width: 100% !important;
  height: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  border-radius: 6px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  font-weight: 600 !important;
}}
</style>
""", unsafe_allow_html=True)


# --- session state ---
if "start" not in st.session_state:
    st.session_state.start = (4, 1)
if "end" not in st.session_state:
    st.session_state.end = (0, 3)
if "obstacles" not in st.session_state:
    st.session_state.obstacles = set()
if "cell_values" not in st.session_state:
    st.session_state.cell_values = {}
if "solution" not in st.session_state:
    st.session_state.solution = None
if "tool" not in st.session_state:
    st.session_state.tool = "Start"

st.title("5×5 Weighted Path Solver")
st.caption("Select a tool, click cells to configure, then hit **Solve**.")

# --- toolbar (unchanged) ---
col_tool, col_vals, col_actions = st.columns([1.1, 1.2, 1.2])
with col_tool:
    tool = st.radio(
        "Tool",
        ["Start", "End", "Obstacle", "Low", "Med", "High", "Erase"],
        index=["Start", "End", "Obstacle", "Low", "Med", "High", "Erase"].index(st.session_state.tool),
    )
    st.session_state.tool = tool

with col_vals:
    low_val = st.number_input("Low", min_value=0, max_value=999, value=2, step=1)
    med_val = st.number_input("Med", min_value=0, max_value=999, value=4, step=1)
    high_val = st.number_input("High", min_value=0, max_value=999, value=10, step=1)

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

# --- Path index lookup ---
path_index = {}
if st.session_state.solution:
    path, total = st.session_state.solution
    for i, cell in enumerate(path, start=1):
        path_index[cell] = i

# --- click handler ---
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
        return
    if tool == "Start":
        st.session_state.obstacles.discard(pos)
        if pos == st.session_state.end:
            st.session_state.end = None
        st.session_state.start = pos
        return
    if tool == "End":
        st.session_state.obstacles.discard(pos)
        if pos == st.session_state.start:
            st.session_state.start = None
        st.session_state.end = pos
        return
    if tool in ("Low", "Med", "High"):
        v = {"Low": low_val, "Med": med_val, "High": high_val}[tool]
        st.session_state.cell_values[pos] = int(v)
        return
    if tool == "Erase":
        st.session_state.cell_values.pop(pos, None)
        st.session_state.obstacles.discard(pos)
        return

# --- Grid: use one <form> per cell, laid out by CSS grid ---
st.markdown('<div id="cellgrid">', unsafe_allow_html=True)

for r in range(N):
    for c in range(N):
        pos = (r, c)
        is_start = (pos == st.session_state.start)
        is_end = (pos == st.session_state.end)
        is_ob = (pos in st.session_state.obstacles)
        val = st.session_state.cell_values.get(pos)
        on_path = path_index.get(pos)

        if on_path:
            main_text, color_token = f"{on_path}", ""
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

        with st.form(key=f"cell-{r}-{c}"):
            if st.form_submit_button(label, help=", ".join(help_txt)):
                click_cell(r, c)

st.markdown('</div>', unsafe_allow_html=True)

# --- result display ---
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

