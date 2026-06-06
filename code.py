# ==================================================
# IMPORTS
# ==================================================

import math
import heapq
import time


# ==================================================
# DEFAULT GRID
# ==================================================

GRID = [
    ['S', '.', '.', '.', 'N', '.', '.', '.'],
    ['.', 'W', '.', '.', 'N', '.', '.', '.'],
    ['.', '.', '.', 'W', 'N', '.', 'W', '.'],
    ['N', 'N', '.', '.', '.', '.', 'N', '.'],
    ['.', '.', '.', 'W', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', 'N', 'N', '.'],
    ['.', '.', 'W', '.', '.', '.', '.', 'E'],
    ['.', '.', '.', 'W', '.', 'W', '.', '.']
]


# ==================================================
# INPUT HANDLING
# ==================================================

def read_input_file(filename="inputPS4.txt"):
    """
    Reads configuration parameters from input file.

    Expected format:

    START_NODE: 0,0
    GOAL_NODE: 6,7
    HEURISTIC: h1
    ALGORITHM: A*
    TESTCASE_ID: TC01
    """

    params = {}

    try:
        with open(filename, "r") as f:

            for line in f:

                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Ignore malformed lines
                if ":" not in line:
                    print(
                        f"Warning: Invalid line ignored -> {line}"
                    )
                    continue

                key, value = line.split(":", 1)

                params[key.strip()] = value.strip()

    except FileNotFoundError:

        print(
            f"Warning: '{filename}' not found. "
            f"Using default values."
        )

    return params


def update_grid(grid, start, goal):
    """
    Updates grid with user supplied Start and Goal nodes.
    """

    # Remove existing S and E

    for i in range(len(grid)):
        for j in range(len(grid[0])):

            if grid[i][j] in ("S", "E"):
                grid[i][j] = "."

    # Insert new S and E

    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"

    return grid


# ==================================================
# HEURISTIC FUNCTION (h1)
# ==================================================

def heuristic_h1(node, goal):
    """
    Euclidean Distance Heuristic

    h(n) = sqrt((x_goal - x)^2 + (y_goal - y)^2)
    """

    return math.sqrt(
        (goal[0] - node[0]) ** 2 +
        (goal[1] - node[1]) ** 2
    )


# ==================================================
# SEARCH UTILITIES
# ==================================================

def get_cell_cost(cell):
    """
    Returns traversal cost according to assignment.

    S = 2
    E = 2
    . = 1
    W = 4
    N = Blocked
    """

    if cell == 'S':
        return 2

    elif cell == 'E':
        return 2

    elif cell == '.':
        return 1

    elif cell == 'W':
        return 4

    else:
        return float('inf')


def get_neighbors(position, grid):
    """
    Generates valid neighbors.

    Movement Priority:
    North -> East -> South -> West

    No-Fly Zones (N) are ignored.
    """

    row, col = position

    directions = [
        (-1, 0),  # North
        (0, 1),   # East
        (1, 0),   # South
        (0, -1)   # West
    ]

    neighbors = []

    for dr, dc in directions:

        nr = row + dr
        nc = col + dc

        if (
            0 <= nr < len(grid)
            and
            0 <= nc < len(grid[0])
            and
            grid[nr][nc] != 'N'
        ):
            neighbors.append((nr, nc))

    return neighbors


def reconstruct_path(parent, goal):
    """
    Reconstructs optimal path from Goal back to Start.

    Example:

    Goal <- A <- B <- Start

    becomes

    Start -> B -> A -> Goal
    """

    path = []

    current = goal

    while current is not None:

        path.append(current)

        current = parent[current]

    path.reverse()

    return path


# ==================================================
# A* SEARCH USING HEURISTIC h1
# ==================================================

def astar_h1(grid, start, goal):
    """
    A* Search

    f(n) = g(n) + h(n)

    g(n) = cumulative path cost
    h(n) = Euclidean distance
    """

    start_time = time.perf_counter()

    open_list = []

    heapq.heappush(
        open_list,
        (
            heuristic_h1(start, goal),
            start
        )
    )

    parent = {
        start: None
    }

    # Cost from start node to current node, start node cost is given as 2

    g_cost = {
        start: get_cell_cost('S')
    }

    explored = []

    closed_set = set()

    nodes_expanded = 0

    while open_list:

        current_f, current = heapq.heappop(
            open_list
        )

        # Skip already expanded nodes

        if current in closed_set:
            continue

        closed_set.add(current)

        explored.append(current)

        nodes_expanded += 1

        # Goal Test

        if current == goal:

            runtime = (
                time.perf_counter()
                - start_time
            ) * 1000

            path = reconstruct_path(
                parent,
                goal
            )

            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime_ms": runtime,
                "explored": explored,
                "path_length": len(path) - 1,
                "path_cost": g_cost[goal]
            }

        # Explore neighbors

        for neighbor in get_neighbors(
            current,
            grid
        ):

            row, col = neighbor

            cell_cost = get_cell_cost(
                grid[row][col]
            )

            tentative_g = (
                g_cost[current]
                + cell_cost
            )

            if (
                neighbor not in g_cost
                or
                tentative_g < g_cost[neighbor]
            ):

                g_cost[neighbor] = tentative_g

                parent[neighbor] = current

                h = heuristic_h1(
                    neighbor,
                    goal
                )

                f = tentative_g + h

                heapq.heappush(
                    open_list,
                    (
                        f,
                        neighbor
                    )
                )

    # Goal not reachable

    return None

def write_output(result, filename="outputPS4.txt"):

    with open(filename, "w") as f:

        f.write("===== A* RESULTS =====\n\n")

        f.write(
            f"Nodes Expanded: "
            f"{result['nodes_expanded']}\n"
        )

        f.write(
            f"Runtime (ms): "
            f"{round(result['runtime_ms'],4)}\n"
        )

        f.write(
            f"Path Length: "
            f"{result['path_length']}\n"
        )

        f.write(
            f"Path Cost: "
            f"{result['path_cost']}\n"
        )

        f.write(
            f"Path: "
            f"{result['path']}\n"
        )

        f.write(
            f"Explored Nodes: "
            f"{result['explored']}\n"
        )

# ==================================================
# HEURISTIC FUNCTION (h2)
# ==================================================

def heuristic_h2(node, goal, grid):
    """
    Bounding Box Risk Weighted Heuristic

    h2 = Manhattan Distance × Average Window Cost
    """

    cost_map = {
        '.': 1,
        'W': 4,
        'N': 8,
        'S': 2,
        'E': 2
    }

    row, col = node
    goal_row, goal_col = goal

    manhattan_distance = (
        abs(row - goal_row)
        + abs(col - goal_col)
    )

    row_start = min(row, goal_row)
    row_end = max(row, goal_row)

    col_start = min(col, goal_col)
    col_end = max(col, goal_col)

    total_cost = 0
    total_cells = 0

    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):

            total_cost += cost_map[
                grid[r][c]
            ]

            total_cells += 1

    average_cost = total_cost / total_cells

    return round(
        manhattan_distance * average_cost,
        2
    )


# ==================================================
# GBFS-H2
# ==================================================

def gbfs_h2(grid, start, goal):

    start_time = time.perf_counter()

    open_list = []

    heapq.heappush(
        open_list,
        (
            heuristic_h2(
                start,
                goal,
                grid
            ),
            start
        )
    )

    parent = {start: None}

    visited = set()

    explored = []

    heuristic_values = {}

    nodes_expanded = 0

    while open_list:

        h, current = heapq.heappop(
            open_list
        )

        if current in visited:
            continue

        visited.add(current)

        explored.append(current)

        heuristic_values[current] = h

        nodes_expanded += 1

        if current == goal:

            runtime = (
                time.perf_counter()
                - start_time
            ) * 1000

            path = reconstruct_path(
                parent,
                goal
            )

            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime_ms": runtime,
                "path_length": len(path) - 1,
                "explored": explored,
                "heuristic_values": heuristic_values,
            }

        for neighbor in get_neighbors(
            current,
            grid
        ):

            if neighbor not in visited:

                parent[neighbor] = current

                heapq.heappush(
                    open_list,
                    (
                        heuristic_h2(
                            neighbor,
                            goal,
                            grid
                        ),
                        neighbor
                    )
                )

    return None

# ==================================================
# MAIN DRIVER
# ==================================================

if __name__ == "__main__":

    config = read_input_file()

    DEFAULTS = {
        "START_NODE": "0,0",
        "GOAL_NODE": "6,7",
        "HEURISTIC": "h1",
        "ALGORITHM": "GBFS",
        "TESTCASE_ID": "TC01"
    }

    # Apply defaults if parameters are missing

    for key, value in DEFAULTS.items():

        if key not in config:

            print(
                f"Warning: '{key}' missing. "
                f"Using default value '{value}'."
            )

    start = tuple(
        map(
            int,
            config.get(
                "START_NODE",
                DEFAULTS["START_NODE"]
            ).split(",")
        )
    )

    goal = tuple(
        map(
            int,
            config.get(
                "GOAL_NODE",
                DEFAULTS["GOAL_NODE"]
            ).split(",")
        )
    )

    heuristic = config.get(
        "HEURISTIC",
        DEFAULTS["HEURISTIC"]
    )

    algorithm = config.get(
        "ALGORITHM",
        DEFAULTS["ALGORITHM"]
    )

    testcase_id = config.get(
        "TESTCASE_ID",
        DEFAULTS["TESTCASE_ID"]
    )

    # Validate input values

    if heuristic not in ["h1", "h2"]:

        print(
            f"Warning: Invalid heuristic '{heuristic}'. "
            f"Using 'h1'."
        )

        heuristic = "h1"

    if algorithm not in ["GBFS", "A*"]:

        print(
            f"Warning: Invalid algorithm '{algorithm}'. "
            f"Using 'GBFS'."
        )

        algorithm = "GBFS"

    # Create working copy of grid

    grid = [row[:] for row in GRID]

    grid = update_grid(
        grid,
        start,
        goal
    )

    # Display configuration

    print("\n===== CONFIGURATION =====")

    print("Start Node :", start)
    print("Goal Node  :", goal)
    print("Heuristic  :", heuristic)
    print("Algorithm  :", algorithm)
    print("Test Case  :", testcase_id)

    # Display grid

    print("\n===== UPDATED GRID =====")

    for row in grid:
        print(" ".join(row))

    # Run A*

    if algorithm == "A*" and heuristic == "h1":

        result = astar_h1(
            grid,
            start,
            goal
        )

        if result:
            write_output(result)

            print(
                "\nResults written to outputPS4.txt"
            )

            print("\n===== A* RESULTS =====")

            print(
                "Nodes Expanded:",
                result["nodes_expanded"]
            )

            print(
                "Runtime (ms):",
                round(
                    result["runtime_ms"],
                    4
                )
            )

            print(
                "Path Length:",
                result["path_length"]
            )

            print(
                "Path Cost:",
                result["path_cost"]
            )

            print(
                "Path:",
                result["path"]
            )

        else:

            print(
                "Goal not reachable."
            )
    # Run GBFS-H2

    elif algorithm == "GBFS" and heuristic == "h2":

        result = gbfs_h2(
            grid,
            start,
            goal
        )

        if result:

            print("\n===== GBFS-H2 RESULTS =====")

            print(
                "Nodes Expanded:",
                result["nodes_expanded"]
            )

            print(
                "Runtime (ms):",
                round(
                    result["runtime_ms"],
                    4
                )
            )

            print(
                "Path Length:",
                result["path_length"]
            )

            print(
                "Path:",
                result["path"]
            )

        else:

            print(
                "Goal not reachable."
            )
