# ==================================================
# IMPORTS
# ==================================================

import math
import heapq
import time
import sys


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
# HEURISTIC FUNCTIONS
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


def heuristic_h2(node, goal, grid):
    """
    Bounding Box Risk Weighted Heuristic

    h2 = Manhattan Distance × Average Window Cost
    
    Professor clarification: 'N' cells contribute risk score 8
    inside the bounding-box average even though they are not traversable.
    """

    cost_map = {
        '.': 1,
        'W': 4,
        'N': 8,  # Risk/no-fly weighting
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


def calculate_path_cost(path, grid):
    """
    Calculates total traversal cost of a path.
    
    Sums up the cost of each cell in the path.
    """
    
    cost = 0
    
    for r, c in path:
        cost += get_cell_cost(grid[r][c])
    
    return cost


def calculate_penalty(path, grid):
    """
    Calculates penalty for encountering weather hazard (W) cells.
    
    Each 'W' cell in the path contributes 4 to the penalty.
    """
    
    penalty = 0
    
    for r, c in path:
        if grid[r][c] == 'W':
            penalty += 4
    
    return penalty


def display_iteration_grid(grid, current, frontier, explored):
    """
    Visualizes the grid for the current search iteration.
    """
    grid_copy = [row[:] for row in grid]

    for r, c in explored:
        if grid_copy[r][c] not in ('S', 'E', 'N', 'W'):
            grid_copy[r][c] = 'o'

    for r, c in frontier:
        if grid_copy[r][c] not in ('S', 'E', 'N', 'W'):
            grid_copy[r][c] = 'f'

    if grid_copy[current[0]][current[1]] not in ('S', 'E', 'N', 'W'):
        grid_copy[current[0]][current[1]] = 'X'

    print("\n===== ITERATION GRID =====")
    for row in grid_copy:
        print(" ".join(row))
    print(f"Selected Node: {current}")
    print(f"Frontier Nodes: {frontier}")
    print(f"Explored Nodes: {explored}")


def count_weather_hazards(path, grid):
    """
    Counts the number of weather hazard cells crossed.
    """
    
    count = 0
    
    for r, c in path:
        if grid[r][c] == 'W':
            count += 1
    
    return count


def count_no_fly_zones(path, grid):
    """
    Counts the number of no-fly zone cells crossed.
    """
    
    count = 0
    
    for r, c in path:
        if grid[r][c] == 'N':
            count += 1
    
    return count


def display_path_grid(grid, path):
    """
    Prints the grid showing the path with '*' markers.
    """
    grid_copy = [row[:] for row in grid]

    for r, c in path:
        if grid_copy[r][c] not in ('S', 'E'):
            grid_copy[r][c] = '*'

    print("\n===== PATH GRID =====")
    for row in grid_copy:
        print(" ".join(row))


# ==================================================
# GBFS WITH HEURISTIC h1
# ==================================================

def gbfs_h1(grid, start, goal):
    """
    Greedy Best First Search using Euclidean Distance heuristic.
    """

    start_time = time.perf_counter()

    open_list = []
    tie_break = 0
    heapq.heappush(open_list, (heuristic_h1(start, goal), tie_break, start))
    tie_break += 1

    parent = {start: None}
    visited = set()
    explored = []
    heuristic_values = {}
    nodes_expanded = 0
    final_frontier = []
    frontier_history = []
    selected_nodes = []
    memory_usage = 0
    frontier_seen = set()
    trap_logs = []

    while open_list:

        h, _, current = heapq.heappop(open_list)
        selected_nodes.append(current)

        if current in visited:
            continue

        visited.add(current)
        explored.append(current)

        heuristic_values[current] = h
        nodes_expanded += 1
        memory_usage = max(memory_usage, len(open_list) + len(visited))

        # Goal check
        if current == goal:

            runtime = (time.perf_counter() - start_time) * 1000

            path = reconstruct_path(parent, goal)
            path_cost = calculate_path_cost(path, grid)
            weather_count = count_weather_hazards(path, grid)
            penalty = calculate_penalty(path, grid)
            no_fly_zones_crossed = count_no_fly_zones(path, grid)
            final_frontier = [node for _, _, node in open_list]

            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime_ms": runtime,
                "path_length": len(path) - 1,
                "path_cost": path_cost,
                "explored": explored,
                "heuristic_values": heuristic_values,
                "penalty": penalty,
                "weather_count": weather_count,
                "no_fly_zones_crossed": no_fly_zones_crossed,
                "memory_usage": memory_usage,
                "frontier": final_frontier,
                "frontier_history": frontier_history,
                "selected_nodes": selected_nodes,
                "trap_logs": trap_logs
            }

        for neighbor in get_neighbors(current, grid):

            if neighbor not in visited:
                # Only set parent if not already set (avoid overwrite)
                if neighbor not in parent:
                    parent[neighbor] = current

                    heapq.heappush(
                        open_list,
                        (heuristic_h1(neighbor, goal), tie_break, neighbor)
                    )
                    tie_break += 1
                    frontier_history.append(
                        [node for _, _, node in open_list]
                    )
                    memory_usage = max(memory_usage, len(open_list) + len(visited))

        # Visualize current iteration and trap detection
        current_frontier = [node for _, _, node in open_list]
        display_iteration_grid(grid, current, current_frontier, explored)
        frontier_set = frozenset(current_frontier)
        if frontier_set in frontier_seen:
            trap_logs.append(
                f"Iteration {nodes_expanded}: repeated frontier set after selecting {current}"
            )
        else:
            frontier_seen.add(frontier_set)

    return None

# ==================================================
# GBFS WITH HEURISTIC h2
# ==================================================

def gbfs_h2(grid, start, goal):
    """
    Greedy Best First Search using Bounding Box Risk Weighted heuristic.
    """

    start_time = time.perf_counter()

    open_list = []
    tie_break = 0

    heapq.heappush(
        open_list,
        (
            heuristic_h2(
                start,
                goal,
                grid
            ),
            tie_break,
            start
        )
    )
    tie_break += 1

    parent = {start: None}

    visited = set()

    explored = []

    heuristic_values = {}
    frontier_seen = set()
    trap_logs = []

    nodes_expanded = 0
    final_frontier = []
    frontier_history = []
    selected_nodes = []
    memory_usage = 0

    while open_list:

        h, _, current = heapq.heappop(
            open_list
        )
        selected_nodes.append(current)

        if current in visited:
            continue

        visited.add(current)

        explored.append(current)

        heuristic_values[current] = h

        nodes_expanded += 1
        memory_usage = max(memory_usage, len(open_list) + len(visited))

        if current == goal:

            runtime = (
                time.perf_counter()
                - start_time
            ) * 1000

            path = reconstruct_path(
                parent,
                goal
            )
            
            path_cost = calculate_path_cost(path, grid)
            weather_count = count_weather_hazards(path, grid)
            penalty = calculate_penalty(path, grid)
            no_fly_zones_crossed = count_no_fly_zones(path, grid)
            final_frontier = [node for _, _, node in open_list]

            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime_ms": runtime,
                "path_length": len(path) - 1,
                "path_cost": path_cost,
                "explored": explored,
                "heuristic_values": heuristic_values,
                "penalty": penalty,
                "weather_count": weather_count,
                "no_fly_zones_crossed": no_fly_zones_crossed,
                "memory_usage": memory_usage,
                "frontier": final_frontier,
                "frontier_history": frontier_history,
                "selected_nodes": selected_nodes,
                "trap_logs": trap_logs
            }

        for neighbor in get_neighbors(
            current,
            grid
        ):

            if neighbor not in visited:
                # Only set parent if not already set (avoid overwrite)
                if neighbor not in parent:
                    parent[neighbor] = current

                    heapq.heappush(
                        open_list,
                        (
                            heuristic_h2(
                                neighbor,
                                goal,
                                grid
                            ),
                            tie_break,
                            neighbor
                        )
                    )
                    tie_break += 1
                    frontier_history.append(
                        [node for _, _, node in open_list]
                    )
                    memory_usage = max(memory_usage, len(open_list) + len(visited))

        current_frontier = [node for _, _, node in open_list]
        display_iteration_grid(grid, current, current_frontier, explored)
        frontier_set = frozenset(current_frontier)
        if frontier_set in frontier_seen:
            trap_logs.append(
                f"Iteration {nodes_expanded}: repeated frontier set after selecting {current}"
            )
        else:
            frontier_seen.add(frontier_set)

    return None

# ==================================================
# A* SEARCH USING HEURISTIC h1
# ==================================================

def astar_h1(grid, start, goal):
    """
    A* Search with Euclidean Distance heuristic

    f(n) = g(n) + h(n)

    g(n) = cumulative path cost
    h(n) = Euclidean distance
    """

    start_time = time.perf_counter()

    open_list = []
    tie_break = 0

    heapq.heappush(
        open_list,
        (
            heuristic_h1(start, goal),
            tie_break,
            start
        )
    )
    tie_break += 1

    parent = {
        start: None
    }
    frontier_seen = set()
    trap_logs = []

    # Cost from start node to current node, start node cost is given as 2

    g_cost = {
        start: get_cell_cost('S')
    }

    explored = []

    closed_set = set()

    nodes_expanded = 0
    final_frontier = []
    frontier_history = []
    selected_nodes = []
    memory_usage = 0

    while open_list:

        current_f, _, current = heapq.heappop(
            open_list
        )
        selected_nodes.append(current)

        # Skip already expanded nodes

        if current in closed_set:
            continue

        closed_set.add(current)

        explored.append(current)

        nodes_expanded += 1
        memory_usage = max(memory_usage, len(open_list) + len(closed_set))

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
            
            path_cost = g_cost[goal]
            weather_count = count_weather_hazards(path, grid)
            penalty = calculate_penalty(path, grid)
            no_fly_zones_crossed = count_no_fly_zones(path, grid)
            final_frontier = [node for _, _, node in open_list]

            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime_ms": runtime,
                "path_length": len(path) - 1,
                "path_cost": path_cost,
                "explored": explored,
                "heuristic_values": {},
                "penalty": penalty,
                "weather_count": weather_count,
                "no_fly_zones_crossed": no_fly_zones_crossed,
                "memory_usage": memory_usage,
                "frontier": final_frontier,
                "frontier_history": frontier_history,
                "selected_nodes": selected_nodes,
                "trap_logs": trap_logs
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
                        tie_break,
                        neighbor
                    )
                )
                tie_break += 1
                frontier_history.append(
                    [node for _, _, node in open_list]
                )
                memory_usage = max(memory_usage, len(open_list) + len(closed_set))

        current_frontier = [node for _, _, node in open_list]
        display_iteration_grid(grid, current, current_frontier, explored)
        frontier_set = frozenset(current_frontier)
        if frontier_set in frontier_seen:
            trap_logs.append(
                f"Iteration {nodes_expanded}: repeated frontier set after selecting {current}"
            )
        else:
            frontier_seen.add(frontier_set)

    # Goal not reachable

    return None
## ==================================================
## PERFORMANCE WRAPPER
## ==================================================

def run_algorithm(algorithm_func, grid, start, goal, algo_name):
    """
    Wrapper to capture execution time, nodes expanded and completeness.
    """
    start_time = time.perf_counter()

    try:
        result = algorithm_func(grid, start, goal)

        end_time = time.perf_counter()

        # Assumption: result contains path and nodes expanded
        #path = result.get("path", [])
        nodes_expanded = result.get("nodes_expanded", 0)

        return {
            "Algorithm": algo_name,
            "Completed": "Yes (Optimal & Complete)" if algo_name.startswith("A*") else "No (Heuristic-based, Not Guaranteed)",
            "Nodes Expanded": nodes_expanded,
            "Time Taken (ms)": round((end_time - start_time) * 1000, 6)
        }

    except Exception as e:
        return {
            "Algorithm": algo_name,
            "Completed": False,
            "Nodes Expanded": 0,
            "Time Taken (ms)": 0
        }
## ==================================================
## ALGORITHM COMPARISON
## ==================================================

def compare_algorithms(grid, start, goal):
    """
    Runs GBFS (h1), GBFS (h2), and A* (h1) and compares them.
    """

    results = []

    results.append(run_algorithm(gbfs_h1, grid, start, goal, "GBFS (h1)"))
    results.append(run_algorithm(gbfs_h2, grid, start, goal, "GBFS (h2)"))
    results.append(run_algorithm(astar_h1, grid, start, goal, "A* (h1)"))

    return results

## ==================================================
## FORMAT OUTPUT TABLE
## ==================================================

def format_results_table(results):

    results = results if isinstance(results, list) else [results]

    header = f"{'Algorithm':<15}{'Complete':<40}{'Nodes Expanded':<18}{'Time (ms)':<12}\n"
    header += "-" * 85 + "\n"

    rows = ""
    for res in results:
        rows += f"{res.get('Algorithm','N/A'):<15}{res.get('Completed','N/A'):<40}{res.get('Nodes Expanded','N/A'):<18}{res.get('Time Taken (ms)','N/A'):<12}\n"

    return header + rows

## ==================================================
## ANALYSIS SUMMARY
## ==================================================

def generate_analysis(results):

    results = results if isinstance(results, list) else [results]

    analysis = "\n\n======== ANALYSIS ========\n"

    # a. Completeness
    complete_algos = [r.get("Algorithm") for r in results if "Yes" in str(r.get("Completed"))]

    if complete_algos:
        analysis += "\n(a) Complete Algorithm:\n"
        for algo in complete_algos:
            analysis += f"    {algo} is complete as it guarantees reaching the goal using cost + heuristic.\n"
    else:
        analysis += "\n(a) No algorithm guaranteed completeness\n"

    # b. Nodes Expanded
    min_nodes = min(results, key=lambda x: x.get("Nodes Expanded", float('inf')))
    analysis += f"\n(b) Least Nodes Expanded: {min_nodes.get('Algorithm','N/A')}\n"

    # c. Fastest
    fastest = min(results, key=lambda x: x.get("Time Taken (ms)", float('inf')))
    analysis += f"(c) Fastest Algorithm: {fastest.get('Algorithm','N/A')}\n"

    return analysis

# ==================================================
# OUTPUT WRITING
# ==================================================

def write_output(results, filename="outputPS4.txt"):
    """
    Writes search results to output file.
    """

    with open(filename, "w") as f:
        
        # Determine algorithm type
        if "heuristic_values" in result and result["heuristic_values"]:
            f.write("===== GBFS RESULTS =====\n\n")
        else:
            f.write("===== A* RESULTS =====\n\n")

        f.write(
            f"Nodes Expanded: "
            f"{result['nodes_expanded']}\n"
        )

        f.write(
            f"Runtime (ms): "
            f"{round(result['runtime_ms'], 4)}\n"
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
            f"Memory Usage: "
            f"{result['memory_usage']}\n"
        )

        f.write(
            f"Weather Hazard Zones Crossed: "
            f"{result['weather_count']}\n"
        )

        f.write(
            f"Penalty Incurred: "
            f"{result['penalty']}\n"
        )

        f.write(
            f"No-Fly Zones Crossed: "
            f"{result['no_fly_zones_crossed']}\n"
        )

        if result.get("heuristic_values"):
            f.write(
                f"Heuristic Values: "
                f"{result['heuristic_values']}\n"
            )

        if result.get("frontier_history"):
            f.write(
                f"Frontier History: "
                f"{result['frontier_history']}\n"
            )

        f.write(
            f"Selected Nodes: "
            f"{result['selected_nodes']}\n"
        )

        if result.get("frontier"):
            f.write(
                f"Frontier: "
                f"{result['frontier']}\n"
            )

        if result.get("trap_logs"):
            f.write(
                f"Trap Logs: "
                f"{result['trap_logs']}\n"
            )

        f.write(
            f"Path: "
            f"{result['path']}\n"
        )

        f.write(
            f"Explored Nodes: "
            f"{result['explored']}\n"
        )

        f.write(
            "Time Complexity: O(V log V)\n"
        )

        f.write(
            "Space Complexity: O(V)\n\n"
        )
     

        output_text = ""

        output_text += "=== Algorithm Comparison Results ===\n\n"
        output_text += format_results_table(results)
        output_text += generate_analysis(results)

    
        f.write(output_text)

    print("\n" + output_text)



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

    # Validate start and goal coordinates

    if not (
        0 <= start[0] < len(GRID)
        and
        0 <= start[1] < len(GRID[0])
    ):
        print("Error: Start coordinates are out of bounds.")
        sys.exit(1)

    if not (
        0 <= goal[0] < len(GRID)
        and
        0 <= goal[1] < len(GRID[0])
    ):
        print("Error: Goal coordinates are out of bounds.")
        sys.exit(1)

    if GRID[start[0]][start[1]] == 'N':
        print("Error: Start node cannot be placed on a no-fly zone.")
        sys.exit(1)

    if GRID[goal[0]][goal[1]] == 'N':
        print("Error: Goal node cannot be placed on a no-fly zone.")
        sys.exit(1)

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

    # ==================================================
    # ALGORITHM DISPATCH (CLEAN)
    # ==================================================

    result = None

    # GBFS with h1
    if algorithm == "GBFS" and heuristic == "h1":

        result = gbfs_h1(grid, start, goal)

        if result:
            write_output(result)

            print("\n===== GBFS-H1 RESULTS =====")

            print(
                "Nodes Expanded:",
                result["nodes_expanded"]
            )

            print(
                "Runtime (ms):",
                round(result["runtime_ms"], 4)
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
                "Memory Usage:",
                result["memory_usage"]
            )

            print(
                "Weather Hazard Zones Crossed:",
                result["weather_count"]
            )

            print(
                "Penalty Incurred:",
                result["penalty"]
            )

            print(
                "No-Fly Zones Crossed:",
                result["no_fly_zones_crossed"]
            )

            print(
                "Selected Nodes:",
                result["selected_nodes"]
            )

            print(
                "Frontier History:",
                result["frontier_history"]
            )

            print(
                "Path:",
                result["path"]
            )

            display_path_grid(grid, result["path"])

            print(
                "Time Complexity: O(V log V)"
            )

            print(
                "Space Complexity: O(V)"
            )

        else:
            print("Goal not reachable.")

    # GBFS with h2
    elif algorithm == "GBFS" and heuristic == "h2":

        result = gbfs_h2(grid, start, goal)

        if result:
            write_output(result)

            print("\n===== GBFS-H2 RESULTS =====")

            print(
                "Nodes Expanded:",
                result["nodes_expanded"]
            )

            print(
                "Runtime (ms):",
                round(result["runtime_ms"], 4)
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
                "Memory Usage:",
                result["memory_usage"]
            )

            print(
                "Weather Hazard Zones Crossed:",
                result["weather_count"]
            )

            print(
                "Penalty Incurred:",
                result["penalty"]
            )

            print(
                "No-Fly Zones Crossed:",
                result["no_fly_zones_crossed"]
            )

            print(
                "Selected Nodes:",
                result["selected_nodes"]
            )

            print(
                "Frontier History:",
                result["frontier_history"]
            )

            print(
                "Path:",
                result["path"]
            )

            display_path_grid(grid, result["path"])

            print(
                "Time Complexity: O(V log V)"
            )

            print(
                "Space Complexity: O(V)"
            )

        else:
            print("Goal not reachable.")

    # A* with h1
    elif algorithm == "A*" and heuristic == "h1":

        result = astar_h1(grid, start, goal)

        if result:
            write_output(result)

            print("\n===== A*-H1 RESULTS =====")

            print(
                "Nodes Expanded:",
                result["nodes_expanded"]
            )

            print(
                "Runtime (ms):",
                round(result["runtime_ms"], 4)
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
                "Memory Usage:",
                result["memory_usage"]
            )

            print(
                "Weather Hazard Zones Crossed:",
                result["weather_count"]
            )

            print(
                "Penalty Incurred:",
                result["penalty"]
            )

            print(
                "No-Fly Zones Crossed:",
                result["no_fly_zones_crossed"]
            )

            print(
                "Selected Nodes:",
                result["selected_nodes"]
            )

            print(
                "Frontier History:",
                result["frontier_history"]
            )

            print(
                "Path:",
                result["path"]
            )

            display_path_grid(grid, result["path"])

            print(
                "Time Complexity: O(V log V)"
            )

            print(
                "Space Complexity: O(V)"
            )

        else:
            print("Goal not reachable.")

    if result:
        print(
            "\nResults written to outputPS4.txt"
        )

    # Run comparison
    comparison_results = compare_algorithms(grid, start, goal)

    # Write output to file
    write_output(comparison_results)
   

