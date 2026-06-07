import heapq
import math
import os
import sys
import time


# ==================================================
# DEFAULT GRID
# ==================================================

GRID = [
    ["S", ".", ".", ".", "N", ".", ".", "."],
    [".", "W", ".", ".", "N", ".", ".", "."],
    [".", ".", ".", "W", "N", ".", "W", "."],
    ["N", "N", ".", ".", ".", ".", "N", "."],
    [".", ".", ".", "W", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", "N", "N", "."],
    [".", ".", "W", ".", ".", ".", ".", "E"],
    [".", ".", ".", "W", ".", "W", ".", "."],
]

COST_MAP = {
    "S": 2,
    "E": 2,
    ".": 1,
    "W": 4,
    "N": 8,
}

DEFAULTS = {
    "START_NODE": "0,0",
    "GOAL_NODE": "6,7",
    "HEURISTIC": "h1",
    "ALGORITHM": "GBFS",
    "TESTCASE_ID": "TC01",
    "DEBUG": "false",
    "SAVE_FRONTIER_HISTORY": "false",
}


# ==================================================
# INPUT HANDLING
# ==================================================

def read_input_file(filename="inputPS4.txt"):
    """
    Reads configuration parameters from inputPS4.txt.

    Expected keys:
    START_NODE, GOAL_NODE, HEURISTIC, ALGORITHM, TESTCASE_ID

    Optional keys:
    DEBUG, SAVE_FRONTIER_HISTORY
    """

    params = {}

    if not os.path.exists(filename):
        print(f"Warning: '{filename}' not found. Using default values.")
        return params

    with open(filename, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                print(f"Warning: Invalid line {line_number} ignored -> {line}")
                continue

            key, value = line.split(":", 1)
            params[key.strip().upper()] = value.strip()

    return params


def parse_coordinate(value, label):
    """
    Safely parses coordinates in row,col format.
    """

    try:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"{label} must be in row,col integer format. Received: '{value}'"
        )


def normalize_algorithm(value):
    """
    Accepts GBFS, gbfs, GbFs, A*, a*, astar, etc.
    """

    normalized = value.strip().upper().replace("_", " ")
    normalized = " ".join(normalized.split())

    aliases = {
        "GBFS": "GBFS",
        "GREEDY": "GBFS",
        "GREEDY BEST FIRST SEARCH": "GBFS",
        "GREEDY BEST-FIRST SEARCH": "GBFS",
        "A*": "A*",
        "ASTAR": "A*",
        "A STAR": "A*",
        "A-STAR": "A*",
        "COMPARE": "COMPARE",
        "COMPARISON": "COMPARE",
        "ALL": "COMPARE",
    }

    if normalized not in aliases:
        raise ValueError(f"Invalid ALGORITHM '{value}'. Use GBFS or A*.")

    return aliases[normalized]


def normalize_heuristic(value):
    """
    Accepts h1, H1, h2, H2.
    """

    normalized = value.strip().lower().replace(" ", "")

    if normalized not in ("h1", "h2"):
        raise ValueError(f"Invalid HEURISTIC '{value}'. Use h1 or h2.")

    return normalized


def parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def build_config():
    config = read_input_file()

    for key, value in DEFAULTS.items():
        if key not in config:
            config[key] = value

    start = parse_coordinate(config["START_NODE"], "START_NODE")
    goal = parse_coordinate(config["GOAL_NODE"], "GOAL_NODE")
    algorithm = normalize_algorithm(config["ALGORITHM"])
    heuristic = normalize_heuristic(config["HEURISTIC"])

    return {
        "start": start,
        "goal": goal,
        "algorithm": algorithm,
        "heuristic": heuristic,
        "testcase_id": config["TESTCASE_ID"],
        "debug": parse_bool(config.get("DEBUG", "false")),
        "save_frontier_history": parse_bool(
            config.get("SAVE_FRONTIER_HISTORY", "false")
        ),
    }


def validate_inputs(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError("START_NODE coordinates are out of bounds.")

    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError("GOAL_NODE coordinates are out of bounds.")

    if grid[start[0]][start[1]] == "N":
        raise ValueError("START_NODE cannot be placed on a no-fly zone.")

    if grid[goal[0]][goal[1]] == "N":
        raise ValueError("GOAL_NODE cannot be placed on a no-fly zone.")


def update_grid(grid, start, goal):
    """
    Updates grid with user supplied Start and Goal nodes.
    """

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] in ("S", "E"):
                grid[row][col] = "."

    grid[start[0]][start[1]] = "S"

    if start != goal:
        grid[goal[0]][goal[1]] = "E"

    return grid


# ==================================================
# HEURISTIC FUNCTIONS
# ==================================================

def heuristic_h1(node, goal):
    """
    Euclidean Distance Heuristic.
    """

    return math.sqrt((goal[0] - node[0]) ** 2 + (goal[1] - node[1]) ** 2)


def heuristic_h2(node, goal, grid, start):
    """
    Bounding-Box Risk Weighted Heuristic.

    Professor clarification:
    - H2 is evaluated for the neighbor node being considered.
    - S and E have score 2.
    - N has score 8 in heuristic calculation, although N is not traversable.
    """

    manhattan_distance = abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    row_start = min(node[0], goal[0])
    row_end = max(node[0], goal[0])
    col_start = min(node[1], goal[1])
    col_end = max(node[1], goal[1])

    total_cost = 0
    total_cells = 0

    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            total_cost += get_cell_cost_at((row, col), grid, start, goal)
            total_cells += 1

    average_cost = total_cost / total_cells
    return round(manhattan_distance * average_cost, 2)


def get_heuristic_value(name, node, goal, grid, start):
    if name == "h1":
        return heuristic_h1(node, goal)
    return heuristic_h2(node, goal, grid, start)


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
    N = blocked for traversal, but score 8 for h2 heuristic
    """

    return COST_MAP.get(cell, float("inf"))


def get_cell_cost_at(node, grid, start, goal):
    """
    Returns the correct score for a coordinate.
    This keeps S/E scores correct even after start/goal are moved.
    """

    if node == start:
        return COST_MAP["S"]

    if node == goal:
        return COST_MAP["E"]

    return get_cell_cost(grid[node[0]][node[1]])


def get_neighbors(position, grid):
    """
    Generates valid neighbors in required tie-break order:
    North -> East -> South -> West.
    """

    row, col = position
    directions = [
        (-1, 0),  # North
        (0, 1),   # East
        (1, 0),   # South
        (0, -1),  # West
    ]

    neighbors = []

    for dr, dc in directions:
        nr = row + dr
        nc = col + dc

        if (
            0 <= nr < len(grid)
            and 0 <= nc < len(grid[0])
            and grid[nr][nc] != "N"
        ):
            neighbors.append((nr, nc))

    return neighbors


def reconstruct_path(parent, goal):
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


def calculate_path_cost(path, grid, start, goal):
    """
    Consistent path cost for all algorithms.
    Includes S=2 and E=2 as clarified by professor.
    """

    return sum(get_cell_cost_at(node, grid, start, goal) for node in path)


def count_weather_hazards(path, grid):
    return sum(1 for row, col in path if grid[row][col] == "W")


def calculate_penalty(path, grid):
    return count_weather_hazards(path, grid) * COST_MAP["W"]


def count_no_fly_zones(path, grid):
    return sum(1 for row, col in path if grid[row][col] == "N")


def display_path_grid(grid, path):
    print("\n===== PATH GRID =====")
    print(render_path_grid(grid, path))


def render_path_grid(grid, path):
    grid_copy = [row[:] for row in grid]

    for row, col in path:
        if grid_copy[row][col] not in ("S", "E"):
            grid_copy[row][col] = "*"

    return "\n".join(" ".join(row) for row in grid_copy)


def display_iteration_grid(grid, current, frontier, explored):
    grid_copy = [row[:] for row in grid]

    for row, col in explored:
        if grid_copy[row][col] not in ("S", "E", "N", "W"):
            grid_copy[row][col] = "o"

    for row, col in frontier:
        if grid_copy[row][col] not in ("S", "E", "N", "W"):
            grid_copy[row][col] = "f"

    if grid_copy[current[0]][current[1]] not in ("S", "E", "N", "W"):
        grid_copy[current[0]][current[1]] = "X"

    print("\n===== ITERATION GRID =====")
    for row in grid_copy:
        print(" ".join(row))
    print(f"Selected Node: {current}")
    print(f"Frontier Nodes: {frontier}")
    print(f"Explored Nodes: {explored}")


def make_result(
    algorithm,
    heuristic,
    testcase_id,
    start,
    goal,
    grid,
    path,
    explored,
    heuristic_values,
    selected_nodes,
    frontier,
    frontier_history,
    trap_logs,
    nodes_expanded,
    runtime_ms,
    memory_usage,
    found,
    message,
):
    if path:
        path_cost = calculate_path_cost(path, grid, start, goal)
        path_length = len(path) - 1
        weather_count = count_weather_hazards(path, grid)
        penalty = calculate_penalty(path, grid)
        no_fly_zones_crossed = count_no_fly_zones(path, grid)
    else:
        path_cost = 0
        path_length = 0
        weather_count = 0
        penalty = 0
        no_fly_zones_crossed = 0

    return {
        "algorithm": algorithm,
        "heuristic": heuristic,
        "testcase_id": testcase_id,
        "start": start,
        "goal": goal,
        "path": path,
        "nodes_expanded": nodes_expanded,
        "runtime_ms": runtime_ms,
        "path_length": path_length,
        "path_cost": path_cost,
        "explored": explored,
        "heuristic_values": heuristic_values,
        "penalty": penalty,
        "weather_count": weather_count,
        "no_fly_zones_crossed": no_fly_zones_crossed,
        "memory_usage": memory_usage,
        "frontier": frontier,
        "frontier_history": frontier_history,
        "selected_nodes": selected_nodes,
        "trap_logs": trap_logs,
        "found": found,
        "message": message,
        "time_complexity": "O(E log V)",
        "space_complexity": "O(V)",
    }


# ==================================================
# SEARCH IMPLEMENTATION
# ==================================================

def search(grid, start, goal, algorithm, heuristic, testcase_id, debug=False,
           save_frontier_history=False):
    """
    Shared search implementation for GBFS and A*.
    Kept procedural so it stays close to the original submission style.
    """

    start_time = time.perf_counter()

    if start == goal:
        runtime = (time.perf_counter() - start_time) * 1000
        return make_result(
            algorithm,
            heuristic,
            testcase_id,
            start,
            goal,
            grid,
            [start],
            [],
            {start: 0},
            [],
            [],
            [],
            [],
            0,
            runtime,
            0,
            True,
            "Start and goal are identical; no movement is required.",
        )

    open_list = []
    tie_break = 0

    start_h = get_heuristic_value(heuristic, start, goal, grid, start)
    heapq.heappush(open_list, (start_h, tie_break, start))
    tie_break += 1

    parent = {start: None}
    g_cost = {start: get_cell_cost_at(start, grid, start, goal)}
    closed_set = set()
    explored = []
    heuristic_values = {start: start_h}
    selected_nodes = []
    frontier_history = []
    trap_logs = []
    frontier_seen = {}
    nodes_expanded = 0
    memory_usage = 0

    while open_list:
        _, _, current = heapq.heappop(open_list)
        selected_nodes.append(current)

        if current in closed_set:
            continue

        closed_set.add(current)
        explored.append(current)
        nodes_expanded += 1
        memory_usage = max(memory_usage, len(open_list) + len(closed_set))

        if current == goal:
            runtime = (time.perf_counter() - start_time) * 1000
            path = reconstruct_path(parent, goal)
            final_frontier = [node for _, _, node in open_list]

            return make_result(
                algorithm,
                heuristic,
                testcase_id,
                start,
                goal,
                grid,
                path,
                explored,
                heuristic_values,
                selected_nodes,
                final_frontier,
                frontier_history,
                trap_logs,
                nodes_expanded,
                runtime,
                memory_usage,
                True,
                "Goal reached successfully.",
            )

        for neighbor in get_neighbors(current, grid):
            h_value = get_heuristic_value(heuristic, neighbor, goal, grid, start)
            heuristic_values[neighbor] = h_value

            if algorithm == "GBFS":
                if neighbor in closed_set or neighbor in parent:
                    continue

                parent[neighbor] = current
                heapq.heappush(open_list, (h_value, tie_break, neighbor))
                tie_break += 1

            else:
                tentative_g = (
                    g_cost[current]
                    + get_cell_cost_at(neighbor, grid, start, goal)
                )

                if tentative_g < g_cost.get(neighbor, float("inf")):
                    parent[neighbor] = current
                    g_cost[neighbor] = tentative_g
                    heapq.heappush(
                        open_list,
                        (tentative_g + h_value, tie_break, neighbor),
                    )
                    tie_break += 1

        current_frontier = [node for _, _, node in open_list]

        if save_frontier_history:
            frontier_history.append(current_frontier[:])

        frontier_key = frozenset(current_frontier)
        if not current_frontier:
            trap_logs.append(
                {
                    "heuristic": heuristic,
                    "heuristic_value": heuristic_values.get(current, 0),
                    "trapped_at_node": current,
                    "next_iteration_to_exit": "No exit; frontier empty",
                }
            )
        elif frontier_key in frontier_seen:
            trap_logs.append(
                {
                    "heuristic": heuristic,
                    "heuristic_value": heuristic_values.get(current, 0),
                    "trapped_at_node": current,
                    "next_iteration_to_exit": nodes_expanded + 1,
                }
            )
        else:
            frontier_seen[frontier_key] = nodes_expanded

        memory_usage = max(memory_usage, len(open_list) + len(closed_set))

        if debug:
            display_iteration_grid(grid, current, current_frontier, explored)

    runtime = (time.perf_counter() - start_time) * 1000

    return make_result(
        algorithm,
        heuristic,
        testcase_id,
        start,
        goal,
        grid,
        [],
        explored,
        heuristic_values,
        selected_nodes,
        [],
        frontier_history,
        trap_logs,
        nodes_expanded,
        runtime,
        memory_usage,
        False,
        "Goal not reachable.",
    )


def gbfs_h1(grid, start, goal, testcase_id="TC01", debug=False,
            save_frontier_history=False):
    return search(
        grid,
        start,
        goal,
        "GBFS",
        "h1",
        testcase_id,
        debug,
        save_frontier_history,
    )


def gbfs_h2(grid, start, goal, testcase_id="TC01", debug=False,
            save_frontier_history=False):
    return search(
        grid,
        start,
        goal,
        "GBFS",
        "h2",
        testcase_id,
        debug,
        save_frontier_history,
    )


def astar_h1(grid, start, goal, testcase_id="TC01", debug=False,
             save_frontier_history=False):
    return search(
        grid,
        start,
        goal,
        "A*",
        "h1",
        testcase_id,
        debug,
        save_frontier_history,
    )


# ==================================================
# COMPARISON
# ==================================================

def run_algorithm(algorithm_func, grid, start, goal, algo_name, testcase_id):
    result = algorithm_func(
        grid,
        start,
        goal,
        testcase_id=testcase_id,
        debug=False,
        save_frontier_history=False,
    )

    result["Algorithm"] = algo_name
    result["Completed"] = "Yes" if result["found"] else "No"
    result["Nodes Expanded"] = result["nodes_expanded"]
    result["Time Taken (ms)"] = round(result["runtime_ms"], 6)
    result["Path Cost"] = result["path_cost"]
    result["Path Length"] = result["path_length"]
    result["Memory Usage"] = result["memory_usage"]
    return result


def compare_algorithms(grid, start, goal, testcase_id):
    """
    Optional full comparison.
    It is not auto-run during normal assignment execution.
    """

    return [
        run_algorithm(gbfs_h1, grid, start, goal, "GBFS (h1)", testcase_id),
        run_algorithm(gbfs_h2, grid, start, goal, "GBFS (h2)", testcase_id),
        run_algorithm(astar_h1, grid, start, goal, "A* (h1)", testcase_id),
    ]


def compare_heuristics(grid, start, goal, testcase_id):
    """
    Dedicated h1 vs h2 comparison for GBFS.
    """

    return [
        run_algorithm(gbfs_h1, grid, start, goal, "GBFS (h1)", testcase_id),
        run_algorithm(gbfs_h2, grid, start, goal, "GBFS (h2)", testcase_id),
    ]


def format_results_table(results):
    results = results if isinstance(results, list) else [results]

    header = (
        f"{'Algorithm':<15}{'Complete':<12}{'Runtime(ms)':<14}"
        f"{'Expanded':<12}{'Cost':<10}{'Length':<10}{'Memory':<10}\n"
    )
    header += "-" * 83 + "\n"

    rows = ""
    for result in results:
        rows += (
            f"{result.get('Algorithm', 'N/A'):<15}"
            f"{result.get('Completed', 'N/A'):<12}"
            f"{result.get('Time Taken (ms)', 'N/A'):<14}"
            f"{result.get('Nodes Expanded', 'N/A'):<12}"
            f"{result.get('Path Cost', 'N/A'):<10}"
            f"{result.get('Path Length', 'N/A'):<10}"
            f"{result.get('Memory Usage', 'N/A'):<10}\n"
        )

    return header + rows


def generate_analysis(results):
    results = results if isinstance(results, list) else [results]
    successful = [result for result in results if result.get("found")]

    analysis = "\n======== ANALYSIS ========\n"

    if not successful:
        analysis += "No algorithm reached the goal, so comparison is inconclusive.\n"
        return analysis

    complete_algorithms = [
        result["Algorithm"]
        for result in results
        if result.get("found") and str(result.get("Algorithm", "")).startswith("A*")
    ]

    analysis += "\n(a) Completeness:\n"
    analysis += (
        "    A* is complete for this finite grid when all step costs are positive.\n"
        "    GBFS can find a path in this run, but it is not guaranteed complete or optimal in general.\n"
    )
    if complete_algorithms:
        analysis += f"    Complete algorithm listed: {', '.join(complete_algorithms)}\n"

    least_nodes = min(successful, key=lambda item: item["nodes_expanded"])
    fastest = min(successful, key=lambda item: item["runtime_ms"])
    cheapest = min(successful, key=lambda item: item["path_cost"])

    analysis += f"\n(b) Least Nodes Expanded: {least_nodes['Algorithm']}\n"
    analysis += f"(c) Fastest Algorithm: {fastest['Algorithm']}\n"
    analysis += f"(d) Lowest Path Cost: {cheapest['Algorithm']}\n"

    gbfs_h1 = next(
        (result for result in results if result.get("Algorithm") == "GBFS (h1)"),
        None,
    )
    gbfs_h2 = next(
        (result for result in results if result.get("Algorithm") == "GBFS (h2)"),
        None,
    )

    if gbfs_h1 and gbfs_h2:
        analysis += "\n(e) Heuristic Comparison:\n"
        if gbfs_h2["nodes_expanded"] < gbfs_h1["nodes_expanded"]:
            analysis += (
                "    h2 expanded fewer nodes because its bounding-box risk score "
                "uses future regional costs, including W and N cells.\n"
            )
        elif gbfs_h2["nodes_expanded"] > gbfs_h1["nodes_expanded"]:
            analysis += (
                "    h1 expanded fewer nodes in this run. h2 can be more cautious "
                "because it penalizes risky regions inside the bounding box.\n"
            )
        else:
            analysis += (
                "    h1 and h2 expanded the same number of nodes in this run. "
                "Runtime, memory, and path cost should be used to decide the better heuristic.\n"
            )

    return analysis


# ==================================================
# OUTPUT WRITING
# ==================================================

def format_trap_logs(trap_logs):
    if not trap_logs:
        return "No dead-end or repeated-frontier trap instances recorded.\n"

    text = f"{'Heuristic':<12}{'Value':<12}{'Trapped At':<16}{'Exit Iteration'}\n"
    text += "-" * 58 + "\n"

    for log in trap_logs:
        text += (
            f"{str(log['heuristic']):<12}"
            f"{str(log['heuristic_value']):<12}"
            f"{str(log['trapped_at_node']):<16}"
            f"{log['next_iteration_to_exit']}\n"
        )

    return text


def write_output(result, grid, filename="outputPS4.txt"):
    """
    Writes one normal algorithm result to outputPS4.txt.
    This function no longer mixes single-result and comparison-result structures.
    """

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"===== {result['algorithm']} - {result['heuristic']} RESULTS =====\n\n")
        file.write(f"Testcase ID: {result['testcase_id']}\n")
        file.write(f"Status: {'SUCCESS' if result['found'] else 'FAILED'}\n")
        file.write(f"Message: {result['message']}\n")
        file.write(f"Start Node: {result['start']}\n")
        file.write(f"Goal Node: {result['goal']}\n\n")

        file.write("===== UPDATED GRID =====\n")
        file.write("\n".join(" ".join(row) for row in grid))
        file.write("\n\n")

        file.write("===== METRICS =====\n")
        file.write(f"Nodes Expanded: {result['nodes_expanded']}\n")
        file.write(f"Runtime (ms): {round(result['runtime_ms'], 6)}\n")
        file.write(f"Path Length: {result['path_length']}\n")
        file.write(f"Path Cost: {result['path_cost']}\n")
        file.write(f"Memory Usage (OPEN + CLOSED): {result['memory_usage']}\n")
        file.write(f"Weather Hazard Zones Crossed: {result['weather_count']}\n")
        file.write(f"Penalty Incurred: {result['penalty']}\n")
        file.write(f"No-Fly Zones Crossed: {result['no_fly_zones_crossed']}\n")
        file.write(f"Time Complexity: {result['time_complexity']}\n")
        file.write(f"Space Complexity: {result['space_complexity']}\n\n")

        file.write("===== SEARCH DETAILS =====\n")
        file.write(f"Selected Nodes: {result['selected_nodes']}\n")
        file.write(f"Frontier at Completion: {result['frontier']}\n")
        file.write(f"Explored Nodes: {result['explored']}\n")
        file.write(f"Heuristic Values: {result['heuristic_values']}\n")
        file.write(f"Path: {result['path']}\n\n")

        file.write("===== FINAL PATH GRID =====\n")
        file.write(render_path_grid(grid, result["path"]))
        file.write("\n\n")

        file.write("===== TRAP / INEFFICIENT REGION LOG =====\n")
        file.write(format_trap_logs(result["trap_logs"]))

        if result.get("frontier_history"):
            file.write("\n===== FRONTIER HISTORY =====\n")
            file.write(str(result["frontier_history"]))
            file.write("\n")


def write_comparison_output(results, filename="comparisonPS4.txt"):
    """
    Writes comparison results to a separate file to avoid overwriting outputPS4.txt.
    """

    output_text = "===== ALGORITHM COMPARISON RESULTS =====\n\n"
    output_text += format_results_table(results)
    output_text += generate_analysis(results)

    with open(filename, "w", encoding="utf-8") as file:
        file.write(output_text)

    print(output_text)


def print_result_summary(result, grid):
    print(f"\n===== {result['algorithm']}-{result['heuristic']} RESULTS =====")
    print("Status:", "SUCCESS" if result["found"] else "FAILED")
    print("Message:", result["message"])
    print("Nodes Expanded:", result["nodes_expanded"])
    print("Runtime (ms):", round(result["runtime_ms"], 6))
    print("Path Length:", result["path_length"])
    print("Path Cost:", result["path_cost"])
    print("Memory Usage:", result["memory_usage"])
    print("Weather Hazard Zones Crossed:", result["weather_count"])
    print("Penalty Incurred:", result["penalty"])
    print("No-Fly Zones Crossed:", result["no_fly_zones_crossed"])
    print("Selected Nodes:", result["selected_nodes"])
    print("Path:", result["path"])
    print("Time Complexity:", result["time_complexity"])
    print("Space Complexity:", result["space_complexity"])

    if result["path"]:
        display_path_grid(grid, result["path"])


# ==================================================
# MAIN DRIVER
# ==================================================

def main():
    try:
        config = build_config()

        start = config["start"]
        goal = config["goal"]
        heuristic = config["heuristic"]
        algorithm = config["algorithm"]
        testcase_id = config["testcase_id"]
        debug = config["debug"] or "--debug" in sys.argv
        save_frontier_history = (
            config["save_frontier_history"]
            or "--save-frontier-history" in sys.argv
        )

        grid = [row[:] for row in GRID]
        validate_inputs(grid, start, goal)
        grid = update_grid(grid, start, goal)

        print("\n===== CONFIGURATION =====")
        print("Start Node :", start)
        print("Goal Node  :", goal)
        print("Heuristic  :", heuristic)
        print("Algorithm  :", algorithm)
        print("Test Case  :", testcase_id)

        print("\n===== UPDATED GRID =====")
        for row in grid:
            print(" ".join(row))

        if "--compare" in sys.argv or algorithm == "COMPARE":
            comparison_results = compare_algorithms(grid, start, goal, testcase_id)
            write_comparison_output(comparison_results, "comparisonPS4.txt")
            print("\nComparison written to comparisonPS4.txt")
            return

        if "--compare-heuristics" in sys.argv:
            comparison_results = compare_heuristics(grid, start, goal, testcase_id)
            write_comparison_output(
                comparison_results,
                "heuristicComparisonPS4.txt",
            )
            print("\nHeuristic comparison written to heuristicComparisonPS4.txt")
            return

        if algorithm == "GBFS" and heuristic == "h1":
            result = gbfs_h1(
                grid,
                start,
                goal,
                testcase_id,
                debug,
                save_frontier_history,
            )
        elif algorithm == "GBFS" and heuristic == "h2":
            result = gbfs_h2(
                grid,
                start,
                goal,
                testcase_id,
                debug,
                save_frontier_history,
            )
        elif algorithm == "A*":
            if heuristic != "h1":
                print("Warning: A* comparison is defined with h1. Using h1.")
            result = astar_h1(
                grid,
                start,
                goal,
                testcase_id,
                debug,
                save_frontier_history,
            )
        else:
            raise ValueError("Unsupported algorithm and heuristic combination.")

        write_output(result, grid, "outputPS4.txt")
        print_result_summary(result, grid)
        print("\nResults written to outputPS4.txt")

        if not result["found"]:
            sys.exit(1)

    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
