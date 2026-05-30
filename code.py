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


def read_input_file(filename="inputPS4.txt"):
    params = {}

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if ":" not in line:
                    print(f"Warning: Invalid line ignored -> {line}")
                    continue

                key, value = line.split(":", 1)
                params[key.strip()] = value.strip()

    except FileNotFoundError:
        print(f"Warning: '{filename}' not found. Using default values.")

    return params


def update_grid(grid, start, goal):

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] in ("S", "E"):
                grid[i][j] = "."

    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"

    return grid


if __name__ == "__main__":

    config = read_input_file()

    DEFAULTS = {
        "START_NODE": "0,0",
        "GOAL_NODE": "6,7",
        "HEURISTIC": "h1",
        "ALGORITHM": "GBFS",
        "TESTCASE_ID": "TC01"
    }

    for key, value in DEFAULTS.items():
        if key not in config:
            print(
                f"Warning: '{key}' missing. "
                f"Using default value '{value}'."
            )

    start = tuple(
        map(int,
            config.get(
                "START_NODE",
                DEFAULTS["START_NODE"]
            ).split(","))
    )

    goal = tuple(
        map(int,
            config.get(
                "GOAL_NODE",
                DEFAULTS["GOAL_NODE"]
            ).split(","))
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

    grid = [row[:] for row in GRID]
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