import time
import random
from collections import deque
import heapq

def create_goal(N):
    return tuple([tuple([1] * N) for _ in range(N)])

def create_initial(N, seed=42):
    random.seed(seed)
    return tuple([tuple([random.randint(0, 1) for _ in range(N)]) for _ in range(N)])

def get_neighbors(state, N):
    neighbors = []
    for i in range(N):
        for j in range(N):
            new_state_list = [list(row) for row in state]
            for di, dj in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    new_state_list[ni][nj] = 1 - new_state_list[ni][nj]
            new_state = tuple(tuple(row) for row in new_state_list)
            neighbors.append((new_state, (i, j)))
    return neighbors

def heuristic(state, N):
    off = sum(row.count(0) for row in state)
    return off

def bfs_solve(initial, N, goal):
    start_time = time.time()
    queue = deque([(initial, 0, [])])
    visited = set([initial])
    nodes_expanded = 0
    while queue:
        state, depth, path = queue.popleft()
        nodes_expanded += 1
        if state == goal:
            return depth, path, time.time() - start_time, len(visited), nodes_expanded
        for new_state, action in get_neighbors(state, N):
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, depth + 1, path + [action]))
    return None, None, time.time() - start_time, len(visited), nodes_expanded

def dfs_solve(initial, N, goal, max_depth=50):
    start_time = time.time()
    stack = [(initial, 0, [])]
    visited = set([initial])
    nodes_expanded = 0
    while stack:
        state, depth, path = stack.pop()
        nodes_expanded += 1
        if state == goal:
            return depth, path, time.time() - start_time, len(visited), nodes_expanded
        if depth >= max_depth:
            continue
        for new_state, action in get_neighbors(state, N):
            if new_state not in visited:
                visited.add(new_state)
                stack.append((new_state, depth + 1, path + [action]))
    return None, None, time.time() - start_time, len(visited), nodes_expanded

def greedy_solve(initial, N, goal):
    start_time = time.time()
    pq = []
    h = heuristic(initial, N)
    heapq.heappush(pq, (h, 0, initial, []))
    visited = set([initial])
    nodes_expanded = 0
    while pq:
        h_val, depth, state, path = heapq.heappop(pq)
        nodes_expanded += 1
        if state == goal:
            return depth, path, time.time() - start_time, len(visited), nodes_expanded
        for new_state, action in get_neighbors(state, N):
            if new_state not in visited:
                visited.add(new_state)
                new_h = heuristic(new_state, N)
                heapq.heappush(pq, (new_h, depth + 1, new_state, path + [action]))
    return None, None, time.time() - start_time, len(visited), nodes_expanded

def a_star_solve(initial, N, goal):
    start_time = time.time()
    pq = []
    g = 0
    h = heuristic(initial, N)
    heapq.heappush(pq, (g + h, g, initial, []))
    visited = {initial: 0}
    nodes_expanded = 0
    while pq:
        f, g_val, state, path = heapq.heappop(pq)
        nodes_expanded += 1
        if state == goal:
            return g_val, path, time.time() - start_time, len(visited), nodes_expanded
        if g_val > visited.get(state, float('inf')):
            continue
        for new_state, action in get_neighbors(state, N):
            new_g = g_val + 1
            if new_g < visited.get(new_state, float('inf')):
                visited[new_state] = new_g
                new_h = heuristic(new_state, N)
                heapq.heappush(pq, (new_g + new_h, new_g, new_state, path + [action]))
    return None, None, time.time() - start_time, len(visited), nodes_expanded

def hill_climbing(initial, N, goal, max_iter=5000):
    start_time = time.time()
    current = initial
    path = []
    nodes_expanded = 0
    for it in range(max_iter):
        nodes_expanded += 1
        if current == goal:
            return len(path), path, time.time() - start_time, nodes_expanded
        neighbors = get_neighbors(current, N)
        best_state = current
        best_h = heuristic(current, N)
        best_action = None
        for new_state, action in neighbors:
            new_h = heuristic(new_state, N)
            if new_h < best_h:
                best_h = new_h
                best_state = new_state
                best_action = action
        if best_h >= heuristic(current, N):
            return None, None, time.time() - start_time, nodes_expanded
        current = best_state
        path.append(best_action)
    return None, None, time.time() - start_time, nodes_expanded


if __name__ == "__main__":
    for N in [2, 3, 5]:
        for seed in [42, 43, 44]:
            initial = create_initial(N, seed)
            goal = create_goal(N)

            print(f"\n=== N={N} | Seed={seed} | OFF={heuristic(initial, N)} ===")

            bfs = bfs_solve(initial, N, goal)
            dfs = dfs_solve(initial, N, goal)
            greedy = greedy_solve(initial, N, goal)
            astar = a_star_solve(initial, N, goal)
            hill = hill_climbing(initial, N, goal)

            print("BFS     ->", bfs)
            print("DFS     ->", dfs)
            print("Greedy  ->", greedy)
            print("A*      ->", astar)
            print("Hill    ->", hill)
