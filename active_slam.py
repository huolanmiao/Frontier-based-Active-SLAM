import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
import heapq

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

def bresenham_line(r0, c0, r1, c1):
    """
    Bresenham直线算法。
    返回从 (r0, c0) 到 (r1, c1) 路径上的所有栅格点。
    """
    points = []
    dr = abs(r1 - r0)
    dc = -abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr + dc
    
    while True:
        points.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 >= dc:
            err += dc
            r0 += sr
        if e2 <= dr:
            err += dr
            c0 += sc
    return points

def heuristic(a, b):
    """A* 启发函数 (曼哈顿距离)"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(point, map_shape):
    """获取A*的有效邻居点"""
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = point[0] + dr, point[1] + dc
        if 0 <= nr < map_shape[0] and 0 <= nc < map_shape[1]:
            neighbors.append((nr, nc))
    return neighbors

def astar_path(start, goal, current_robot_map):
    """A* 路径规划算法"""
    if start == goal: return [start], 0
    open_set = []; heapq.heappush(open_set, (0, start)); came_from = {}; g_score = {tuple(np.array(p).flatten()): float('inf') for p in np.ndindex(current_robot_map.shape)}; g_score[start] = 0; f_score = {tuple(np.array(p).flatten()): float('inf') for p in np.ndindex(current_robot_map.shape)}; f_score[start] = heuristic(start, goal); path_found = False
    while open_set:
        _, current_node_tuple = heapq.heappop(open_set); current_node = tuple(current_node_tuple)
        if current_node == goal: path_found = True; break
        for neighbor_tuple in get_neighbors(current_node, current_robot_map.shape):
            neighbor = tuple(neighbor_tuple)
            if current_robot_map[neighbor[0], neighbor[1]] == OCCUPIED: continue
            tentative_g_score = g_score.get(current_node, float('inf')) + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current_node; g_score[neighbor] = tentative_g_score; f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if not any(item[1] == neighbor for item in open_set): heapq.heappush(open_set, (f_score[neighbor], neighbor))
    if path_found:
        path = []; temp = goal
        while temp in came_from: path.append(temp); temp = came_from[temp]
        path.append(start); path.reverse(); return path, len(path) - 1
    return None, float('inf')

class ActiveSLAMBlockSimulator:
    def __init__(self, sensor_range=8):
        map_layout = [
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "X S                                                          X",
            "X XXXXXXX XXXXXXXXXXXXXXXXX   XXXXXXXXXXXXXXXXXX XXXXXXXXXXX X",
            "X X     X X               X   X                X X         X X",
            "X X     X X  XXXXXXXXXX   X   X   XXXXXXXXXXXXXX X         X X",
            "X X     X X  X        X   X   X   X              X         X X",
            "X XXXXXXX X  X        X   XXXXX   X XXXXXX XXXXXXX   XXXXXXX X",
            "X         X  X        X           X      X X     X   X       X",
            "X XXXXXXXXX  X  XXXXXXXXXXXXXXXXXXX   XX X X     X   X XXXXX X",
            "X X          X  X                   X XX X X     XXXXX X   X X",
            "X X XXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXX XX X X         X X   X X",
            "X X X           X                     XX X XXXXXXXXX X X   X X",
            "X X X           X  XXXXXXXXXXXXXXXXXX XX X           X X   X X",
            "X X X           X  X                X    XXXXXXXXXXX X X   X X",
            "X X X           X  X   XXXXXXXXXXXXXX  X            X X   X X",
            "X X X           X  X   X            X  X            X X   X X",
            "X X XXXXXXXXXXX X  X   X            X  XXXXXXXXXXXXXX X   X X",
            "X X             X  X   X            X                 X   X X",
            "X XXXXXXXXXXXXXXX  X   X            XXXXXXXXXXXXXXXXXXX   X X",
            "X                  X   X                                  X X",
            "X XXXXXXXXXXXXXXXXXXXXXXXXXXXXX   XXXXXXXXXXXXXXXXXXXXXXXXXX X",
            "X                                                          X",
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        ]
        
        self.ground_truth_map, self.robot_pose = self._create_world_from_layout(map_layout)
        map_size = self.ground_truth_map.shape
        self.robot_map = np.full(map_size, UNKNOWN, dtype=np.int8)
        self.sensor_range = sensor_range
        if self.robot_pose is None: raise ValueError("地图布局中未找到起始点 'S'。")
        self.robot_map[self.robot_pose] = FREE
        self.update_map_with_sensor_data()

    def _create_world_from_layout(self, layout):
        height = len(layout); width = len(layout[0]) if height > 0 else 0; world = np.full((height, width), FREE, dtype=np.int8); start_pose = None
        for r, row_str in enumerate(layout):
            for c, char in enumerate(row_str):
                if char == 'X': world[r, c] = OCCUPIED
                elif char == 'S': start_pose = (r, c); world[r, c] = FREE
        return world, start_pose

    def update_map_with_sensor_data(self):
        """
        使用Bresenham算法和视线遮挡模型更新机器人地图。
        """
        pr, pc = self.robot_pose
        max_r, max_c = self.robot_map.shape

        # 遍历传感器范围边界上的所有点
        for angle in np.linspace(0, 2 * np.pi, 360): # 从360个角度发射射线
            # 计算边界点坐标
            end_r = int(round(pr + self.sensor_range * np.sin(angle)))
            end_c = int(round(pc + self.sensor_range * np.cos(angle)))

            # 获取从机器人到边界点的视线路径
            line_of_sight = bresenham_line(pr, pc, end_r, end_c)

            # 沿着视线路径更新地图，直到遇到障碍物
            for r, c in line_of_sight:
                # 检查坐标是否在地图范围内
                if not (0 <= r < max_r and 0 <= c < max_c):
                    break # 超出地图范围，停止该射线

                # 从真实地图获取信息并更新机器人地图
                self.robot_map[r, c] = self.ground_truth_map[r, c]

                # 如果遇到障碍物，视线被阻挡，停止该射线
                if self.ground_truth_map[r, c] == OCCUPIED:
                    break
    def find_frontiers(self):
        free_space_mask = (self.robot_map == FREE)
        # Ensure structure allows connectivity; default is cross, can use np.ones((3,3)) for 8-connectivity
        free_space_dilated = morphology.binary_dilation(free_space_mask, structure=np.ones((3,3)))
        frontiers_mask = (free_space_dilated) & (self.robot_map == UNKNOWN)
        frontiers = np.argwhere(frontiers_mask)
        
        valid_frontiers = []
        for r, c in frontiers:
            # A frontier point (candidate for observation) should ideally be reachable
            # and allow observation of the unknown.
            # For simplicity, we consider the frontier cell itself as a potential viewpoint.
            # A better approach might be to find a free cell *near* the frontier.
            if self.robot_map[r,c] != OCCUPIED : # Ensure the frontier itself is not known as occupied
                is_adj_to_free = False
                for dr, dc in get_neighbors((r,c), self.robot_map.shape): # Use get_neighbors for consistency
                    if self.robot_map[dr,dc] == FREE:
                        is_adj_to_free = True
                        break
                if is_adj_to_free: # Must be adjacent to an already explored free cell
                    valid_frontiers.append((r,c))
        return [tuple(p) for p in valid_frontiers]


    def choose_next_goal(self, frontiers):
        if not frontiers:
            return None, None, float('-inf') # goal, path, utility

        best_goal = None
        best_path = None
        max_utility = -float('inf')

        w_ig = 1.0
        w_cost = 0.05 # Adjusted for path length as cost
        w_lc = 0.1 # Loop closure potential weight
        w_lc = 0

        for fr, fc in frontiers:
            candidate_goal = (fr, fc)
            
            current_path, path_cost = astar_path(self.robot_pose, candidate_goal, self.robot_map)

            if current_path is None: # Unreachable frontier
                utility = -float('inf')
            else:
                gain = 0
                # Calculate IG from the candidate_goal (end of path)
                goal_r, goal_c = candidate_goal 
                for r_offset in range(-self.sensor_range, self.sensor_range + 1):
                    for c_offset in range(-self.sensor_range, self.sensor_range + 1):
                        if r_offset**2 + c_offset**2 <= self.sensor_range**2:
                            nr, nc = goal_r + r_offset, goal_c + c_offset
                            if 0 <= nr < self.robot_map.shape[0] and \
                               0 <= nc < self.robot_map.shape[1] and \
                               self.robot_map[nr, nc] == UNKNOWN:
                                gain += 1
                
                loop_closure_potential = 0 # Simplified
                # Example: bonus if goal is near map center and far from current pose
                if path_cost > self.sensor_range:
                    dist_to_center = np.sqrt((goal_r - self.robot_map.shape[0]//2)**2 + \
                                             (goal_c - self.robot_map.shape[1]//2)**2)
                    if dist_to_center < self.sensor_range * 1.5 : # Arbitrary condition
                        loop_closure_potential = gain * 0.1 # Small bonus

                utility = w_ig * gain - w_cost * path_cost + w_lc * loop_closure_potential
            
            if utility > max_utility:
                max_utility = utility
                best_goal = candidate_goal
                best_path = current_path
        
        if best_goal:
            print(f"Selected Goal: {best_goal} (Path length: {len(best_path)-1 if best_path else 'N/A'}) Utility: {max_utility:.2f}")
        else:
            print("No suitable goal found.")
        return best_goal, best_path, max_utility

    def move_along_path(self, path):
        """
        Moves the robot step-by-step along the given path, sensing at each step.
        :param path: A list of (r, c) tuples representing the path.
        :return: True if the entire path was traversed, False if movement was interrupted.
        """
        if not path or len(path) <= 1: # Path is just the current location or empty
            return True

        print(f"Attempting to move along path of length {len(path)-1} from {path[0]} to {path[-1]}")
        for i in range(1, len(path)): # Start from the first step *after* current pose
            next_step_pose = path[i]
            
            # Collision check BEFORE making the step, based on current map knowledge
            if self.robot_map[next_step_pose[0], next_step_pose[1]] == OCCUPIED:
                print(f"  Path blocked at {next_step_pose}! Obstacle detected in robot's map. Stopping.")
                # Robot remains at path[i-1] which is self.robot_pose
                return False # Movement interrupted

            # If clear, make the step
            self.robot_pose = next_step_pose

            # SENSE AFTER EACH STEP
            self.update_map_with_sensor_data()
            
        print(f"  Successfully moved to {self.robot_pose} (end of path).")
        return True


    def visualize(self, step, goal, current_path_to_goal, frontiers):
        plt.figure(figsize=(10, 8)) 
        
        display_map = np.zeros(self.robot_map.shape + (3,), dtype=np.uint8)
        display_map[self.robot_map == UNKNOWN] = [150, 150, 150] # Darker Gray
        display_map[self.robot_map == FREE] = [255, 255, 255]   # White
        display_map[self.robot_map == OCCUPIED] = [0, 0, 0]     # Black
        
        for fr, fc in frontiers:
            if 0 <= fr < display_map.shape[0] and 0 <= fc < display_map.shape[1]:
                display_map[fr, fc] = [0, 150, 255] # Light Blue for frontiers
        
        plt.imshow(display_map, origin='lower')
        
        if current_path_to_goal:
            path_r = [p[0] for p in current_path_to_goal]
            path_c = [p[1] for p in current_path_to_goal]
            plt.plot(path_c, path_r, 'y--', linewidth=2, label="Planned Path")

        if self.robot_pose:
             plt.plot(self.robot_pose[1], self.robot_pose[0], 'ro', markersize=8, label="Robot Pose")
        if goal:
            plt.plot(goal[1], goal[0], 'g*', markersize=12, label="Next Goal")
        
        plt.title(f"Active SLAM Simulation (Step {step})")
        plt.legend()
        plt.grid(False)
        plt.savefig(f"active_slam_step_{step}.png")
        # plt.show()
  
  
if __name__ == "__main__":
    simulator = ActiveSLAMBlockSimulator(sensor_range=6) # Use block simulator for more realistic map
    max_exploration_steps = 500 # Max number of frontier selection attempts
    
    # Visualize the global (ground truth) map before exploration starts
    plt.figure(figsize=(10, 8))
    global_map_rgb = np.zeros(simulator.ground_truth_map.shape + (3,), dtype=np.uint8)
    global_map_rgb[simulator.ground_truth_map == UNKNOWN] = [150, 150, 150]
    global_map_rgb[simulator.ground_truth_map == FREE] = [255, 255, 255]
    global_map_rgb[simulator.ground_truth_map == OCCUPIED] = [0, 0, 0]
    plt.imshow(global_map_rgb, origin='lower')
    plt.title("Ground Truth Map (Global Map)")
    plt.plot(simulator.robot_pose[1], simulator.robot_pose[0], 'ro', markersize=8, label="Robot Start")
    plt.legend()
    plt.grid(False)
    plt.savefig("active_slam_global_map.png")
    
    step_cnt = 0
    print("Starting Active SLAM Exploration Simulation")
    for i in range(max_exploration_steps):
        step_cnt += 1
        print(f"\nEXPLORATION STEP {i+1}")
        
        current_frontiers = simulator.find_frontiers()
        if not current_frontiers:
            print("No frontiers left. Exploration might be complete.")
            break
            
        # Decide on the next goal and the path to it
        chosen_goal, planned_path, utility = simulator.choose_next_goal(current_frontiers)
        
        # Visualize current state BEFORE movement
        simulator.visualize(step=f"{i+1} - Planning", goal=chosen_goal, 
                            current_path_to_goal=planned_path, frontiers=current_frontiers)

        if not chosen_goal or not planned_path:
            print("No valid goal or path found. Ending exploration.")
            break
        
        # Attempt to move along the planned path
        movement_successful = simulator.move_along_path(planned_path)
        
        if not movement_successful:
            print("Movement along path was interrupted. Replanning in next step.")
            # The loop will continue, and find_frontiers/choose_next_goal will run with updated map

        # Termination condition (e.g., map sufficiently explored)
        known_cells = np.sum(simulator.robot_map != UNKNOWN)
        total_cells = simulator.robot_map.size
        if (known_cells / total_cells) > 0.98: # Explore 98%
            print(f"Map coverage: {known_cells / total_cells:.2%}.")
            print(f"step cnt: {step_cnt}.")
            print("Map coverage target reached. Exploration complete.")
            break
            
    print("\n--- Final State ---")
    simulator.visualize(step='Final', goal=None, 
                        current_path_to_goal=None, frontiers=simulator.find_frontiers())