import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
import heapq

# --- Constants ---
UNKNOWN = -1
FREE = 0
OCCUPIED = 1
N_CANDIDATE_FRONTIERS = 5 # The number of top frontiers the agent can choose from.

def bresenham_line(r0, c0, r1, c1):
    points = []
    dr, dc = abs(r1 - r0), -abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr + dc
    while True:
        points.append((r0, c0))
        if r0 == r1 and c0 == c1: break
        e2 = 2 * err
        if e2 >= dc: err += dc; r0 += sr
        if e2 <= dr: err += dr; c0 += sc
    return points

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(point, map_shape):
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = point[0] + dr, point[1] + dc
        if 0 <= nr < map_shape[0] and 0 <= nc < map_shape[1]:
            neighbors.append((nr, nc))
    return neighbors

def astar_path(start, goal, current_robot_map):
    if start == goal: return [start], 0
    open_set = []; heapq.heappush(open_set, (0, start))
    came_from = {}; g_score = {start: 0}; f_score = {start: heuristic(start, goal)}
    path_found = False
    while open_set:
        _, current_node = heapq.heappop(open_set)
        if current_node == goal: path_found = True; break
        for neighbor in get_neighbors(current_node, current_robot_map.shape):
            if current_robot_map[neighbor[0], neighbor[1]] == OCCUPIED: continue
            tentative_g_score = g_score.get(current_node, float('inf')) + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current_node; g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    if path_found:
        path = []; temp = goal
        while temp in came_from: path.append(temp); temp = came_from[temp]
        path.append(start); path.reverse(); return path, len(path) - 1
    return None, float('inf')


# --- Core Simulation Logic (adapted into a helper class) ---
class SLAMSimulator:
    """Encapsulates the underlying simulation logic."""
    def __init__(self, sensor_range=8):
        self.sensor_range = sensor_range
        self.map_layout = [
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "X S                                                          X",
            "X XXXXXXX XXXXXXXXXXXXXXXXX   XXXXXXXXXXXXXXXXXX XXXXXXXXXXX X",
            "X X     X X                 X   X                X X         X X",
            "X X     X X   XXXXXXXXXX    X   X   XXXXXXXXXXXXXX X         X X",
            "X X     X X   X        X    X   X   X              X         X X",
            "X XXXXXXX X   X        X    XXXXX   X XXXXXX XXXXXXX   XXXXXXX X",
            "X         X   X        X            X      X X     X   X       X",
            "X XXXXXXXXX   X   XXXXXXXXXXXXXXXXXXX   XX X X     X   X XXXXX X",
            "X X           X   X                   X XX X X     XXXXX X   X X",
            "X X XXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXX XX X X         X X   X X",
            "X X X           X                     XX X XXXXXXXXX X X   X X",
            "X X X           X   XXXXXXXXXXXXXXXXXX XX X           X X   X X",
            "X X X           X   X                X     XXXXXXXXXXX X X   X X",
            "X X X           X   X   XXXXXXXXXXXXXX   X           X X   X X",
            "X X X           X   X   X            X   X           X X   X X",
            "X X XXXXXXXXXXX X   X   X            X   XXXXXXXXXXXXXX X   X X",
            "X X             X   X   X            X                  X   X X",
            "X XXXXXXXXXXXXXXX   X   X            XXXXXXXXXXXXXXXXXXX   X X",
            "X                   X   X                                  X X",
            "X XXXXXXXXXXXXXXXXXXXXXXXXXXXXX   XXXXXXXXXXXXXXXXXXXXXXXXXX X",
            "X                                                          X",
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        ]
        self.ground_truth_map, self.start_pose = self._create_world_from_layout(self.map_layout)
        self.map_shape = self.ground_truth_map.shape
        self.reset()

    def reset(self):
        """Resets the simulation to its initial state."""
        self.robot_pose = self.start_pose
        self.robot_map = np.full(self.map_shape, UNKNOWN, dtype=np.int8)
        self.robot_map[self.robot_pose] = FREE
        self.update_map_with_sensor_data()

    def _create_world_from_layout(self, layout):
        height = len(layout); width = len(layout[0])
        world = np.full((height, width), FREE, dtype=np.int8)
        start_pose = None
        for r, row_str in enumerate(layout):
            for c, char in enumerate(row_str):
                if char == 'X': world[r, c] = OCCUPIED
                elif char == 'S': start_pose = (r, c)
        if start_pose is None: raise ValueError("Map layout must contain a start point 'S'.")
        return world, start_pose

    def update_map_with_sensor_data(self):
        pr, pc = self.robot_pose
        newly_seen_mask = np.zeros(self.map_shape, dtype=bool)
        for angle in np.linspace(0, 2 * np.pi, 180): # 180 rays are sufficient
            end_r = int(round(pr + self.sensor_range * np.sin(angle)))
            end_c = int(round(pc + self.sensor_range * np.cos(angle)))
            line_of_sight = bresenham_line(pr, pc, end_r, end_c)
            for r, c in line_of_sight:
                if not (0 <= r < self.map_shape[0] and 0 <= c < self.map_shape[1]): break
                if self.robot_map[r, c] == UNKNOWN: newly_seen_mask[r, c] = True
                self.robot_map[r, c] = self.ground_truth_map[r, c]
                if self.ground_truth_map[r, c] == OCCUPIED: break
        return np.sum(newly_seen_mask)

    def find_frontiers(self):
        free_space_dilated = morphology.binary_dilation(self.robot_map == FREE, structure=np.ones((3,3)))
        frontiers_mask = free_space_dilated & (self.robot_map == UNKNOWN)
        return [tuple(p) for p in np.argwhere(frontiers_mask)]

    def move_robot(self, new_pose):
        """Move robot to a new pose and update map."""
        self.robot_pose = new_pose
        return self.update_map_with_sensor_data()

# --- The RL Environment ---
class ActiveSLAMEnv(gym.Env):
    """A Gymnasium environment for Active SLAM."""
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, sensor_range=8, max_steps=200):
        super().__init__()
        self.simulator = SLAMSimulator(sensor_range=sensor_range)
        self.map_shape = self.simulator.map_shape
        self.max_steps = max_steps
        self.current_step = 0

        # Action Space: Choose one of the top N candidate frontiers
        self.action_space = spaces.Discrete(N_CANDIDATE_FRONTIERS)

        # Observation Space: 2-channel image (Map + Robot Pose)
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self.map_shape[0], self.map_shape[1], 2),
            dtype=np.int8
        )
        
        self.fig = None
        self.ax = None

    def _get_obs(self):
        """Constructs the 2-channel observation from the current state."""
        # Channel 1: The robot's map
        obs_map = self.simulator.robot_map.copy()

        # Channel 2: The robot's pose (one-hot encoded)
        obs_pose = np.zeros(self.map_shape, dtype=np.int8)
        obs_pose[self.simulator.robot_pose] = 1

        return np.stack([obs_map, obs_pose], axis=-1)

    def _get_info(self):
        """Returns diagnostic information."""
        known_cells = np.sum(self.simulator.robot_map != UNKNOWN)
        total_cells = self.simulator.robot_map.size
        coverage = known_cells / total_cells if total_cells > 0 else 0
        return {"coverage": coverage, "steps": self.current_step}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset()
        self.current_step = 0
        
        # Reset visualization
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        
        # 1. Identify and rank frontiers to present to the agent
        frontiers = self.simulator.find_frontiers()
        if not frontiers:
            # No more frontiers, exploration is done
            return self._get_obs(), 0, True, False, self._get_info()

        # Calculate utility for each frontier (Info Gain - Cost)
        frontier_utilities = []
        for f in frontiers:
            path, cost = astar_path(self.simulator.robot_pose, f, self.simulator.robot_map)
            if path:
                # Simple IG: count unknown cells around frontier
                gain = np.sum(self.simulator.robot_map[
                    max(0, f[0]-self.simulator.sensor_range):min(self.map_shape[0], f[0]+self.simulator.sensor_range+1),
                    max(0, f[1]-self.simulator.sensor_range):min(self.map_shape[1], f[1]+self.simulator.sensor_range+1)
                ] == UNKNOWN)
                utility = gain - 0.5 * cost # Penalize path cost
                frontier_utilities.append({'frontier': f, 'path': path, 'cost': cost, 'utility': utility})
        
        # Sort frontiers by utility
        frontier_utilities.sort(key=lambda x: x['utility'], reverse=True)
        
        # 2. Execute the chosen action
        if action >= len(frontier_utilities):
            # Agent chose an action index that is out of bounds (less frontiers than N_CANDIDATE_FRONTIERS)
            # Penalize and end episode as this is a non-productive state.
            reward = -200 # Heavy penalty for invalid action
            terminated = True
            info = self._get_info()
            info["termination_reason"] = "Invalid action: Chose non-existent frontier."
            return self._get_obs(), reward, terminated, False, info
        
        # Select the goal based on the agent's action
        selected_goal = frontier_utilities[action]
        goal_pos, path, cost = selected_goal['frontier'], selected_goal['path'], selected_goal['cost']
        
        # 3. Move robot and calculate reward
        info_gain = self.simulator.move_robot(goal_pos)
        reward = info_gain - 0.5 * cost # Reward = new cells seen - travel cost

        # 4. Check for termination conditions
        observation = self._get_obs()
        info = self._get_info()
        terminated = (info['coverage'] > 0.98) or (not self.simulator.find_frontiers())
        truncated = self.current_step >= self.max_steps
        
        if terminated: info["termination_reason"] = "Exploration complete."
        if truncated: info["termination_reason"] = "Max steps reached."

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.fig is None:
            plt.ion() # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        self.ax.clear()
        
        # Create display map
        display_map = np.zeros(self.map_shape + (3,), dtype=np.uint8)
        display_map[self.simulator.robot_map == UNKNOWN] = [128, 128, 128]  # Gray
        display_map[self.simulator.robot_map == FREE] = [255, 255, 255]    # White
        display_map[self.simulator.robot_map == OCCUPIED] = [0, 0, 0]      # Black
        
        # Highlight frontiers
        frontiers = self.simulator.find_frontiers()
        for r, c in frontiers:
            display_map[r, c] = [0, 150, 255] # Blue

        self.ax.imshow(display_map, origin='upper')
        
        # Plot robot
        r, c = self.simulator.robot_pose
        self.ax.plot(c, r, 'ro', markersize=8, label="Robot")
        
        # Add info text
        info = self._get_info()
        self.ax.set_title(f"Active SLAM RL - Step: {self.current_step}, Coverage: {info['coverage']:.2%}")
        self.ax.legend(loc='upper right')
        self.ax.grid(False)
        
        plt.draw()
        plt.pause(0.01) # Small pause to allow the plot to update

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig, self.ax = None, None

# --- Example Usage ---
if __name__ == '__main__':
    from gymnasium.utils.env_checker import check_env

    # 1. Check if the environment follows Gymnasium's API
    print("Checking environment validity...")
    env = ActiveSLAMEnv()
    check_env(env.unwrapped)
    print("Environment check passed!")

    # 2. Run a short episode with a random agent
    print("\nRunning a short episode with a random agent...")
    obs, info = env.reset()
    env.render()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        action = env.action_space.sample() # Take a random action
        print(f"Step: {env.current_step+1}, Action: {action}", end="")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f", Reward: {reward:.2f}, Coverage: {info['coverage']:.2%}")
        
        env.render()

    print("\nEpisode finished!")
    print(f"Termination Reason: {info.get('termination_reason', 'N/A')}")
    print(f"Total steps: {info['steps']}")
    print(f"Final Coverage: {info['coverage']:.2%}")
    print(f"Total Reward: {total_reward:.2f}")

    env.close()