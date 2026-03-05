import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import random

# Sector indices for 360 rays: angle 0 = +X (goal), 90 = +Y (left), 180 = -X (back), 270 = -Y (right)
# 45° cone per sector
FRONT_RAYS = list(range(0, 45)) + list(range(315, 360))
LEFT_RAYS = list(range(45, 135))
BACK_RAYS = list(range(135, 225))
RIGHT_RAYS = list(range(225, 315))


class SpiderEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()

        self.render_mode = render_mode
        self.max_lidar_distance = 5.0
        self.num_rays = 360
        self.position = [0, 0, 0]

        # Action smoothing memory
        self.prev_action = np.zeros(12, dtype=np.float32)

        # Obs: 360 lidar + 4 sector mins + 2 yaw (sin,cos) + 2 velocity (vx,vy) + 4 padding = 372
        # Sector and lidar in [0, max_lidar]; yaw in [-1,1]; velocity normalized
        obs_low = np.concatenate([
            np.zeros(360),
            np.zeros(4),
            np.full(2, -1.0),
            np.full(2, -2.0),  # max linear speed 2
            np.zeros(4),
        ]).astype(np.float32)
        obs_high = np.concatenate([
            np.full(360, self.max_lidar_distance),
            np.full(4, self.max_lidar_distance),
            np.ones(2),
            np.full(2, 2.0),
            np.ones(4),
        ]).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(372,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        self.client = -1
        self._connect_client()

        self.robot_id = None
        self.obstacles = []

        self.reset()

    def _connect_client(self):
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        if self.client < 0:
            raise RuntimeError("Failed to connect to PyBullet physics server")

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.robot_id = None
        self.obstacles = []

    def _ensure_client(self):
        if self.client < 0 or not p.isConnected(self.client):
            self._connect_client()

    # -------------------------
    # WORLD CREATION
    # -------------------------

    def _create_robot(self):
        box_size = [0.2, 0.2, 0.1]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=box_size,
            physicsClientId=self.client
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=box_size,
            rgbaColor=[0, 0, 1, 1],
            physicsClientId=self.client
        )

        robot_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0.2],
            physicsClientId=self.client
        )

        # Damping prevents endless sliding/spinning
        p.changeDynamics(robot_id, -1,
                         linearDamping=0.2,
                         angularDamping=0.2,
                         physicsClientId=self.client)

        return robot_id

    def _create_wall(self, center_position, length=5, thickness=0.1, height=1.0):
        half_extents = [length / 2, thickness / 2, height / 2]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1],
            physicsClientId=self.client
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center_position,
            physicsClientId=self.client
        )

    def _create_wall90(self, center_position, length=5, thickness=0.1, height=1.0):
        half_extents = [length / 2, thickness / 2, height / 2]
        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2], physicsClientId=self.client)

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1],
            physicsClientId=self.client
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center_position,
            baseOrientation=orientation,
            physicsClientId=self.client
        )

    def _create_obstacle(self, position):
        half_extents = [0.2, 0.2, 0.5]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=self.client
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.client
        )

    # -------------------------
    # LIDAR & STATE
    # -------------------------

    def _get_lidar(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)

        rays_from = []
        rays_to = []

        for angle in np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False):
            dx = self.max_lidar_distance * np.cos(angle)
            dy = self.max_lidar_distance * np.sin(angle)
            rays_from.append(base_pos)
            rays_to.append([base_pos[0] + dx, base_pos[1] + dy, base_pos[2]])

        results = p.rayTestBatch(rays_from, rays_to, physicsClientId=self.client)
        distances = np.array([r[2] * self.max_lidar_distance for r in results], dtype=np.float32)
        return distances

    def _get_sector_mins(self, lidar):
        """Min distance in each 90° sector: front (+X), left (+Y), back (-X), right (-Y)."""
        return np.array([
            np.min(lidar[FRONT_RAYS]),
            np.min(lidar[LEFT_RAYS]),
            np.min(lidar[BACK_RAYS]),
            np.min(lidar[RIGHT_RAYS]),
        ], dtype=np.float32)

    def _get_robot_yaw(self):
        _, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        return np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32)

    def _get_robot_velocity(self):
        linear, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        return np.array([float(linear[0]), float(linear[1])], dtype=np.float32)

    # -------------------------
    # GYM API
    # -------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._ensure_client()

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.robot_id = self._create_robot()
        self.prev_action = np.zeros(12, dtype=np.float32)
        
        self.obstacles = []
        self.obstacles.append(self._create_wall([2.5, 3, 0.5], length=7))
        self.obstacles.append(self._create_wall90([-1, 0, 0.5], length=6))
        self.obstacles.append(self._create_wall([2.5, -3, 0.5], length=7))
        self.obstacles.append(self._create_wall90([6, 0, 0.5], length=6))

        
        for _ in range(5):
            x = random.uniform(1, 5)
            y = random.uniform(-3, 3)
            self.obstacles.append(self._create_obstacle([x, y, 0.5]))

        self.position = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)[0]

        return self._get_observation(), {}

    def _get_observation(self):
        lidar = self._get_lidar()
        sector_mins = self._get_sector_mins(lidar)
        yaw = self._get_robot_yaw()
        velocity = self._get_robot_velocity()
        padding = np.zeros(4, dtype=np.float32)
        return np.concatenate([lidar, sector_mins, yaw, velocity, padding])
    def step(self, action):

        self._ensure_client()
        if self.robot_id is None or p.getNumBodies(physicsClientId=self.client) == 0:
            self.reset()

        terminated = False

        # --- Smooth actions
        alpha = 0.2
        action = alpha * self.prev_action + (1 - alpha) * action
        self.prev_action = action

        forward_signal = np.mean(action[0:4])
        lateral_signal = np.mean(action[4:8])
        rotation_signal = np.mean(action[8:12])

        max_linear_speed = 2.0
        max_angular_speed = 1.0

        vx = forward_signal * max_linear_speed
        vy = lateral_signal * max_linear_speed
        wz = np.clip(rotation_signal * max_angular_speed, -1.0, 1.0)

        # Direct velocity control
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[vx, vy, 0],
            angularVelocity=[0, 0, wz],
            physicsClientId=self.client
        )

        p.stepSimulation(physicsClientId=self.client)

        # --- Collision check
        collision = False
        for obs_id in self.obstacles:
            if len(p.getContactPoints(self.robot_id, obs_id, physicsClientId=self.client)) > 0:
                collision = True
                break

        new_position = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)[0]
        progress = new_position[0] - self.position[0]
        lidar = self._get_lidar()
        sector_mins = self._get_sector_mins(lidar)
        front_min, left_min, back_min, right_min = sector_mins
        min_distance = np.min(lidar)

        # ---------- Reward computation ----------

        reward = 0.0

        # Forward progress (primary objective)
        reward += progress * 8.0

        # Goal reward
        if new_position[0] >= 5.0:
            reward += 30.0
            terminated = True
        elif collision:
            reward -= 10.0
            terminated = True
        else:
            # Continuous proximity penalty: strong gradient to stay away from obstacles
            # Penalty grows as we get closer (inverse distance style)
            safe_distance = 1.2
            if min_distance < safe_distance:
                reward -= (safe_distance - min_distance) * 4.0
            # Extra penalty in the danger zone so policy learns to brake/turn early
            if min_distance < 0.6:
                reward -= 2.0

            # Bonus for moving forward only when front is actually clear
            if front_min > 1.8 and vx > 0:
                reward += 0.4 * vx

            # Encourage turning/strafe when front is blocked (learn to go around)
            if front_min < 0.9 and not collision:
                lateral_or_turn = abs(vy) + 0.5 * abs(wz)
                reward += 0.3 * min(lateral_or_turn, 1.0)

            # Small time penalty
            reward -= 0.01

        self.position = new_position

        return self._get_observation(), reward, terminated, False, {}
    def render(self):
        pass

    def close(self):
        if self.client >= 0 and p.isConnected(self.client):
            p.disconnect(physicsClientId=self.client)
        self.client = -1