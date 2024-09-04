import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, mdp
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, ContactSensor

from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import torch
from typing import Sequence
import math
import numpy as np

##
# Pre-defined configs
##
from torch._tensor import Tensor

@configclass
class DynamicCatchEnvCfgV1(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 2.0
    num_actions = 22
    one_frame_obs = 70
    num_observations = 140
    one_frame_states = 73
    num_states = 146

    asymmetric_obs = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=3.5, replicate_physics=True)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/kimsy/RL-kimsy/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/catchpolicy/assets/sy_ur5e_with_allegro_right_fsr/sy_ur5e_with_allegro_right_fsr.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -(2/5)*math.pi,
                "elbow_joint": -(2/6)*math.pi,
                "wrist_1_joint": 0.8*math.pi,#-(4/8)*math.pi-0.5*math.pi,
                "wrist_2_joint": -0.5*math.pi,
                "wrist_3_joint": 0.0,#1.0*math.pi,

                "joint_index_0": 0.0,
                "joint_index_1": 0.0,
                "joint_index_2": 0.0,
                "joint_index_3": 0.0,
                "joint_middle_0": 0.0,
                "joint_middle_1": 0.0,
                "joint_middle_2": 0.0,
                "joint_middle_3": 0.0,
                "joint_ring_0":0.0,
                "joint_ring_1":0.0,
                "joint_ring_2":0.0,
                "joint_ring_3":0.0,
                "joint_thumb_0":0.263,
                "joint_thumb_1":0.0,
                "joint_thumb_2":0.0,
                "joint_thumb_3":0.0,
            },
            pos=(0.0,0.0,0.81),
            rot=(0.0,0.0,0.0,1.0),
        ),
        actuators={            
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["joint_index_[0-3]","joint_middle_[0-3]","joint_ring_[0-3]","joint_thumb_[0-3]"],
                effort_limit=0.5,
                velocity_limit=100.0,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
    )

    actuated_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "joint_index_0",
        "joint_index_1",
        "joint_index_2",
        "joint_index_3",
        "joint_middle_0",
        "joint_middle_1",
        "joint_middle_2",
        "joint_middle_3",
        "joint_ring_0",
        "joint_ring_1",
        "joint_ring_2",
        "joint_ring_3",
        "joint_thumb_0",
        "joint_thumb_1",
        "joint_thumb_2",
        "joint_thumb_3",
    ]

    # contact sensors
    contact_sensors = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/fsr_.*", update_period=0.0, history_length=1, debug_vis=True
    )

    # table
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.81),
                                                  rot=(0.0, 0.0, 0.0, -1.0))
    )

    # multi-color cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"/home/kimsy/Desktop/006_mustard_bottle_instanceable.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0,0.0,0.0),
                rot=(0.0,0.0,0.0,1.0),
            ),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            )
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )

    # reward scales
    dist_reward_scale = 10.0
    action_penalty_scale = 0.001
    hit_bonus = 3.0
    drop_penalty_scale = 10.0
    object_height_threshold = 1.0
    reach_goal_bonus = 10.0
    success_tolerance = 0.05
    act_moving_average = 1.0
    num_stacks = 2

    # object initial pos
    gravity = 9.81
    X_offset = 0.
    Y_offset = 0.
    Z_offset = 0.9

    range_Xs = (2.5, 3)
    range_Ys = (-0.8, 0.8)
    range_Zs = (-0.1, 0.0)
    range_t = (0.8, 1.2)

    robot_reach = 0.8

# @configclass
# class EventCfg:
#     object_mass = EventTerm(
#         func=mdp.
#     )

class DynamicCatchEnvV1(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: DynamicCatchEnvCfgV1

    def __init__(self, cfg: DynamicCatchEnvCfgV1,
                 render_mode: str | None = None, **kwargs):
        
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.num_robot_dofs = self._robot.num_joints

        joint_pos_limits = self._robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.palm_link_idx = self._robot.find_bodies("palm_link")[0][0]

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in self.cfg.actuated_joint_names:
            print(self._robot.joint_names.index(joint_name), joint_name)
            self.actuated_dof_indices.append(self._robot.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # buffers for position targets
        self.robot_dof_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # default goal position
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_rot[:,0] = 1.0

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # buffers for obs
        self.obs_buf_stack_frames = []
        self.states_buf_stack_frames = []
        for i in range(self.cfg.num_stacks):
            self.obs_buf_stack_frames.append(torch.zeros((self.num_envs, self.cfg.one_frame_obs), device=self.device))
            self.states_buf_stack_frames.append(torch.zeros((self.num_envs, self.cfg.one_frame_states), device=self.device))

        self.reduced_obs_buf = torch.zeros((self.num_envs, self.cfg.one_frame_obs*self.cfg.num_stacks), device=self.device)
        self.state_buf = torch.zeros((self.num_envs, self.cfg.one_frame_states*self.cfg.num_stacks), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.cube)
        self._contact_sensors = ContactSensor(self.cfg.contact_sensors)
        self._table = RigidObject(self.cfg.table)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object
        self.scene.sensors["contact_sensors"] = self._contact_sensors
        # self.scene.rigid_objects["table"] = self._table

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        self.cur_targets[:,self.actuated_dof_indices[6:]] = scale(
            self.actions[:,6:],
            self.robot_dof_lower_limits[:,self.actuated_dof_indices[6:]],
            self.robot_dof_upper_limits[:,self.actuated_dof_indices[6:]],
        )
        self.cur_targets[:,self.actuated_dof_indices[6:]] = (
            self.cfg.act_moving_average * self.cur_targets[:,self.actuated_dof_indices[6:]]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices[6:]]
        )

        self.cur_targets[:,self.actuated_dof_indices[:6]] = self.prev_targets[:,self.actuated_dof_indices[:6]] + self.actions[:,:6] * self.cfg.action_scale * self.dt

        self.cur_targets[:,self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.robot_dof_lower_limits[:, self.actuated_dof_indices],
            self.robot_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:,self.actuated_dof_indices] = self.cur_targets[:,self.actuated_dof_indices]

        self._robot.set_joint_position_target(self.cur_targets[:,self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = (self._object.data.body_pos_w[:,0,2] < self.cfg.object_height_threshold) & (self.episode_length_buf > 10)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        (
            total_reward,
            self.successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.episode_length_buf,
            self.max_episode_length,
            self.goal_pos,
            self.goal_rot,
            self.object_pos,
            self.hand_pos,
            self.actions,
            self.cfg.dist_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.drop_penalty_scale,
            self.cfg.reach_goal_bonus,
            self.cfg.hit_bonus,
            self.cfg.success_tolerance,
        )

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_goal_pose(goal_env_ids)
        
        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES    

        super()._reset_idx(env_ids)

        self.reduced_obs_buf[env_ids] = torch.zeros((self.cfg.one_frame_obs*self.cfg.num_stacks),device=self.device)
        self.state_buf[env_ids] = torch.zeros((self.cfg.one_frame_states*self.cfg.num_stacks),device=self.device)

        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )

        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits[env_ids], self.robot_dof_upper_limits[env_ids])
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos
        self.prev_targets[env_ids] = joint_pos

        # object state
        object_default_state = self._object.data.default_root_state.clone()[env_ids]
        random_pos, random_vel, goal_pos = self.get_object_random_pose(env_ids=env_ids)

        self.goal_pos[env_ids] = goal_pos
        # reset goals
        self._reset_goal_pose(env_ids)

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + random_pos + self.scene.env_origins[env_ids]
        )
        object_default_state[:, 7:10] = (
            object_default_state[:, 7:10] + random_vel
        )
        self._object.write_root_state_to_sim(object_default_state, env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()

    def _reset_goal_pose(self, env_ids):
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        self.reset_goal_buf[env_ids] = 0

    def get_object_random_pose(self, env_ids: torch.Tensor | None):
        Xs = torch.rand_like(self._object.data.default_root_state[env_ids,0])*(self.cfg.range_Xs[1]-self.cfg.range_Xs[0]) + self.cfg.range_Xs[0]
        Ys = torch.rand_like(Xs)*(self.cfg.range_Ys[1]-self.cfg.range_Ys[0]) + self.cfg.range_Ys[0]
        Zs = torch.rand_like(Xs)*(self.cfg.range_Zs[1]-self.cfg.range_Zs[0]) + self.cfg.range_Zs[0]

        # t를 먼저 선택
        t = torch.rand_like(Xs)*(self.cfg.range_t[1]-self.cfg.range_t[0]) + self.cfg.range_t[0]

        # Xg 선택
        Xg = torch.rand_like(Xs)*(self.cfg.robot_reach)
        Xg = torch.clamp_min(Xg, 0.5)

        # Vx 계산
        Vx = (Xg-Xs)/t

        # Yg의 범위 설정
        max_Yg = torch.clamp_min((self.cfg.robot_reach*self.cfg.robot_reach - Xg * Xg)**0.5, 0.)
        # Yg를 random으로 선택
        Yg = torch.rand_like(Xs)*(max_Yg-(-max_Yg)) + (-max_Yg)

        # Vy를 계산
        Vy = (Yg - Ys) / t

        # Zg의 범위 설정
        max_Zg = torch.clamp_min((self.cfg.robot_reach*self.cfg.robot_reach - Xg * Xg - Yg * Yg)**0.5, 0.)

        # Zg를 random으로 선택
        Zg = torch.rand_like(Xs)*(max_Zg)

        # Vz를 계산
        Vz = (Zg - Zs + 0.5 * self.cfg.gravity * (t ** 2)) / t
        Zg = torch.clamp_min(Zg+self.cfg.Z_offset, self.cfg.Z_offset)
        Zs += self.cfg.Z_offset

        random_pos = torch.cat([Xs.unsqueeze(-1),Ys.unsqueeze(-1),Zs.unsqueeze(-1)],dim=-1)
        random_vel = torch.cat([Vx.unsqueeze(-1),Vy.unsqueeze(-1),Vz.unsqueeze(-1)],dim=-1)

        goal_pos = torch.cat([Xg.unsqueeze(-1),Yg.unsqueeze(-1),Zg.unsqueeze(-1)],dim=-1)

        return random_pos, random_vel, goal_pos

    def _get_observations(self) -> dict:
        self.compute_reduced_obs()

        observations = {"policy": self.reduced_obs_buf}
        if self.cfg.asymmetric_obs:
            self.compute_full_state()
        if self.cfg.asymmetric_obs:
            observations = {"policy": self.reduced_obs_buf, "critic":self.state_buf}
        return observations

    def compute_full_state(self):
        self.state_buf[:,0:3] = self.hand_pos
        self.state_buf[:,3:7] = self.hand_rot
        self.state_buf[:,7:29] = self.robot_joint_pos
        self.state_buf[:,29:32] = (self.goal_pos-self.hand_pos).clone()
        self.state_buf[:,32:48] = self.contact_sensors_val
        self.state_buf[:,48:70] = self.actions
        self.state_buf[:,70:73] = self.object_pos

        for i in range(len(self.states_buf_stack_frames)-1):
            self.state_buf[:, (i+1)*self.cfg.one_frame_states:(i+2)*self.cfg.one_frame_states] = self.states_buf_stack_frames[i]
            self.states_buf_stack_frames[i] = self.state_buf[:, (i)*self.cfg.one_frame_states:(i+1)*self.cfg.one_frame_states].clone()

    def compute_reduced_obs(self):
        self.reduced_obs_buf[:,0:3] = self.hand_pos
        self.reduced_obs_buf[:,3:7] = self.hand_rot
        self.reduced_obs_buf[:,7:29] = self.robot_joint_pos
        self.reduced_obs_buf[:,29:32] = (self.goal_pos-self.hand_pos).clone()
        self.reduced_obs_buf[:,32:48] = self.contact_sensors_val
        self.reduced_obs_buf[:,48:70] = self.actions

        for i in range(len(self.obs_buf_stack_frames)-1):
            self.reduced_obs_buf[:, (i+1)*self.cfg.one_frame_obs:(i+2)*self.cfg.one_frame_obs] = self.obs_buf_stack_frames[i]
            self.obs_buf_stack_frames[i] = self.reduced_obs_buf[:, (i)*self.cfg.one_frame_obs:(i+1)*self.cfg.one_frame_obs].clone()

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.hand_pos = self._robot.data.body_pos_w[env_ids, self.palm_link_idx]
        self.hand_pos -= self.scene.env_origins
        self.hand_rot = self._robot.data.body_quat_w[env_ids, self.palm_link_idx]
        self.robot_joint_pos = self._robot.data.joint_pos
        self.object_pos = self._object.data.root_pos_w - self.scene.env_origins
        self.contact_sensors_val_raw = self.scene["contact_sensors"].data.net_forces_w[..., 2]
        self.contact_sensors_val = torch.where(self.contact_sensors_val_raw != 0.0, 1.0, 0.0)

@torch.jit.script
def compute_rewards(
    reset_buf: Tensor,
    reset_goal_buf: Tensor,
    successes: Tensor,
    episode_length: Tensor,
    max_episode_length: int,
    goal_pos: Tensor,
    goal_rot: Tensor,
    object_pos: Tensor,
    hand_pos: Tensor,
    actions: Tensor,
    dist_reward_scales: float,
    action_penalty_scales: float,
    drop_penalty_scales: float,
    reach_goal_bonus: float,
    hit_bonus: float,
    success_tolerance: float,
):

    # distance from hand to goal position
    d_catch = torch.norm(hand_pos - goal_pos, p=2, dim=-1)
    dist_reward = torch.exp(-10.0*d_catch)

    # distance from hand to obejct
    d_object = torch.norm(hand_pos - object_pos, p=2, dim=-1)

    # distance from object to goal
    d_goal = torch.norm(object_pos - goal_pos, p=2, dim=-1)

    # torque penalty for preventing weird motion
    action_penalty = torch.sum(actions**2, dim=-1)

    rewards = (
        dist_reward_scales * dist_reward
        - action_penalty_scales * action_penalty
    )

    # bonus reward for hitting hand
    rewards = torch.where((d_object < 0.1)&(d_goal < 0.1), rewards + hit_bonus, rewards)
    rewards = torch.where((episode_length > 10)&(object_pos[:,2] < 1.0), rewards - drop_penalty_scales, rewards)

    return rewards, successes

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )