import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, ContactSensor

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import torch
from typing import Sequence
import math

##
# Pre-defined configs
##
from torch._tensor import Tensor

@configclass
class DynamicCatchEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 2.0
    num_actions = 22
    num_observations = 32
    num_states = 0

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
                "wrist_1_joint": -(4/8)*math.pi,
                "wrist_2_joint": 0.5*math.pi,
                "wrist_3_joint": 1.0*math.pi,

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

            # "UR5e_shoulder": ImplicitActuatorCfg(
            #     joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint"],
            #     stiffness=800,
            # ),
            # "UR5e_forearm": ImplicitActuatorCfg(
            #     joint_names_expr=["elbow_joint","wrist_1_joint","wrist_2_joint"],
            #     stiffness=64,
            # ),
            # "UR5e_eef": ImplicitActuatorCfg(
            #     joint_names_expr=["wrist_3_joint"],
            #     stiffness=40,
            # ),
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
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"/home/kimsy/Desktop/006_mustard_bottle_instanceable.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd",
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0,0.0,0.0),
                rot=(0.0,0.0,0.0,1.0),
            ),
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
    dist_reward_scale = 1.0
    action_penalty_scale = 0.001
    hit_bonus = 3.0
    drop_penalty_scale = 1.0
    object_height_threshold = 1.2

    # object initial pos
    gravity = 9.81
    X_offset = 0.
    Y_offset = 0.
    Z_offset = 0.9

    range_Xs = (2.5, 3)
    range_Ys = (-0.8, 0.8)
    range_Zs = (-0.1, 0)
    range_t = (0.8, 1.2)

    robot_reach = 0.8

class DynamicCatchEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: DynamicCatchEnvCfg

    def __init__(self, cfg: DynamicCatchEnvCfg,
                 render_mode: str | None = None, **kwargs):
        
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.palm_link_idx = self._robot.find_bodies("palm_link")[0][0]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)
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
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = (self._object.data.body_pos_w[:,0,2] < self.cfg.object_height_threshold) & (self.episode_length_buf > 10)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return self._compute_rewards(
            actions=self.actions,
            object_pos=self.object_pos,
            hand_pos=self.hand_pos,
            dist_reward_scales=self.cfg.dist_reward_scale,
            action_penalty_scales=self.cfg.action_penalty_scale,
            hit_bonus=self.cfg.hit_bonus,
            drop_penalty_scales=self.cfg.drop_penalty_scale,
            num_envs=self.num_envs,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # object state
        object_default_state = self._object.data.default_root_state.clone()[env_ids]
        random_pos, random_vel = self.get_object_random_pose(env_ids=env_ids)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + random_pos + self.scene.env_origins[env_ids]
        )
        object_default_state[:, 7:10] = (
            object_default_state[:, 7:10] + random_vel
        )
        self._object.write_root_state_to_sim(object_default_state, env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()

    def get_object_random_pose(self, env_ids: torch.Tensor | None):
        Xs = torch.rand_like(self._object.data.default_root_state[env_ids,0])*(self.cfg.range_Xs[1]-self.cfg.range_Xs[0]) + self.cfg.range_Xs[0]
        Ys = torch.rand_like(Xs)*(self.cfg.range_Ys[1]-self.cfg.range_Ys[0]) + self.cfg.range_Ys[0]
        Zs = torch.rand_like(Xs)*(self.cfg.range_Zs[1]-self.cfg.range_Zs[0]) + self.cfg.range_Zs[0]

        # t를 먼저 선택
        t = torch.rand_like(Xs)*(self.cfg.range_t[1]-self.cfg.range_t[0]) + self.cfg.range_t[0]

        # Xg 선택
        Xg = torch.rand_like(Xs)*(self.cfg.robot_reach)

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
        Zs += self.cfg.Z_offset

        random_pos = torch.cat([Xs.unsqueeze(-1),Ys.unsqueeze(-1),Zs.unsqueeze(-1)],dim=-1)
        random_vel = torch.cat([Vx.unsqueeze(-1),Vy.unsqueeze(-1),Vz.unsqueeze(-1)],dim=-1)

        return random_pos, random_vel

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.hand_pos,
                self.hand_rot,
                self.robot_joint_pos,
                self.object_pos,
            ),
            dim=-1,
        )

        return {"policy": obs}

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

        print("-------------------------------")
        print(self.contact_sensors_val_raw)
        print(self.contact_sensors_val)

    def _compute_rewards(
            self,
            actions,
            object_pos,
            hand_pos,
            dist_reward_scales,
            action_penalty_scales,
            drop_penalty_scales,
            hit_bonus,
            num_envs,
    ):
        # distance from hand to object
        d = torch.norm(hand_pos - object_pos, p=2, dim=-1)
        dist_reward = torch.exp(-10.0*d)

        # distance for aligning
        d_align = torch.norm(hand_pos[:,1:] - object_pos[:,1:], p=2, dim=-1)
        align_reward = torch.exp(-10.0*d_align)

        # torque penalty for preventing weird motion
        action_penalty = torch.sum(actions**2, dim=-1)

        rewards = (
            dist_reward_scales * dist_reward
            # + dist_reward_scales * align_reward
            - action_penalty_scales * action_penalty
        )

        # bonus reward for hitting hand
        rewards = torch.where(d < 0.05, rewards + hit_bonus, rewards)
        rewards = torch.where(object_pos[:,2] < self.cfg.object_height_threshold, rewards - drop_penalty_scales, rewards)

        return rewards