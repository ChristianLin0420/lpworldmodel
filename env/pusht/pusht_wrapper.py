import os
import numpy as np
import gym
from env.pusht.pusht_env import PushTEnv, pymunk_to_shapely
from utils import aggregate_dct

class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
        ):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target, 
        )
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state():
            if self.with_velocity:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,
                        0,  # agent velocities default 0
                    ]
                )
            else:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        
        init_state = generate_state()
        goal_state = generate_state()
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        self.shape = env_info['shape']
    
    def _block_geom(self, pose):
        """Shapely geometry of the block placed at `pose` = (x, y, theta).

        Uses the *same* construction the env uses for its own coverage reward
        (`pusht_env.py:499-501`): a synthetic body at the requested pose carrying the
        real block's shapes.  Because both sides of the IoU below are built this way,
        the measure is self-consistent and equals 1.0 for identical poses.
        """
        if getattr(self, "block", None) is None:
            # `eval_state` may be called on an env that has never been reset (unit
            # tests, offline rescoring).  `_setup` is exactly what `reset` does first
            # and it consumes no RNG, so this cannot perturb a seeded episode.
            self._setup()
        body = self._get_goal_pose_body(np.asarray(pose, dtype=np.float64))
        return pymunk_to_shapely(body, self.block.shapes)

    def _block_iou(self, cur_pose, goal_pose):
        """IoU of the block polygon at `cur_pose` vs at `goal_pose`.

        Pure function of two (x, y, theta) poses, so it needs no rollout widening.
        Unlike the env's own `final_coverage` (`pusht_env.py:524`) it scores the
        *planned* goal rather than the fixed constant `self.goal_pose`
        (`pusht_env.py:716`), and it is an IoU rather than intersection/goal_area.
        """
        try:
            cur_geom = self._block_geom(cur_pose)
            goal_geom = self._block_geom(goal_pose)
            inter = cur_geom.intersection(goal_geom).area
            union = cur_geom.area + goal_geom.area - inter
            return float(inter / union) if union > 0 else 0.0
        except Exception as ex:  # never let a metric kill an eval
            print(f"[eval_state] block_iou failed: {ex}")
            return float("nan")

    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]

        `success` and `state_dist` keep their EXACT historical definitions (including
        dtype: no upcast is introduced) so every archived number stays reproducible.
        Everything else is new and additive:
          success_block   -- the actual task: block within 20px and pi/9 of the goal,
                             irrespective of where the end effector is parked.
                             Note success => success_block, since block_pos_diff <=
                             pos_diff; the historical metric is the STRICTER one.
          block_pos_diff  -- ||goal[2:4] - cur[2:4]||   (the task)
          agent_pos_diff  -- ||goal[0:2] - cur[0:2]||   (end-effector parking)
          angle_diff      -- wrapped |dtheta|, same as used by `success`
          block_iou       -- pose-only IoU of the T at cur vs goal
        """
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = np.linalg.norm(goal_state - cur_state)
        # --- new, additive keys (M2) -------------------------------------------
        agent_pos_diff = np.linalg.norm(goal_state[:2] - cur_state[:2])
        block_pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
        success_block = block_pos_diff < 20 and angle_diff < np.pi / 9
        return {
            'success': success,
            'state_dist': state_dist,
            'success_block': success_block,
            'block_pos_diff': block_pos_diff,
            'agent_pos_diff': agent_pos_diff,
            'angle_diff': angle_diff,
            'block_iou': self._block_iou(cur_state[2:5], goal_state[2:5]),
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset_to_state = init_state
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
