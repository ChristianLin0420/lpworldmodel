import os
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from collections.abc import Mapping
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
    "link",  # RDMReg link h(.); part of the forward path
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        # NB seed*n+1 (the original) DEGENERATES AT seed=0 to [1]*n_evals -- every
        # "episode" is then the same initial condition, so a seed-0 eval measures CEM
        # stochasticity on one task instance rather than an n_evals-episode success
        # rate. It also gave overlapping episode sets across seeds ([1..50] for seed 1,
        # [1,3,5,..] for seed 2), so eval noise was not common-mode and did not cancel
        # in a paired difference. Disjoint blocks fix both: seed s -> [s*n_evals+1 ...].
        n_ev = cfg_dict["n_evals"]
        self.eval_seed = [cfg_dict["seed"] * n_ev + n + 1 for n in range(n_ev)]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    result = {k: v for k, v in payload.items() if k in ALL_MODEL_KEYS}
    result["epoch"] = payload.get("epoch")
    return result


def _in_chans_from(sd):
    """Recover in_chans from a proprio/action encoder state_dict.

    Both ProprioceptiveEmbedding and Embedder start with nn.Conv1d(in_chans, ...),
    whose weight is (out, in_chans, k). Reading it here means planning does not
    need the training dataset just to rebuild the module.
    """
    if not isinstance(sd, Mapping):
        return None
    w = sd.get("patch_embed.weight")
    return None if w is None else w.shape[1]


def _emb_dim_from(sd):
    """Recover emb_dim from a proprio/action encoder state_dict.

    Same trick as _in_chans_from but on dim 0: nn.Conv1d(in_chans, emb_dim, k) has
    weight (emb_dim, in_chans, k). Needed because use_pose changes the proprio
    encoder's OUTPUT width -- with pose on, its embedding is ADDED to the action
    embedding, so it is built with action_emb_dim (384) instead of the vestigial
    proprio_emb_dim (10). Reading the width from the checkpoint keeps planning
    correct for both, and does not depend on the flag surviving in train_cfg.
    """
    if not isinstance(sd, Mapping):
        return None
    w = sd.get("patch_embed.weight")
    return None if w is None else w.shape[0]


def _load_pose_dyn(model, repo_root=None):
    """Attach the environment's pose-dynamics map to a use_pose model.

    Fit once by assets/pose_dynamics_pusht.pt (see VWorldModel._roll_pose). It is
    environment dynamics, so it is not part of any checkpoint and is shared by every
    arm. Silent no-op when pose is off; warns loudly when pose is ON but the map is
    missing, because the fallback (hold-last pose) is what made a healthy refframe
    checkpoint score 0.020.
    """
    import torch as _t
    from pathlib import Path as _P
    if not getattr(model, "use_pose", False):
        return model
    root = _P(repo_root) if repo_root else _P(__file__).resolve().parent
    f = root / "assets" / "pose_dynamics_pusht.pt"
    if not f.exists():
        print(f"WARNING use_pose is ON but {f} is missing -- rollout will hold the last "
              f"observed pose, which is known to destroy planning. Refusing to pretend.")
        return model
    model.pose_dyn = _t.load(f, map_location="cpu")["W"]
    print(f"pose dynamics loaded: {tuple(model.pose_dyn.shape)} from {f}")
    return model


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    """Rebuild the world model from train_cfg and restore its weights.

    Checkpoints hold state_dicts (see train.py save_ckpt), so every submodule is
    constructed from config here and then loaded. Checkpoints from before that
    change pickled whole modules; those are still accepted.
    """
    model_ckpt = Path(model_ckpt)  # callers pass either a str or a Path
    if not model_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {model_ckpt}")
    payload = load_ckpt(model_ckpt, device)
    print(f"Loading model from epoch {payload.get('epoch')}: {model_ckpt}")

    def restore(name, module):
        """Return the legacy pickled module, or load a state_dict into `module`."""
        v = payload.get(name)
        if isinstance(v, torch.nn.Module):
            return v.to(device)
        if module is None:
            return None
        if isinstance(v, Mapping):
            module.load_state_dict(v)
        else:
            print(f"WARNING: '{name}' absent from checkpoint; using fresh init")
        return module.to(device)

    encoder = restore("encoder", hydra.utils.instantiate(train_cfg.encoder))

    proprio_encoder = restore(
        "proprio_encoder",
        hydra.utils.instantiate(
            train_cfg.proprio_encoder,
            in_chans=_in_chans_from(payload.get("proprio_encoder")) or 1,
            emb_dim=_emb_dim_from(payload.get("proprio_encoder")) or train_cfg.proprio_emb_dim,
        ),
    )
    action_encoder = restore(
        "action_encoder",
        hydra.utils.instantiate(
            train_cfg.action_encoder,
            in_chans=_in_chans_from(payload.get("action_encoder")) or 1,
            emb_dim=train_cfg.action_emb_dim,
        ),
    )

    action_conditioning = train_cfg.get("action_conditioning", "concat")
    if not train_cfg.has_predictor:
        raise ValueError("Planning requires a predictor")
    # mirrors train.py init_models so the shapes match the checkpoint
    if action_conditioning == "adaln":
        predictor = hydra.utils.instantiate(
            train_cfg.predictor,
            num_frames=train_cfg.num_hist,
            num_patches=encoder.num_patches,
            input_dim=encoder.emb_dim,
            hidden_dim=encoder.emb_dim,
            output_dim=encoder.emb_dim,
        )
    else:
        num_patches = 1 if encoder.latent_ndim == 1 else (train_cfg.img_size // 16) ** 2
        if train_cfg.concat_dim == 0:
            num_patches += 2
        predictor = hydra.utils.instantiate(
            train_cfg.predictor,
            num_patches=num_patches,
            num_frames=train_cfg.num_hist,
            dim=encoder.emb_dim
            + (
                proprio_encoder.emb_dim * train_cfg.num_proprio_repeat
                + action_encoder.emb_dim * num_action_repeat
            )
            * train_cfg.concat_dim,
        )
    predictor = restore("predictor", predictor)

    link_cfg = train_cfg.get("link", None)
    link = None
    if link_cfg is not None and link_cfg.get("_target_", None) is not None:
        link = restore("link", hydra.utils.instantiate(link_cfg))

    decoder = None
    if train_cfg.has_decoder:
        if isinstance(payload.get("decoder"), torch.nn.Module):
            decoder = payload["decoder"].to(device)
        elif train_cfg.env.decoder_path is not None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            ckpt = torch.load(os.path.join(base_path, train_cfg.env.decoder_path))
            decoder = (ckpt["decoder"] if isinstance(ckpt, dict) else ckpt).to(device)
            if isinstance(payload.get("decoder"), Mapping):
                decoder.load_state_dict(payload["decoder"])
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint and is not provided in config"
            )

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=decoder,
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
        action_conditioning=action_conditioning,
        regularizer=None,
        reg_weight=0.0,
        detach_target=train_cfg.get("detach_target", True),
        # Without this the pose signal is silently dropped at PLAN time while the
        # model still trains with it -- the arm would be evaluated as a different model.
        use_pose=bool(train_cfg.get("use_pose", False)),
        link=link,
        # n_heads must come across: the predictor is built with J readouts either
        # way, but a model left at n_heads=1 takes the single-head forward and
        # plans with head 0 only, so a union-head run would be evaluated as if
        # the intervention were absent -- silently, with no shape error.
        n_heads=train_cfg.get("n_heads", 1),
        head_entropy_coef=train_cfg.get("head_entropy_coef", 0.0),
        burst_tau=train_cfg.get("burst_tau", 0.5),
    )
    _load_pose_dyn(model, Path(__file__).resolve().parent)
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def _member_spec(spec, default_epoch):
    """"<run_dir>" or "<run_dir>@<epoch>" -> (run_dir, epoch)."""
    run, _, epoch = str(spec).partition("@")
    return run, (epoch or default_epoch)


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_base_path = cfg_dict["ckpt_base_path"]
    # Plan-time consensus (planning/ensemble.py): `ensemble_members` lists M run dirs
    # (optionally "<run>@<epoch>") whose models vote on every CEM candidate. In that mode
    # `model_name` is only the label for the output dir, and the dataset / env / frameskip /
    # num_hist are read from the FIRST member -- every member must share them. Absent or
    # empty (the default in every conf/plan_*.yaml), this is exactly the single-model path.
    # NB utils.cfg_to_dict (utils.py:72) comma-joins every TOP-LEVEL list into a string,
    # so `ensemble_members=[a,b,c]` arrives here as "a,b,c"; accept both forms.
    _members = cfg_dict.get("ensemble_members") or []
    if isinstance(_members, str):
        _members = [m for m in _members.split(",") if m]
    members = [_member_spec(m, cfg_dict["model_epoch"]) for m in _members]
    if members:
        member_cfgs = [
            OmegaConf.load(os.path.join(f"{ckpt_base_path}/outputs/{run}", "hydra.yaml"))
            for run, _ in members
        ]
        model_cfg = member_cfgs[0]
    else:
        model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
        with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f)

    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"InfoJEPA_eval_{model_cfg.env.name}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    if members:
        from planning.ensemble import EnsembleWorldModel

        columns = [
            load_model(
                Path(f"{ckpt_base_path}/outputs/{run}") / "checkpoints" / f"model_{ep}.pth",
                c,
                c.num_action_repeat,
                device=device,
            )
            for (run, ep), c in zip(members, member_cfgs)
        ]
        model = EnsembleWorldModel(columns, [f"{r}@{e}" for r, e in members])
        print(f"Ensemble of {len(columns)} columns: {model.names}")
    else:
        model_ckpt = (
            Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
        )
        model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
