import os
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
        trace_file=None,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        # --- per-episode traces (M2) ------------------------------------------
        # Every eval before this one collapsed to a mean at the moment it was
        # computed, so no archived run can be re-scored without re-running it on a
        # GPU.  Traces are therefore ON BY DEFAULT and written next to the run's
        # other outputs (hydra has already chdir'd into the run dir).
        #
        # NB the prefix is read from the environment rather than threaded through
        # plan.py / conf as the spec suggested, because this wave does not own
        # plan.py or conf/**.  `trace_file=` is still accepted as a ctor kwarg, so
        # a later wave can add the conf key without touching this file again.
        #   LPWM_TRACE_FILE=""      -> disable
        #   LPWM_TRACE_FILE=/tmp/tr -> write /tmp/tr_<filename>.npz
        if trace_file is None:
            trace_file = os.environ.get("LPWM_TRACE_FILE", "traces")
        self.trace_file = trace_file or None

        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        with torch.no_grad():
            # same goal-conditioned head selection the planner used, so the
            # reported wm rollout matches the one the actions were chosen under
            # (z_g is hoisted out of the call only so the trace can reuse it; the
            #  kwarg was evaluated before the call anyway, so this is a no-op)
            z_g = self.wm.encode_obs_linked(trans_obs_g)["visual"]
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
                z_goal=z_g,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes, eval_results, per_ep = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
            z_g=z_g,
        )

        # TERMINAL metrics: the state at the END of the executed rollout, NOT
        # truncated at action_len.  `logs` above is the LATCHED view -- for an
        # episode that hit `success` at MPC iter j, e_final_state is frozen at iter
        # j and mpc.py then ORs it forward, so the headline is a max over up to ten
        # draws.  These keys are the un-latched terminal outcome.
        e_terminal_state = e_states[:, -1]
        term = self.env.eval_state(self.state_g, e_terminal_state)
        logs.update(
            {
                f"terminal_{'success_rate' if k == 'success' else 'mean_' + k}": float(
                    np.mean(np.asarray(v, dtype=float))
                )
                for k, v in term.items()
            }
        )

        # CEM calls eval_actions once per opt step (cem.py:125, eval_every=1,
        # opt_steps=30) -> ~300 evals per run.  Only the checkpoint evals -- the MPC
        # iterations and the final eval, i.e. the ones that also save video -- get a
        # trace, unless LPWM_TRACE_ALL=1 asks for every one.
        if save_video or os.environ.get("LPWM_TRACE_ALL", "0") == "1":
            self._write_trace(
                filename=filename,
                action_len=action_len,
                e_terminal_state=e_terminal_state,
                e_latched_state=e_final_state,
                term=term,
                latched=eval_results,
                per_ep=per_ep,
            )

        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )
        elif save_video:
            self._plot_rollout_real(
                e_visuals_raw=e_obses["visual"],
                goal_raw=self.obs_g["visual"],
                successes=successes,
                action_len=action_len,
                filename=filename,
            )

        return logs, successes, e_obses, e_states

    def _plot_rollout_real(self, e_visuals_raw, goal_raw, successes, action_len,
                           filename="", n_frames=6):
        """Decoder-free planning viz: save a real-env key-frame strip per trajectory.
        e_visuals_raw: (b, t, h, w, c) raw env renders; goal_raw: (b[,1], h, w, c) raw goal.
        Writes {filename}_real_{idx}_{success|failure}.png (strip) and .gif (full rollout)."""
        to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        e = to_np(e_visuals_raw)
        g = to_np(goal_raw)

        def u8(a):
            a = np.asarray(a, dtype=np.float32)
            if a.max() <= 1.0 + 1e-3:
                a = a * 255.0
            return np.clip(a, 0, 255).astype(np.uint8)

        n = min(self.n_plot_samples, e.shape[0])
        for idx in range(n):
            gf = g[idx]
            while gf.ndim > 3:            # (1,h,w,c) -> (h,w,c)
                gf = gf[0]
            T = e.shape[1]
            if action_len is not None and np.isfinite(action_len[idx]):
                T = min(T, int(action_len[idx] * self.frameskip) + 1)
            sel = np.unique(np.linspace(0, T - 1, min(n_frames, T)).round().astype(int))
            tag = "success" if successes[idx] else "failure"

            def goalify(img):
                """Mark a frame as the GOAL so it clearly reads as the target, not another
                executed frame: (1) replace the near-white background with a distinct grey, and
                (2) lay a light grey wash over the whole panel, plus a dark grey border."""
                out = np.array(img, dtype=np.float32)
                bg = out.min(axis=-1) > 235          # near-white env background
                out[bg] = 170                        # -> clearly grey background
                out = 0.82 * out + 0.18 * 150.0      # light grey wash over the whole panel
                out = np.clip(out, 0, 255).astype(np.uint8)
                b = 8                                 # dark grey border
                out[:b, :] = 110; out[-b:, :] = 110; out[:, :b] = 110; out[:, -b:] = 110
                return out

            goal_panel = goalify(u8(gf))
            sep = np.full((gf.shape[0], 12, gf.shape[2]), 110, dtype=np.uint8)
            strip = np.concatenate([u8(e[idx, t]) for t in sel] + [sep, goal_panel], axis=1)
            imageio.imwrite(f"{filename}_real_{idx}_{tag}.png", strip)
            try:
                frames = [np.concatenate([u8(e[idx, t]), sep, goal_panel], axis=1)
                          for t in range(T)]
                imageio.mimsave(f"{filename}_real_{idx}_{tag}.gif", frames, fps=12)
            except Exception as ex:
                print(f"[viz] gif skip traj {idx}: {ex}")

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs, z_g=None):
        """
        Args
            e_state
            e_obs
            i_z_obs
            z_g: encoded goal (only used to build per-episode traces)
        Return
            logs
            successes
            eval_results: the RAW per-episode dict, before it is averaged away
            per_ep: dict of per-episode arrays for the trace file
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        # NB the two norms above take axis=1, which is the SINGLETON time axis, so
        # they are elementwise |diff| arrays and their means are mean |diff| per
        # pixel -- not per-episode distances.  Those log values are historical and
        # are left exactly as they are; the honest per-episode versions are computed
        # here (batch axis kept, every other axis reduced) for the trace, and must be
        # taken BEFORE e_obs is overwritten by the transformed/cuda version below.
        def _np_ep_norm(a):
            a = np.asarray(a, dtype=np.float64)
            return np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)

        per_ep = {
            "visual_dist": _np_ep_norm(e_obs["visual"] - self.obs_g["visual"]),
            "proprio_dist": _np_ep_norm(e_obs["proprio"] - self.obs_g["proprio"]),
        }

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs_linked(e_obs)  # linked, to match i_z_obs (rollout)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
        })
        if "proprio" in e_z_obs and "proprio" in i_z_obs:
            logs["mean_div_proprio_emb"] = torch.norm(
                e_z_obs["proprio"] - i_z_obs["proprio"]
            ).item()

        # per-episode versions of the latent quantities the logs above collapse.
        # `mean_div_visual_emb` is a WHOLE-BATCH torch.norm, not a mean of
        # per-episode norms, so d_pred/d_real are strictly more informative.
        per_ep["div_visual_emb"] = self._per_ep_norm(
            e_z_obs["visual"] - i_z_obs["visual"]
        )
        if z_g is not None:
            per_ep["d_pred"] = self._per_ep_norm(i_z_obs["visual"] - z_g)
            per_ep["d_real"] = self._per_ep_norm(e_z_obs["visual"] - z_g)

        return logs, successes, eval_results, per_ep

    @staticmethod
    def _per_ep_norm(t):
        """||.|| over every axis but the batch axis, as a numpy array."""
        t = t.reshape(t.shape[0], -1)
        return torch.norm(t.float(), dim=1).detach().cpu().numpy()

    def _write_trace(
        self,
        filename,
        action_len,
        e_terminal_state,
        e_latched_state,
        term,
        latched,
        per_ep,
    ):
        """Persist the per-episode record of one eval so it never has to be re-run
        on a GPU to be re-scored.  Best-effort: a trace failure must never take an
        eval down with it."""
        if not self.trace_file:
            return
        try:
            payload = dict(
                seed=np.asarray(self.seed),
                state_0=np.asarray(self.state_0),
                state_g=np.asarray(self.state_g),
                action_len=np.asarray(action_len, dtype=float),
                e_state_final=np.asarray(e_terminal_state),
                e_state_latched=np.asarray(e_latched_state),
            )
            payload.update({k: np.asarray(v) for k, v in term.items()})
            payload.update({f"latched_{k}": np.asarray(v) for k, v in latched.items()})
            payload.update({k: np.asarray(v) for k, v in per_ep.items()})
            np.savez_compressed(f"{self.trace_file}_{filename}.npz", **payload)
        except Exception as ex:
            print(f"[trace] failed to write {self.trace_file}_{filename}.npz: {ex}")

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
