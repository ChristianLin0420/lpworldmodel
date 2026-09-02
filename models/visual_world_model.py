import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat


class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        action_conditioning="concat",  # "concat" (DINO-WM) or "adaln" (LeWM)
        regularizer=None,
        reg_weight=0.0,  # weight on the regularizer term
        detach_target=True,
        link=None,
        lamb_var=0.0,
        lamb_cov=0.0,
        lamb_decode=1.0,
        var_space="u",
        var_gamma=1.0,   # VICReg hinge target: penalise per-dim std below this
        use_pose=False,  # bind the allocentric location signal to the action (TBT reference frame)
        n_heads=1,
        head_entropy_coef=0.0,
        burst_tau=0.5,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.action_conditioning = action_conditioning
        assert action_conditioning in ("concat", "adaln"), (
            f"action_conditioning {action_conditioning} not supported."
        )
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.detach_target = detach_target
        self.link = link
        # Step 4 union head. n_heads=1 + coef=0 is upstream exactly.
        self.n_heads = n_heads
        self.head_entropy_coef = head_entropy_coef
        self.burst_tau = burst_tau
        self.lamb_var = lamb_var
        self.lamb_cov = lamb_cov
        self.lamb_decode = lamb_decode
        self.var_space = var_space
        self.var_gamma = var_gamma
        self.use_pose = use_pose
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat
        base_encoder = (
            encoder.module
            if isinstance(encoder, nn.parallel.DistributedDataParallel)
            else encoder
        )
        self.emb_dim = base_encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in base_encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * base_encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

        # Diagnostics-only views of the last forward, for train.py's live logging.
        # Plain attributes, NOT buffers, so the state dict stays byte-identical to
        # upstream and the bit-identity fixtures are unaffected.
        self._diag = None
        self._diag_heads = None

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act

    def _act_emb_with_pose(self, act, proprio):
        """Action embedding with the location signal bound to it (TBT reference frame).

        The adaln path drops obs["proprio"] at encode_obs(), so the agent pose -- which
        the loader materialises and moves to GPU every batch -- never reaches the model
        and self.proprio_encoder receives zero gradient while still being optimised and
        checkpointed. Here it is added to the action embedding, which is the only
        conditioning-shaped tensor the predictor consumes (infojepa_modules.py:555,566).

        ADDED rather than concatenated on purpose: concatenating pose onto the raw action
        would move action_encoder.patch_embed's fan_in from 10 to 14 and therefore its muP
        learning rate, so "adding pose" would silently also be "changing an LR" and the
        arm would not be single-factor.

        proprio may be SHORTER than act: at plan time obs_0 carries pose only for the
        num_hist observed frames while act runs the full horizon. The last observed pose
        is held for the remaining steps -- an explicit approximation, not an oversight.
        PushT actions are relative agent displacements, so pose is in principle
        integrable from them; that is a strictly better rollout and is left for later.
        """
        act_emb = self.encode_act(act)
        if not self.use_pose or proprio is None:
            return act_emb
        pose_emb = self.encode_proprio(proprio)            # (b, t_obs, d)
        t = act_emb.shape[1]
        if pose_emb.shape[1] < t:                          # hold the last observed pose
            pad = pose_emb[:, -1:].expand(-1, t - pose_emb.shape[1], -1)
            pose_emb = torch.cat([pose_emb, pad], dim=1)
        return act_emb + pose_emb[:, :t]
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        if self.action_conditioning == "adaln":
            return {"visual": visual_embs}

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z, act_emb=None):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
                act_emb: (b, num_hist, act_emb_dim), only for adaln conditioning
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        if self.action_conditioning == "adaln":
            return self.predictor(z, act_emb)  # (b, num_hist, p, d) pre-link


        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs.get("proprio"), # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.action_conditioning == "adaln":
            return {"visual": z}, None
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def _link(self, x):
        """Apply the RDMReg link h(.); identity when no link is configured."""
        return self.link(x) if self.link is not None else x

    def encode_obs_linked(self, obs):
        """Observation encoded into the LINKED representation space that the predictor,
        rollout, and planning operate in. Identical to encode_obs when no link is set
        (concat / DINO-WM, or identity link); applies h(.) to the visual embedding
        otherwise. Use this (not encode_obs) whenever the encoding is compared against
        rollout outputs (planning goal, rollout-divergence metrics) so both sides live
        in the same space. encode_obs stays raw for the training forward, which needs
        the pre-link u (rate loss) and links explicitly."""
        z = self.encode_obs(obs)
        z["visual"] = self._link(z["visual"])
        return z

    def _variance_loss(self, x):
        """VICReg variance hinge on (b, t, p, d): per-dim std over all (b,t,p) samples,
        then mean_d relu(gamma - std_d). Anti-collapse / balance term. On the pre-link u it
        anchors the dense code; on the post-link z it floors the used representation's spread.
        Validated on CIFAR-100 (topk_reg) as the var component of SWD + var + l1."""
        xf = rearrange(x, "b t p d -> (b t p) d")
        std = torch.sqrt(xf.var(dim=0) + 1e-4)
        return torch.relu(self.var_gamma - std).mean()

    def _covariance_loss(self, x):
        """VICReg covariance term on (b, t, p, d): decorrelate the d dims so the code stays
        full-rank (the piece missing from variance-alone, which rank-collapses). Pool (b,t,p)
        as N samples; penalize the off-diagonal of the dxd covariance. NB we divide by d(d-1)
        (a MEAN over the d(d-1) off-diagonal entries), NOT the VICReg-standard /d: the raw sum
        scales as d^2, so /d would grow ~d and (a) dwarf the O(1) variance floor and (b) make
        the term's effective strength width-dependent across the proj_dim sweep. /d(d-1) is O(1)
        and width-invariant, matching _variance_loss."""
        xf = rearrange(x, "b t p d -> (b t p) d")
        xf = xf - xf.mean(dim=0)
        d = xf.shape[-1]
        cov = (xf.T @ xf) / (xf.shape[0] - 1)             # (d, d) sample covariance
        off = cov - torch.diag(torch.diag(cov))
        return (off ** 2).sum() / (d * (d - 1))           # mean of squared off-diagonal covariances

    @property
    def _pred(self):
        """The predictor with any parallel wrapper stripped off.

        accelerate wraps the predictor in DistributedDataParallel (train.sh sets
        WORLD_SIZE/RANK, so a 1-process group is still created), and a DDP wrapper
        only proxies forward() -- reaching a custom entry point like forward_heads
        through it raises AttributeError. At world_size 1 the wrapper's allreduce is
        a no-op, so calling the inner module is numerically identical; with real
        DDP the union head would need its gradients synced explicitly, which is one
        more reason this project stays single-GPU per run.
        """
        return getattr(self.predictor, "module", self.predictor)

    def _union_head_loss(self, z_src, act_src, target):
        """Union head: J parallel readouts, loss = mean_{b,t} min_j L_j(b,t).

        Returns (z_pred_of_best_head, z_loss, logs).

        j* is per-(SAMPLE, TIMESTEP). That is the only granularity where the
        head-switch rate is well defined (num_hist-1 transitions per sample), where
        p_bar can average over (b,t), and which reduces exactly to the upstream
        emb_criterion at J=1 -- both are uniform means over the same elements.

        Each head is LINKED before the min, because the upstream loss is on the
        linked value and Pi = union_j supp(z_hat_j) needs post-link supports.
        """
        u_all = self._pred.forward_heads(z_src, act_src)       # (J,B,T,P,D) pre-link
        z_all = self._link(u_all)                              # same shared link
        # per-head, per-(sample, timestep) error: mean over patches and features
        per_head = ((z_all - target.unsqueeze(0)) ** 2).mean(dim=(-1, -2))  # (J,B,T)
        l_min, j_star = per_head.min(dim=0)                    # (B,T), (B,T)
        z_loss = l_min.mean()

        # gather the winning head's prediction for downstream metrics/plots
        idx = j_star[None, ..., None, None].expand(1, *z_all.shape[1:])
        z_pred = torch.gather(z_all, 0, idx).squeeze(0)         # (B,T,P,D)

        with torch.no_grad():
            usage = torch.nn.functional.one_hot(j_star, self.n_heads).float()
            p_bar = usage.mean(dim=(0, 1))                      # (J,)
            switch = (
                (j_star[:, 1:] != j_star[:, :-1]).float().mean()
                if j_star.shape[1] > 1
                else torch.zeros((), device=z_loss.device)
            )
            # burst = support-change spike, the same statistic Step 1 measures:
            # S = 1 - J_S(z_hat, z) above tau. tau is provisional (see plan open items).
            num = torch.minimum(z_pred, target).sum(-1)
            den = torch.maximum(z_pred, target).sum(-1).clamp_min(1e-8)
            burst = ((1.0 - num / den) > self.burst_tau).float().mean()

        # The entropy bonus needs a gradient, and one_hot(argmin) has none, so the
        # gradient rides on SOFT responsibilities q_j = softmax(-L_j / tau). The
        # temperature is the detached mean loss, which keeps the softmax argument O(1)
        # whatever the absolute MSE scale is (otherwise q is uniform, the entropy sits
        # at its maximum, and the bonus exerts no force). Hard p_bar above is what the
        # collapse precondition is judged on; this is only the optimization surrogate.
        tau_ent = per_head.detach().mean().clamp_min(1e-8)
        q = torch.softmax(-per_head / tau_ent, dim=0)          # (J,B,T)
        p_soft = q.mean(dim=(1, 2))                            # (J,)
        entropy = -(p_soft * (p_soft + 1e-12).log()).sum()

        # per-head views for train.py's head diagnostics (usage bars, loss
        # distribution, j* raster, per-head specialisation). Detached, so nothing
        # here holds an autograd graph alive past the backward.
        self._diag_heads = {
            "per_head": per_head.detach(),
            "j_star": j_star.detach(),
            "z_all": z_all.detach(),
        }

        logs = {
            "head_switch_rate": switch,
            "head_burst_rate": burst,
            "head_usage_max": p_bar.max(),
            "head_usage_entropy": entropy,
        }
        for j in range(self.n_heads):
            logs[f"head_usage_p{j}"] = p_bar[j]
        return z_pred, z_loss, logs

    def _forward_adaln(self, obs, act):
        """LeWM/RDMReg forward: AR predictor + AdaLN action conditioning, link h(.)
        on encoder & predictor outputs, no stop-grad (configurable), anti-collapse
        regularizer (SIGReg or RDMReg) + optional rate / variance / L1 loss, no decoder."""
        loss_components = {}
        u_emb = self.encode_obs(obs)["visual"]  # raw encoder output (b, num_frames, p, d)
        z_emb = self._link(u_emb)               # linked
        act_emb = self._act_emb_with_pose(act, obs.get("proprio"))  # (b, num_frames, act_emb_dim)

        z_src = z_emb[:, : self.num_hist]       # (b, num_hist, p, d) linked input
        act_src = act_emb[:, : self.num_hist]   # (b, num_hist, act_emb_dim)
        z_tgt = z_emb[:, self.num_pred :]       # (b, num_hist, p, d)

        target = z_tgt.detach() if self.detach_target else z_tgt

        if self.n_heads > 1:
            z_pred, z_loss, head_logs = self._union_head_loss(z_src, act_src, target)
            loss_components.update(head_logs)
        else:
            u_pred = self.predict(z_src, act_src)   # predictor output (pre-link)
            z_pred = self._link(u_pred)             # linked (SAME link -> tied threshold)
            z_loss = self.emb_criterion(z_pred, target)
            self._diag_heads = None

        # The encoder code and the loss's own target are not on the return path, and
        # they are what almost every live diagnostic is about (sparsity, predictive
        # Jaccard, per-support error). Stashing detached references costs no maths
        # and no copy; recomputing them in train.py would cost a second ViT pass.
        self._diag = {
            "u": u_emb.detach(),
            "z": z_emb.detach(),
            "z_pred": z_pred.detach(),
            "target": target.detach(),
        }

        loss = z_loss
        loss_components["z_loss"] = z_loss
        loss_components["z_visual_loss"] = z_loss  # CLS-only: all of z is visual

        if self.n_heads > 1 and self.head_entropy_coef > 0:
            # Maximize entropy of per-head usage: min_j alone is winner-take-all and
            # collapses to one head, which would make J=4 numerically identical to
            # J=1 and rig the gate to produce its own falsifying result.
            loss = loss - self.head_entropy_coef * loss_components["head_usage_entropy"]

        loss_components["l0_frac"] = (z_emb != 0).float().mean()

        with torch.no_grad():
            zf = rearrange(z_emb, "b t p d -> (b t p) d")
            loss_components["diag_cov_loss"] = self._covariance_loss(z_emb)
            loss_components["diag_var_perdim"] = zf.var(dim=0).mean()

        if self.regularizer is not None and self.reg_weight > 0:
            if hasattr(self.regularizer, "reg_loss"):  # RDMReg (sliced-Wasserstein)
                reg_loss = self.regularizer.reg_loss(z_emb, link=self.link)
            else:  # SIGReg (legacy char-function test): (t, b*p, d)
                reg_loss = self.regularizer(rearrange(z_emb, "b t p d -> t (b p) d"))
            loss = loss + self.reg_weight * reg_loss
            loss_components["reg_loss"] = reg_loss

        if self.lamb_var > 0:
            if self.var_space == "u":
                var_loss = self._variance_loss(u_emb)
            elif self.var_space == "z":
                var_loss = self._variance_loss(z_emb)
            else:  # "both"
                var_loss = 0.5 * (self._variance_loss(u_emb) + self._variance_loss(z_emb))
            loss = loss + self.lamb_var * var_loss
            loss_components["var_loss"] = var_loss

        if self.lamb_cov > 0:
            cov_loss = self._covariance_loss(z_emb)
            loss = loss + self.lamb_cov * cov_loss
            loss_components["cov_loss"] = cov_loss

        visual_reconstructed = None
        if self.decoder is not None and self.train_decoder:
            z_dec = z_emb.detach()                    # (b, num_frames, p, d), stop-grad
            dec_obs, _ = self.decode_obs({"visual": z_dec, "proprio": None})
            visual_reconstructed = dec_obs["visual"]  # (b, num_frames, 3, H, W)
            decoder_loss = self.decoder_criterion(visual_reconstructed, obs["visual"])
            loss = loss + self.lamb_decode * decoder_loss
            loss_components["decoder_recon_loss"] = decoder_loss

        loss_components["loss"] = loss
        return z_pred, None, visual_reconstructed, loss, loss_components

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        if self.action_conditioning == "adaln":
            return self._forward_adaln(obs, act)

        loss = 0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def _predict_next_adaln(self, emb, act_emb_all, z_goal=None):
        """Predict the next frame embedding from the last num_hist frames.
        emb: LINKED (b, L, p, d); act_emb_all: (b, T_total, a). Frame i is conditioned
        on action i (causal: output at frame j predicts frame j+1). Returns the next
        LINKED frame so the autoregression stays in the linked space (matches training)."""
        L = emb.shape[1]
        h = min(self.num_hist, L)
        emb_win = emb[:, L - h :]  # (b, h, p, d) linked
        act_win = act_emb_all[:, L - h : L]  # (b, h, a)
        if self.n_heads > 1 and z_goal is not None:
            return self._optimistic_next(emb_win, act_win, z_goal)
        pred = self.predictor(emb_win, act_win)  # (b, h, p, d) pre-link
        return self._link(pred[:, -1:])  # (b, 1, p, d) linked

    def _optimistic_next(self, emb_win, act_win, z_goal):
        """Union-head rollout step: advance with the head whose prediction lands
        nearest the goal latent.

        Training says the model is right if ANY head is right (loss = min_j L_j),
        so at plan time -- where there is no target to run that argmin against --
        the corresponding rule is a min over heads too, against the goal. Using
        head 0 instead would advance a J-head model with the head that owns only
        ~1/J of transitions, so the arm would lose for a reason unrelated to
        multimodality. Optimism does not flatter the arm either: CEM's success is
        scored by replaying the chosen actions in the real environment, so an
        over-optimistic model simply picks worse actions.

        Greedy per step, i.e. O(J) instead of the O(J^H) min over head sequences.
        """
        u_all = self._pred.forward_heads(emb_win, act_win)        # (J,b,h,p,d)
        z_all = self._link(u_all[:, :, -1:])                     # (J,b,1,p,d)
        g = z_goal if z_goal.dim() == z_all.dim() - 1 else z_goal.unsqueeze(1)
        d = ((z_all - g.unsqueeze(0)) ** 2).mean(dim=(-1, -2))   # (J,b,1)
        idx = d.argmin(dim=0)[None, ..., None, None].expand(1, *z_all.shape[1:])
        return torch.gather(z_all, 0, idx).squeeze(0)            # (b,1,p,d)

    def _rollout_adaln(self, obs_0, act, z_goal=None):
        act_emb_all = self._act_emb_with_pose(act, obs_0.get("proprio"))  # (b, T_total, a)
        emb = self._link(self.encode_obs(obs_0)["visual"])  # (b, n, p, d) linked
        while emb.shape[1] < act.shape[1]:
            emb = torch.cat(
                [emb, self._predict_next_adaln(emb, act_emb_all, z_goal)], dim=1
            )
        emb = torch.cat([emb, self._predict_next_adaln(emb, act_emb_all, z_goal)], dim=1)
        z = emb  # (b, T_total + 1, p, d)
        z_obses, _ = self.separate_emb(z)
        return z_obses, z

    def rollout(self, obs_0, act, z_goal=None):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
                z_goal: (b, p, d) or (b, 1, p, d) linked goal embedding. Only used
                  when n_heads > 1, to pick which head advances each step; see
                  _optimistic_next. Ignored at n_heads == 1, so the default path
                  is bit-identical to upstream.
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        if self.action_conditioning == "adaln":
            return self._rollout_adaln(obs_0, act, z_goal)

        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z