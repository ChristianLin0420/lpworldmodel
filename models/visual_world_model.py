import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange, repeat

from .heads import MLPHead


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
        # T1: let the reconstruction gradient reach the ENCODER. False keeps the
        # historical z_emb.detach(), under which 0/144 encoder params receive any
        # gradient from decoder_recon_loss (measured; see diary 2026-09-03 s13.4).
        decode_grad=False,
        var_space="u",
        var_gamma=1.0,   # VICReg hinge target: penalise per-dim std below this
        use_pose=False,  # bind the allocentric location signal to the action (TBT reference frame)
        # --- causal-objective knobs (V1-V3). All default-off and bit-identical when unset.
        incr_norm=False,     # V1: per-sample increment normalisation of the prediction loss
        act_info=0.0,        # V2: weight on InfoNCE over actions -> max I(a_t ; z_t+1 | z_t)
        act_info_k=4,        # V2: number of in-batch action negatives
        act_info_neg="perm", # P5: perm (q = p(a))  |  knn (q ~ p(a|z_t))
        ctrb_w=0.0,          # P4: weight on -logdet of the controllability Gramian
        ctrb_h=5,            # P4: horizon; = CEM's goal_H
        path_int=False,      # V3: learned path integration of the location signal
        path_int_w=1.0,      # V3: weight on the pose-prediction loss
        path_int_dims=None,  # V3: (raw_pose_dim, raw_action_dim), e.g. (4, 10) on pushT
        # --- T3: proprio-residual contact weighting. Inert at contact_gamma=0.0.
        contact_gamma=0.0,      # exponent on the weight; 0 => uniform (upstream)
        contact_shuffle=False,  # ESS-matched control: same weights, permuted
        contact_eps=1e-8,       # floor so a fully static transition is not weight 0
        contact_geom=None,      # (mean_x, mean_y, std_x, std_y, px_per_unit, radius_px)
        # --- T6: K-step option model. num_pred IS K; overshoot is its matched control.
        overshoot=False,        # chain the 1-step map K times instead of one K-step jump
        # --- T4: hindsight goal-conditioned value V(z, g). Inert at value_w=0.0.
        value_w=0.0,            # weight on the value loss; 0 => the head is not even built
        value_mode="td",        # td (T4) | mc (control a) | geom (control b)
        value_tau=0.7,          # expectile: the in-sample max over dataset actions
        value_gamma=0.98,       # discount; 1/(1-gamma) = 50 steps is the value floor
        value_hidden=None,      # head width; None => D
        value_ema=0.005,        # Polyak rate of the target network
        value_p_future=0.5,     # P(goal is an in-trajectory future frame vs cross-traj)
        # --- V4: reactive goal-conditioned policy a = pi(z, g). Inert at policy_w=0.0.
        policy_w=0.0,           # weight on the BC loss; 0 => the head is not even built
        act_dim_raw=None,       # RAW action width (2 * frameskip = 10 on pushT), NOT the
                                # 384-d embedding; required whenever policy_w > 0
        head_ckpt=None,         # plan-time only: restore value/policy heads from this file
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
        self.decode_grad = bool(decode_grad)
        self.var_space = var_space
        self.var_gamma = var_gamma
        self.use_pose = use_pose
        # V1-V3. The action is causally near-inert in this objective: measured across 9 arms,
        # changing the action moves the PREDICTED latent by 0.02-0.1% of its own magnitude
        # (d_state/d_action = 37x for the baseline), and that displacement is the strongest
        # predictor of CEM success we have (Spearman +0.81, p=0.0079). min ||h(P(z,a)) -
        # h(Enc(o'))||^2 is PREDICTIVE, not CAUSAL: nothing in it requires a to matter.
        self.incr_norm = bool(incr_norm)
        self.act_info = float(act_info)
        self.act_info_k = int(act_info_k)
        self.act_info_neg = str(act_info_neg)
        self.ctrb_w = float(ctrb_w)
        self.ctrb_h = int(ctrb_h)
        self.path_int_w = float(path_int_w)
        # T3. Weight each transition by the pixel change OUTSIDE the agent's own disc.
        # Proprio is a model INPUT, so the agent's occupancy is exactly known and the
        # residual is, to first order, the block. No privileged state anywhere.
        self.contact_gamma = float(contact_gamma)
        self.contact_shuffle = bool(contact_shuffle)
        self.contact_eps = float(contact_eps)
        self.contact_geom = tuple(float(x) for x in contact_geom) if contact_geom else None
        # T6. num_pred == K is the option length: the loss already pairs frame i with
        # frame i+K (z_tgt = z_emb[:, num_pred:]), so the only thing missing at K > 1 is
        # that the CONDITIONING must be the option's action rather than the first of the
        # K rows it spans. overshoot=True is the matched control: same 8-frame window,
        # same 25-env-step horizon, reached by chaining the 1-step map K times.
        self.overshoot = bool(overshoot)
        if self.overshoot and num_pred <= 1:
            # Loud: at K=1 "overshoot" is the default one-step loss, so an arm named
            # overshoot would silently be its own control (the V1-V3 failure mode).
            raise ValueError("overshoot=True is meaningless at num_pred=1; set num_pred=K>1")
        if self.contact_gamma != 0.0 and self.contact_geom is None:
            # Loud, not silent. A weighting arm that quietly falls back to uniform is
            # the path_int failure mode: it would look like a run and be a control.
            raise ValueError(
                "contact_gamma != 0 requires contact_geom="
                "(mean_x, mean_y, std_x, std_y, px_per_unit, radius_px)"
            )
        # Plain attributes, NOT buffers: train.py assembles VWorldModel from separately
        # prepared submodules and never calls model.to(device), so anything owned by the
        # model itself has to reach the device on its own (same rationale as pose_dyn /
        # path_int above). Caching by (device, dtype, H) keeps the state_dict untouched.
        self._contact_grid = None
        self._contact_gen = None
        self.path_int = None
        if path_int:
            # pose_{t+1} = f(pose_t, a_t). Initialised from the linear map fit on the dataset
            # (assets/pose_dynamics_pusht.pt, 95.2% of the 1-step pose change explained), then
            # trained jointly -- so location becomes action-DRIVEN rather than a static
            # side-channel, which is what the null PiWM-refframe arm was missing.
            # RAW dims, not the embedding dims: the constructor's proprio_dim/action_dim are
            # the EMBEDDING widths (10 / 384), while path integration operates on the raw
            # pose [x, y, vx, vy] (4) and the raw action (10 = 2 x frameskip).
            if not path_int_dims:
                raise ValueError("path_int=True requires path_int_dims=(pose_dim, act_dim)")
            _pd, _ad = int(path_int_dims[0]), int(path_int_dims[1])
            self.path_int = nn.Linear(_pd + _ad, _pd)
        # Non-persistent on purpose: checkpoints trained before this existed have no such
        # key, and a persistent buffer would make every one of them fail to load.
        self.register_buffer("pose_dyn", None, persistent=False)
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

        # --- T4 / V4: the value and policy heads --------------------------------------
        # Built HERE, after base_encoder is resolved, because their input width is 3*D
        # where D is base_encoder.emb_dim -- self.emb_dim above is the concat-path width
        # and is documented "Not used" on the adaln path this campaign runs.
        #
        # Inside fork_rng with a FIXED seed, and that is load-bearing twice over:
        #   * an unforked nn.Linear init advances the global CPU stream that RDMReg draws
        #     its target from, so PiWM-vp would differ from its control by the head AND by
        #     every subsequent random draw -- more than one factor, which is the whole
        #     point of a matched control;
        #   * a deterministic init makes "was this head ever optimised?" a decidable
        #     question: rebuild MLPHead under the same seed and compare against the
        #     checkpoint. A head that was built, never stepped and saved anyway (the
        #     path_int defect) is then visible on disk instead of invisible.
        self.value_w = float(value_w)
        self.value_mode = str(value_mode)
        self.value_tau = float(value_tau)
        self.value_gamma = float(value_gamma)
        self.value_ema = float(value_ema)
        self.value_p_future = float(value_p_future)
        self.policy_w = float(policy_w)
        self.act_dim_raw = int(act_dim_raw) if act_dim_raw else None
        self.value_head, self.value_target, self.policy_head = None, None, None
        # Where the heads' weights came from. None => fresh init. PolicyPlanner REFUSES
        # to plan on a fresh init, because that is exactly what _roll_pose did with
        # path_int and it produced a published conclusion that had to be retracted.
        self.heads_restored_from = None
        # Plain attribute, NOT a buffer: the state dict must stay byte-identical to
        # upstream or every checkpoint on disk fails to load.
        self._head_gen = None
        if self.value_mode not in ("td", "mc", "geom"):
            raise ValueError(f"value_mode must be td|mc|geom, got {self.value_mode!r}")
        if self.value_w > 0 or self.policy_w > 0:
            _D = int(base_encoder.emb_dim)
            _H = int(value_hidden) if value_hidden else _D
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260903)
                if self.value_w > 0:
                    self.value_head = MLPHead(3 * _D, _H, 1)
                    # deepcopy AFTER the head exists: the target must start EQUAL to the
                    # online net or the first bootstrap is a pure random-init regression.
                    self.value_target = copy.deepcopy(self.value_head).requires_grad_(False)
                if self.policy_w > 0:
                    if not self.act_dim_raw:
                        # Loud. A BC head with the wrong output width is either a shape
                        # error at train time or, if it happened to broadcast, an arm
                        # whose actions are not actions.
                        raise ValueError(
                            "policy_w > 0 requires act_dim_raw = the RAW action width "
                            "(2 * frameskip = 10 on pushT), not the action embedding dim"
                        )
                    self.policy_head = MLPHead(3 * _D, _H, self.act_dim_raw)
            print(f"T4/V4 heads: value={None if self.value_head is None else (3*_D,_H,1)} "
                  f"policy={None if self.policy_head is None else (3*_D,_H,self.act_dim_raw)}")
            # plan.py's load_model() restores a FIXED list of submodules (ALL_MODEL_KEYS)
            # and knows nothing about these heads, so at plan time they would otherwise be
            # freshly initialised -- the path_int defect, exactly. plan.py is owned by
            # another agent this wave, so the checkpoint arrives through an env var
            # instead of a new kwarg threaded through a file we must not edit.
            _hc = head_ckpt or os.environ.get("LPWM_HEAD_CKPT") or None
            if _hc:
                self.load_head_state(_hc)

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

    def _action_infonce(self, z_src, act_src, target, z_pred):
        """V2: maximise I(a_t ; z_{t+1} | z_t) -- the quantity CEM actually consumes.

        The true next latent must be better explained by the action that was TAKEN than by
        actions that were not. Negatives are free: permute act_src within the batch. This is
        the first intervention in this project that changes what the model must DISTINGUISH
        rather than what its code's marginal looks like, and it cannot be gamed by arbitrary
        action-dependence because the positive still has to match the true z_{t+1}.

        tau is the detached mean energy, the same scale-free trick the union entropy bonus
        uses: without it the softmax argument is O(MSE) and the distribution is uniform, so
        the term exerts no force.
        """
        E_pos = (z_pred - target).pow(2).mean(dim=(-1, -2))            # (b, t)
        Es = [E_pos]
        perms = self._action_negatives(z_src, act_src)
        for perm in perms:
            z_neg = self._link(self.predict(z_src, act_src[perm]))
            Es.append((z_neg - target).pow(2).mean(dim=(-1, -2)))
        E = torch.stack(Es, dim=-1)                                    # (b, t, K+1)
        tau = E.detach().mean().clamp_min(1e-8)
        logits = (-E / tau).flatten(0, 1)                              # (b*t, K+1)
        tgt = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, tgt)

    def _action_negatives(self, z_src, act_src):
        """P5: which distribution the InfoNCE negatives are drawn from.

        InfoNCE with K negatives from a proposal q bounds mutual information RELATIVE TO q.
        "perm" permutes the batch, i.e. q = p(a), the MARGINAL. PushT demonstrations are
        policy-generated, so a and z are dependent and the discriminator can win on
        PLAUSIBILITY -- "would this action occur at this state?" -- without learning any
        dynamics; what it bounds is I(a; (z_t, z_t+1)), not I(a; z_t+1 | z_t). Taking that
        shortcut costs code capacity, which is exactly what V2 measured: d_action maximal at
        0.699 while rho collapsed 0.448 -> 0.052.

        "knn" draws each negative from the K nearest z_t IN THE BATCH, i.e. actions taken at
        similar states, so q ~ p(a | z_t) and only the predicted EFFECT can discriminate.
        Zero extra parameters: one cdist on (b, b).
        """
        b = act_src.shape[0]
        if self.act_info_neg != "knn" or b <= self.act_info_k + 1:
            return [torch.randperm(b, device=act_src.device)
                    for _ in range(self.act_info_k)]
        with torch.no_grad():
            zf = z_src.detach().flatten(1)                       # (b, t*p*d)
            d = torch.cdist(zf, zf)
            d.fill_diagonal_(float("inf"))                       # never pick self
            nn_idx = d.topk(self.act_info_k, dim=1, largest=False).indices   # (b, K)
        return [nn_idx[:, k] for k in range(self.act_info_k)]

    def _ctrb_loss(self):
        """P4: -logdet of the H-step controllability Gramian of the linearised dynamics.

        CEM optimises a_{t:t+H-1} to reach a goal latent. For the state-augmented system
        s_k = [z_k; z_{k-1}; z_{k-2}],

            s_{k+1} = A_aug s_k + B_aug a_k,  A_aug companion,  B_aug = [B; 0; 0]
            C_H = [B_aug, A_aug B_aug, ..., A_aug^{H-1} B_aug],   W_c = C_H C_H^T

        and the effort to reach direction v is v^T W_c^-1 v. An ill-conditioned W_c means
        whole latent directions are unreachable by ANY action sequence, so the CEM objective
        is flat along them however good the 1-step prediction is -- which is exactly V3's
        result: healthy 1-step d_action, null planning.

        Computed once per step ON THE WEIGHTS, not per sample. W_c is rank-deficient in
        general, so the ridge is not optional -- an unridged logdet is -inf by construction.
        """
        pred = self._pred
        lags = getattr(pred, "lags", None)
        if lags is None or not hasattr(pred, "B"):
            return None                                # no linear core (lie / action_linear)
        A = [m.weight for m in lags]
        B = pred.B.weight
        if hasattr(pred, "W"):                         # mlp_var / ltv: fold the readout in
            A = [pred.W.weight @ Ak for Ak in A]
            B = pred.W.weight @ B
        D, H = A[0].shape[0], len(A)
        n = D * H
        M = torch.zeros(n, n, device=B.device, dtype=B.dtype)
        M[:D] = torch.cat(A, dim=1)
        eye = torch.eye(D, device=B.device, dtype=B.dtype)
        for k in range(1, H):
            M[k * D:(k + 1) * D, (k - 1) * D:k * D] = eye
        Baug = torch.zeros(n, B.shape[1], device=B.device, dtype=B.dtype)
        Baug[:D] = B
        blocks, cur = [Baug], Baug
        for _ in range(self.ctrb_h - 1):
            cur = M @ cur
            blocks.append(cur)
        C = torch.cat(blocks, dim=1)
        Wc = C @ C.T
        Wc = 0.5 * (Wc + Wc.T)
        # TRACE-NORMALISE. -logdet(W_c) alone is UNBOUNDED BELOW: it is minimised by
        # inflating ||A|| and ||B|| without limit, which costs the predictor nothing to
        # supply and everything to obey. Measured in production: ctrb ran +0.645 at init
        # to -58 within one epoch, a -0.58 contribution against a z_loss of 0.03-0.09,
        # and half the seeds reached rel_mse = 1.0 (predicting the mean). The optimiser
        # farmed the reward and abandoned prediction.
        #
        # What the objective is ABOUT is whether the reachable set SPANS -- the isotropy
        # of W_c, which is a property of its shape, not its size. Rescaling to tr = n
        # makes the quantity scale-invariant and BOUNDED BELOW BY ZERO, attained exactly
        # when every eigenvalue is 1, i.e. when the action reaches every latent direction
        # equally. Inflating the weights now buys nothing.
        n_f = float(n)
        tr = Wc.diagonal().sum()
        if not torch.isfinite(tr) or float(tr) <= 0.0:
            return None                     # W_c == 0: B still at its zero init
        Wn = Wc * (n_f / tr)                # tr(Wn) = n  =>  mean eigenvalue 1
        eye = torch.eye(n, device=Wc.device, dtype=torch.float32)
        Wf = Wn.float()
        for rel in (1e-6, 1e-4, 1e-2):
            L, info = torch.linalg.cholesky_ex(Wf + rel * eye)
            if int(info) == 0:
                # -logdet/n >= 0 by AM-GM, with equality iff the spectrum is flat
                return -2.0 * torch.log(L.diagonal().clamp_min(1e-30)).sum() / n_f
        return None                         # skip this step rather than kill the run

    def _path_int_loss(self, proprio, act):
        """V3: predict the next location from (location, action) -- TBT path integration.

        PiWM-refframe RECEIVED pose and was a null (-0.063, p=0.169). TBT requires the
        reference frame to be UPDATED BY THE MOVEMENT; that is the causal part we omitted.
        """
        if self.path_int is None or proprio is None or proprio.shape[1] < 2:
            return None
        x = torch.cat([proprio[:, :-1], act[:, :-1]], dim=-1)
        # train.py prepares encoder/predictor/... individually and assembles the model from
        # them, so a parameter created by VWorldModel itself never reaches the accelerator's
        # device. Move it once, lazily, rather than assuming a .to() that never happens.
        if self.path_int.weight.device != x.device:
            self.path_int.to(x.device)
        return (self.path_int(x) - proprio[:, 1:].detach()).pow(2).mean()

    # --- T4 / V4 -----------------------------------------------------------------------

    def load_head_state(self, ckpt, strict=True):
        """Restore the value/policy heads from a train.py checkpoint (path or payload).

        Exists because `plan.py:load_model` restores a FIXED list of submodules
        (`ALL_MODEL_KEYS`) and knows nothing about these heads. Without this, a plan-time
        rebuild would carry a freshly initialised head and every V4/V5 number would be a
        measurement of `torch.manual_seed(20260903)` -- which is precisely what
        `_roll_pose` did with `path_int` (diary 2026-09-03 s13.3).

        Prints a checksum of what was loaded, so "the head the planner uses is the head
        on disk" is a line in the log rather than an assumption.
        """
        if isinstance(ckpt, (str, os.PathLike)):
            payload = torch.load(str(ckpt), map_location="cpu")
        else:
            payload = ckpt
        pairs = []
        if self.value_head is not None:
            pairs.append(("value_head", self.value_head, True))
            # optional: a policy-only run has no value_target, and a value run that was
            # saved before the target existed should still load its online net
            pairs.append(("value_target", self.value_target, False))
        if self.policy_head is not None:
            pairs.append(("policy_head", self.policy_head, True))
        missing = [k for k, _m, req in pairs if req and not isinstance(payload.get(k), dict)]
        if missing:
            msg = (f"{missing} absent from {ckpt}: the head was never saved. Refusing to "
                   "plan on a fresh init (the path_int defect).")
            if strict:
                raise KeyError(msg)
            print("WARNING: " + msg)
            return
        for k, mod, _req in pairs:
            sd = payload.get(k)
            if isinstance(sd, dict):
                mod.load_state_dict(sd)
            elif k == "value_target" and self.value_head is not None:
                self.value_target.load_state_dict(self.value_head.state_dict())
                sd = None
            chk = sum(float(p.detach().abs().sum()) for p in mod.parameters())
            print(f"{k} restored from {ckpt}  sum|W|={chk:.6f}"
                  + ("" if sd is not None else "  (copied from value_head)"))
        self.heads_restored_from = str(ckpt)

    def _head_generator(self, device):
        """A PRIVATE RNG for hindsight goal sampling, cached per device.

        Not the global stream, and that is the whole point. RDMReg draws its target
        projections from the global stream LATER in the same forward, so sampling goals
        from it would move every subsequent random draw and PiWM-vp would differ from
        LpWM-ltv by the head AND by the regulariser's noise -- two factors, not one.
        With this generator the encoder/predictor trajectory is bit-identical to the
        control at the same seed, which is what makes this run's OWN CEM eval the control
        for V4 and V5. Same rationale (and same seed) as T3's _contact_gen.
        """
        if self._head_gen is None or self._head_gen.device != torch.device(device):
            g = torch.Generator(device=device)
            g.manual_seed(20260903)
            self._head_gen = g
        return self._head_gen

    def _v_floor(self):
        """-1/(1-gamma): the most pessimistic value an episode of -1 rewards can have."""
        return -1.0 / max(1.0 - self.value_gamma, 1e-6)

    def _hindsight_pairs(self, z):
        """(z_t, z_{t+1}, g, done, k, used_future) for T4's TD, from latents alone.

        Every future frame in a window is a valid goal with label -k, so the labels are
        FREE and fully self-supervised: no reward, no privileged state, only the data's
        own temporal order. -E[steps] is also invariant to any monotone reparametrisation
        of the latent, which is what makes it a different quantity from the ||z - g||^2
        that CEM already minimises (that one is ranked against the TRUE task distance at
        only Spearman +0.398, n=296).

        `f` is DETACHED: the encoder and predictor must be bit-identical to the control
        at the same seed, so this run's own CEM eval IS the control for V4/V5.

        Cross-trajectory goals (probability 1 - value_p_future) are what stop the head
        from being a within-window step counter: without them every goal is reachable in
        <= T-1 steps and the head never has to represent "far".
        """
        b, T = z.shape[0], z.shape[1]
        dev = z.device
        f = z.mean(dim=2).detach()                       # (b, T, D); p == 1 (CLS) -> exact
        ar = torch.arange(b, device=dev)
        gen = self._head_generator(dev)
        zt, zn, gs, dn, ks, uf = [], [], [], [], [], []
        for i in range(T - 1):                           # anchors 0..T-2
            k = torch.randint(1, T - i, (b,), device=dev, generator=gen)  # offset >= 1
            fut = f[ar, i + k]
            rnd = f[torch.randperm(b, device=dev, generator=gen),
                    torch.randint(0, T, (b,), device=dev, generator=gen)]
            useF = torch.rand(b, device=dev, generator=gen) < self.value_p_future
            g = torch.where(useF[:, None], fut, rnd)
            zt.append(f[:, i])
            zn.append(f[:, i + 1])
            gs.append(g)
            # terminal iff the goal IS the next frame: V(g, g) is then defined by r only
            dn.append((useF & (k == 1)).float())
            ks.append(k.float())
            uf.append(useF.float())
        return (torch.cat(zt), torch.cat(zn), torch.cat(gs),
                torch.cat(dn), torch.cat(ks), torch.cat(uf))

    def _value_loss(self, z):
        """T4: in-sample expectile TD on a hindsight goal-conditioned value.

        tau = 0.7 is the in-sample max: a residual from a transition that reached the
        goal FASTER than V expected is weighted 0.7 against 0.3 for the other side, so V
        approaches the best action the DATASET took from z without ever querying an
        out-of-distribution action (no actor, no max over a learned model).
        """
        if self.value_head is None:
            return None
        # Heads are built inside VWorldModel, which train.py never prepare()s and never
        # .to()s -- only encoder/predictor/decoder/action_encoder/proprio_encoder are
        # moved (train.py:688-768). Same rationale as path_int / pose_dyn above.
        if next(self.value_head.parameters()).device != z.device:
            self.value_head.to(z.device)
            self.value_target.to(z.device)
        zt, zn, g, done, k, useF = self._hindsight_pairs(z)
        v = self.value_head(zt, g).squeeze(-1)
        if self.value_mode == "mc":
            # control (a): hindsight MC regression to -k. No bootstrap, no target net --
            # isolates TD from "any learned scalar that correlates with progress".
            # Cross-trajectory goals have no known k, so they are masked out.
            return (((v + k) ** 2) * useF).sum() / useF.sum().clamp_min(1.0)
        if self.value_mode == "geom":
            # control (b): the SAME head regressed to the latent distance CEM already
            # uses, in units of one mean latent step. Isolates temporal structure from
            # "a smooth MLP reparametrisation of ||z - g||".
            s = (zn - zt).norm(dim=-1).mean().detach().clamp_min(1e-6)
            tgt = -((g - zt).norm(dim=-1) / s).clamp(max=-self._v_floor())
            return (v - tgt.detach()).pow(2).mean()
        with torch.no_grad():                            # r = -1 per model step
            y = (-1.0 + self.value_gamma * (1.0 - done)
                 * self.value_target(zn, g).squeeze(-1)).clamp(min=self._v_floor())
        u = y - v
        if self.training:
            # Polyak, before this batch's step -- a one-batch lag, which is what every
            # target-network implementation has. no_grad + detach: value_target params
            # have requires_grad=False and must stay leaves.
            with torch.no_grad():
                for p, q in zip(self.value_target.parameters(),
                                self.value_head.parameters()):
                    p.mul_(1.0 - self.value_ema).add_(self.value_ema * q.detach())
        return ((self.value_tau - (u < 0).float()).abs() * u.pow(2)).mean()

    def _policy_loss(self, z, act):
        """V4: goal-conditioned BC in latent space.

        `act` is the NORMALISED action the loader already provides
        (conf/train_rdmreg.yaml normalize_action: True), so this needs nothing the CEM
        objective does not already have. Latents are DETACHED, so the encoder/predictor
        remain bit-identical to the control at the same seed.

        If this head matches CEM through the same PlanEvaluator, the 300 x 30 x 10 =
        90,000 model rollouts CEM spends per episode buy nothing. The user has confirmed
        that outcome is acceptable and interesting, not a failure.
        """
        if self.policy_head is None:
            return None
        if next(self.policy_head.parameters()).device != z.device:
            self.policy_head.to(z.device)
        f = z.mean(dim=2).detach()                        # (b, T, D) LINKED, stop-grad
        b, T = f.shape[0], f.shape[1]
        dev = z.device
        ar = torch.arange(b, device=dev)
        gen = self._head_generator(dev)
        out, n = None, 0
        for i in range(T - 1):                            # anchors 0..T-2
            k = torch.randint(1, T - i, (b,), device=dev, generator=gen)
            g = f[ar, i + k]                              # hindsight goal
            term = (self.policy_head(f[:, i], g) - act[:, i].detach().float()).pow(2).mean()
            out = term if out is None else out + term
            n += 1
        return out / max(n, 1)

    def _contact_grid_for(self, H, device, dtype):
        key = (H, str(device), dtype)
        if self._contact_grid is None or self._contact_grid[0] != key:
            ax = torch.arange(H, device=device, dtype=dtype)
            # image row == sim y, image col == sim x: env/pusht/pusht_env.py:22 sets
            # positive_y_is_up = False (no flip) and :617 transposes pixels3d (x,y,c)
            # to (y,x,c). Verified offline against the dataset, see the T3 probe.
            yy = ax.view(H, 1).expand(H, H)
            xx = ax.view(1, H).expand(H, H)
            self._contact_grid = (key, yy, xx)
        return self._contact_grid[1], self._contact_grid[2]

    def _contact_weight(self, obs):
        """T3: per-transition weight = visual change proprio does NOT explain.

        The task is a tail: over one action row the block's displacement is exactly
        0.000 px in 48.1% of transitions and its median is 0.084 px, while the top 5%
        of transitions carry 77.9% of all its motion. A mean objective spends its
        gradient by frequency, so it spends almost all of it where nothing happens.

        ||dz|| cannot be the weight -- on a latent that ignores the block it upweights
        AGENT motion, which is circular. The agent's position is a model INPUT
        (obs["proprio"]), so its occupancy disc can be masked out exactly and what is
        left is, to first order, the block. Measured on 60 episodes / 1,494 transitions:
        Spearman(w, true block motion) = +0.889 (unmasked +0.799), 97.5% of the weight
        mass on block-moving transitions against an oracle's 99.9%, ESS/N = 0.359.

        Renormalised to unit mean, so this is a reweighting and not secretly a change
        of learning rate. Returns (b, num_hist) or None when the flag is off.
        """
        if self.contact_gamma == 0.0:
            return None
        v, pr = obs.get("visual"), obs.get("proprio")
        if v is None or pr is None or v.shape[1] < 2:
            return None
        with torch.no_grad():
            v = v.float()
            H = v.shape[-1]
            mx, my, sx, sy, scale, R = self.contact_geom
            px = pr[..., 0].float() * sx + mx                 # (b, T) sim x
            py = pr[..., 1].float() * sy + my                 # (b, T) sim y
            px, py = px * scale, py * scale                   # -> image pixels
            yy, xx = self._contact_grid_for(H, v.device, v.dtype)
            r2 = R * R
            d0 = (xx - px[:, :-1, None, None]) ** 2 + (yy - py[:, :-1, None, None]) ** 2
            d1 = (xx - px[:, 1:, None, None]) ** 2 + (yy - py[:, 1:, None, None]) ** 2
            keep = ((d0 > r2) & (d1 > r2)).to(v.dtype)        # (b, T-1, H, H)
            del d0, d1
            sq = (v[:, 1:] - v[:, :-1]).pow(2).mean(2)        # (b, T-1, H, H)
            r = (sq * keep).sum((-1, -2)) / keep.sum((-1, -2)).clamp_min(1.0)
            k = self.num_pred
            if k > 1:
                # the loss pairs frame i with frame i+K, so the weight for that pair is
                # the unexplained change accumulated over the whole option. K == 1 is
                # this expression's own identity, so the default path is untouched.
                r = torch.stack(
                    [r[:, i:i + k].sum(1) for i in range(r.shape[1] - k + 1)], dim=1
                )
            w = (r[:, : self.num_hist] + self.contact_eps).pow(self.contact_gamma)
            if self.contact_shuffle:
                # ESS-matched control: identical weight DISTRIBUTION (so identical
                # effective compute), alignment with contact destroyed. Its own
                # generator, so the global RNG stream -- which RDMReg draws its target
                # from -- is bit-identical to the unshuffled arm's.
                if self._contact_gen is None or self._contact_gen.device != w.device:
                    self._contact_gen = torch.Generator(device=w.device)
                    self._contact_gen.manual_seed(20260903)
                idx = torch.randperm(w.numel(), device=w.device, generator=self._contact_gen)
                w = w.reshape(-1)[idx].view_as(w)
            w = w / w.mean().clamp_min(1e-12)
        return w

    def _option_act(self, act_emb):
        """T6: the action of a K-step OPTION -- the mean of the K rows it spans.

        `num_pred` is K. Row i of `act_emb` is the embedding of the action taken from
        frame i to frame i+1, so a K=5 arm conditioned on row i alone would see 1/5 of
        the actions its target depends on and would be a null for a trivial reason.

        MEAN of the EMBEDDINGS, not a concatenation of the raw actions and not a sum.
        Concatenating would move action_encoder.patch_embed's fan_in from 10 to 50 and
        therefore its muP learning rate by 5x (models/mup.py: used_lr = base_lr *
        base_width / fan_in) -- the exact confound documented for use_pose below, so
        "jumpy" would silently also be "5x smaller action LR". Summing would multiply
        the AdaLN conditioning norm by K.

        At K == 1 this returns the INPUT OBJECT unchanged, so the default path is
        bit-identical (no new op, no new node in the graph).
        """
        k = self.num_pred
        if k <= 1:
            return act_emb
        # pad by holding the last row, so every row i in [0, T) has K rows to average.
        # The training slice never reads the pad: the largest row it touches is
        # num_hist-1+K-1 = 6 <= T-1 = 7 at num_hist=3, K=5.
        x = torch.cat([act_emb, act_emb[:, -1:].expand(-1, k - 1, -1)], dim=1)
        return torch.stack(
            [x[:, i : i + k].mean(1) for i in range(act_emb.shape[1])], dim=1
        )

    def _overshoot_rollout(self, z_emb, act_emb):
        """T6's matched control: the same horizon, reached by COMPOUNDING.

        Same num_pred=K window (identical data, identical window count, identical batch
        composition as the jumpy arm), but the 1-step map is chained K times from frame
        num_hist-1 with its own per-row actions, and every intermediate frame
        num_hist..num_hist+K-1 is supervised. So the horizon is held fixed and the only
        difference from PiWM-jump5 is whether error compounds through the K steps --
        which is exactly the attribution T6 claims.

        Returns (z_pred, target) both (b, K, p, d).
        """
        emb = z_emb[:, : self.num_hist]
        preds = []
        for _ in range(self.num_pred):
            # window slides exactly as the plan-time K=1 rollout does, so step j is
            # conditioned on act rows [j, j+num_hist) and predicts frame num_hist+j
            nxt = self._predict_next_adaln(emb, act_emb)   # (b, 1, p, d) linked
            preds.append(nxt)
            emb = torch.cat([emb, nxt], dim=1)
        z_pred = torch.cat(preds, dim=1)                   # (b, K, p, d)
        tgt = z_emb[:, self.num_hist : self.num_hist + self.num_pred]
        return z_pred, (tgt.detach() if self.detach_target else tgt)

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
        t = act_emb.shape[1]
        if proprio.shape[1] < t:
            proprio = self._roll_pose(proprio, act, t)
        return act_emb + self.encode_proprio(proprio[:, :t])

    def _roll_pose(self, proprio, act, t):
        """Extend observed pose to t steps by rolling the ENVIRONMENT's pose dynamics.

        At plan time obs_0 carries pose only for the num_hist observed frames, but the
        rollout window slides (_predict_next_adaln indexes act_emb_all[:, L-h:L]), so
        future frames need pose too. Holding the last observed pose is NOT a small
        approximation: measured on a trained checkpoint it moved the conditioning by
        0.595x its own scale, while the pose embedding is 1.27x the action embedding's
        norm -- a healthy model (rel_mse 0.0137) then scored 0.020 at CEM.

        pose_{k+1} = [pose_k, act_k, 1] @ W is fit ONCE on the dataset, because this is
        environment dynamics, not model dynamics -- so it needs no retraining and is
        shared by every arm. On pushT it explains 95.2% of the 1-step pose change and
        drifts only 7.6% of a pose sd over the planner's 5-step horizon (vs 22.9% at 8
        steps, so it should not be trusted much beyond goal_H).

        Falls back to hold-last when no map is loaded, so the path never crashes -- but
        that fallback is the broken behaviour above and is logged by the caller's config.
        """
        # The model's own learned head takes precedence: with V3 the pose dynamics is part
        # of the model, so rollout and training use the SAME map and cannot drift apart.
        if self.path_int is not None:
            if self.path_int.weight.device != proprio.device:
                self.path_int.to(proprio.device)
            out = [proprio[:, k] for k in range(proprio.shape[1])]
            for k in range(len(out), t):
                out.append(self.path_int(torch.cat([out[-1], act[:, k - 1]], dim=-1)))
            return torch.stack(out, dim=1)
        W = getattr(self, "pose_dyn", None)
        out = [proprio[:, k] for k in range(proprio.shape[1])]
        if W is None:
            out += [out[-1]] * (t - len(out))
            return torch.stack(out, dim=1)
        W = W.to(proprio.dtype).to(proprio.device)
        ones = torch.ones(proprio.shape[0], 1, dtype=proprio.dtype, device=proprio.device)
        for k in range(len(out), t):
            out.append(torch.cat([out[-1], act[:, k - 1], ones], dim=-1) @ W)
        return torch.stack(out, dim=1)
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b, t_frames = visual.shape[0], visual.shape[1]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        # Block-causal temporal attention encodes the CLIP jointly, so a frame's
        # representation depends on the current and past frames. The default path
        # encodes every frame independently (t folded into the batch), which is why
        # the encoder otherwise has no temporal structure whatsoever.
        # getattr on the UNWRAPPED module: accelerate's DDP wrapper only proxies
        # forward(), so reaching forward_temporal through it raises AttributeError --
        # the same trap documented for forward_heads on _pred above.
        _enc = getattr(self.encoder, "module", self.encoder)
        if getattr(_enc, "block_causal", False):
            visual_embs = _enc.forward_temporal(visual, t_frames)
        else:
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

        # T6. num_pred IS K. z_tgt below already pairs frame i with frame i+K for any
        # K, so the jump arm only needs the OPTION action (mean of the K rows the option
        # spans); _option_act is the identity object at K=1. The overshoot control keeps
        # the 1-step map and its per-row actions, so it must NOT see the option action.
        act_opt = act_emb if self.overshoot else self._option_act(act_emb)

        z_src = z_emb[:, : self.num_hist]       # (b, num_hist, p, d) linked input
        act_src = act_opt[:, : self.num_hist]   # (b, num_hist, act_emb_dim)
        z_tgt = z_emb[:, self.num_pred :]       # (b, num_hist, p, d)

        target = z_tgt.detach() if self.detach_target else z_tgt

        if self.n_heads > 1:
            z_pred, z_loss, head_logs = self._union_head_loss(z_src, act_src, target)
            loss_components.update(head_logs)
        else:
            if self.overshoot:
                # T6 control: K chained 1-step predictions, every intermediate frame
                # supervised. z_pred/target are (b, K, p, d) here, NOT (b, num_hist, ..).
                z_pred, target = self._overshoot_rollout(z_emb, act_emb)
            else:
                u_pred = self.predict(z_src, act_src)   # predictor output (pre-link)
                z_pred = self._link(u_pred)             # linked (SAME link -> tied threshold)
            if self.overshoot:
                # deliberately ahead of incr_norm / contact in this cascade: both index
                # their weights by num_hist and would silently broadcast against a
                # (b, K, p, d) error. Neither is combined with overshoot in any arm.
                z_loss = self.emb_criterion(z_pred, target)
            elif self.incr_norm:
                # V1. Loss "on the increment" is a NO-OP -- (z_pred - z_src) - (target -
                # z_src) = z_pred - target exactly -- and a batch-level normaliser only
                # rescales the term. Only PER-SAMPLE weighting changes gradient direction:
                # it stops frames with large autonomous motion from dominating, and those
                # are exactly where the action's relative contribution is smallest.
                # Renormalised to unit mean weight so the balance against reg_weight is
                # unchanged and this is not secretly a learning-rate change.
                d_true = (target - z_src).detach()
                w = 1.0 / (d_true.pow(2).mean(dim=(-1, -2), keepdim=True) + 1e-4)
                w = w / w.mean().clamp_min(1e-12)
                z_loss = ((z_pred - target).pow(2) * w).mean()
            elif (_cw := self._contact_weight(obs)) is not None:
                # T3. Both components are TENSORS (train.py:1096-1099 gathers them) and
                # are emitted ONLY when contact_gamma > 0, so the default component-name
                # set that tests/test_bit_identity.py pins is unchanged.
                z_loss = ((z_pred - target).pow(2) * _cw[..., None, None]).mean()
                loss_components["contact_ess"] = _cw.mean().pow(2) / _cw.pow(2).mean()
                loss_components["contact_w_max"] = _cw.max()
            else:
                z_loss = self.emb_criterion(z_pred, target)
            if self.act_info > 0:
                loss_components["act_info_loss"] = self._action_infonce(
                    z_src, act_src, target, z_pred)
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
            # for the causal diagnostic: d_action needs the predictor's own inputs, and
            # re-encoding them in train.py would cost a second ViT pass.
            "z_src": z_src.detach(),
            "act_src": act_src.detach(),
        }

        loss = z_loss
        loss_components["z_loss"] = z_loss
        # V2: the causal term. Added here, not folded into z_loss, so the two are separable
        # in the logs and an ablation can read them apart.
        if self.act_info > 0 and "act_info_loss" in loss_components:
            loss = loss + self.act_info * loss_components["act_info_loss"]
        # V3: path integration of the location signal.
        _pl = self._path_int_loss(obs.get("proprio"), act)
        if _pl is not None:
            loss = loss + self.path_int_w * _pl
            loss_components["path_int_loss"] = _pl
        if self.ctrb_w > 0:
            _cl = self._ctrb_loss()
            if _cl is not None:
                loss = loss + self.ctrb_w * _cl
                loss_components["ctrb_loss"] = _cl
        # T4 / V4. Both terms are gated on a weight whose default is 0.0, and the heads
        # are not even BUILT at that default, so no new loss_components key can appear at
        # defaults (tests/test_bit_identity.py asserts set(got) == set(want)) and both
        # values are TENSORS (train.py:1096-1099 gathers them).
        if self.value_w > 0:
            _vl = self._value_loss(z_emb)
            if _vl is not None:
                loss = loss + self.value_w * _vl
                loss_components["value_loss"] = _vl
        if self.policy_w > 0:
            _bl = self._policy_loss(z_emb, act)
            if _bl is not None:
                loss = loss + self.policy_w * _bl
                loss_components["policy_loss"] = _bl
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
            # T1. With decode_grad=False this is the campaign's historical behaviour:
            # the decoder is fit to a FROZEN code, so reconstruction constrains the
            # decoder and nothing else. With it True the pixel loss is the first
            # objective in this codebase whose optimum requires z to carry the block.
            z_dec = z_emb if self.decode_grad else z_emb.detach()
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


    def _predict_next_adaln(self, emb, act_emb_all, z_goal=None, rows=None):
        """Predict the next frame embedding from the last num_hist frames.
        emb: LINKED (b, L, p, d); act_emb_all: (b, T_total, a). Frame i is conditioned
        on action i (causal: output at frame j predicts frame j+1). Returns the next
        LINKED frame so the autoregression stays in the linked space (matches training).

        rows (T6): the action row that each latent in `emb` is conditioned on. An option
        model advances K rows per call, so after the first step the latents no longer sit
        at consecutive rows and `act_emb_all[:, L-h:L]` would read the wrong actions.
        rows=None reproduces that slice exactly, so the K=1 path is untouched."""
        L = emb.shape[1]
        h = min(self.num_hist, L)
        emb_win = emb[:, L - h :]  # (b, h, p, d) linked
        act_win = (
            act_emb_all[:, L - h : L] if rows is None else act_emb_all[:, rows[-h:]]
        )  # (b, h, a)
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
        # T6. The OVERSHOOT arm has num_pred=5 too, but it was trained as the 1-step map
        # chained K times with per-row actions -- so it must ROLL that way, or the arm
        # would be evaluated as a model it never was. This is the one T6 flag that has to
        # reach plan time, which is why conf/train_rdmreg.yaml puts overshoot inside the
        # `model:` block (plan.py rebuilds from train_cfg.model, not from the top level).
        K = 1 if self.overshoot else self.num_pred
        if K > 1:
            # T6. One predictor call advances K action rows, so the horizon is covered
            # in ceil(T_total / K) calls with NO compounding: at goal_H=5 with K=5 and
            # plan.py's single-frame obs_0 that is exactly ONE call (verified by a
            # forward hook in tests/test_arms.py).
            # rows tracks which action row each latent sits at; the option action is the
            # mean of the K rows the option spans (same object the loss is trained on).
            opt_emb = self._option_act(act_emb_all)
            rows = list(range(emb.shape[1]))
            while rows[-1] + K <= act.shape[1]:
                emb = torch.cat(
                    [emb, self._predict_next_adaln(emb, opt_emb, z_goal, rows=rows)],
                    dim=1,
                )
                rows.append(rows[-1] + K)
            # Re-index onto the ACTION-ROW axis, holding each option's endpoint over
            # the K-1 rows inside it, so the returned length is act.shape[1] + 1 exactly
            # as at K = 1. NOT cosmetic: planning/mpc.py:113 sets a FINITE
            # action_len = (iter+1) * n_taken_actions and evaluator._get_traj_last then
            # indexes the imagined rollout at that ROW, so a K-length tensor would raise
            # IndexError the first time any episode succeeds. Every index the planner
            # actually reads (-1, and multiples of n_taken_actions = goal_H = K) lands on
            # a real predicted frame; the held rows are the honest statement that an
            # option model makes no claim about its own interior.
            T = act.shape[1]
            idx = torch.zeros(T + 1, dtype=torch.long)
            for m, r in enumerate(rows):
                if r <= T:
                    idx[r:] = m
            emb = emb[:, idx.to(emb.device)]
        else:
            while emb.shape[1] < act.shape[1]:
                emb = torch.cat(
                    [emb, self._predict_next_adaln(emb, act_emb_all, z_goal)], dim=1
                )
            emb = torch.cat([emb, self._predict_next_adaln(emb, act_emb_all, z_goal)], dim=1)
        z = emb  # (b, T_total + 1, p, d) at K = 1
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