#!/bin/bash
# Submit one gate's arms x seeds, each as its own chained single-GPU run.
#
# Every arm carries its matched upstream control at the same predictor, D, HP and
# seed, so each contrast is paired. Steps 3 and 4 share a predictor (ltv) and
# therefore share their control arm (LpWM-ltv). Submitting
# both gates reuses those three runs rather than retraining them, because
# submit_until_done.sh exits early when the DONE sentinel already exists.
#
# Usage:
#   scripts/run_campaign.sh sparse            # 3 arms x 3 seeds
#   scripts/run_campaign.sh gate            # 4 arms x 3 seeds
#   scripts/run_campaign.sh union            # 3 arms x 3 seeds
#   scripts/run_campaign.sh gate union      # both, sharing the control arm
#   scripts/run_campaign.sh wave2           # 3 arms x 3 seeds (task-sensitivity wave)
#   EVAL=1 scripts/run_campaign.sh sparse     # CEM eval trained arms instead of training
#
#   KWTA_MATCHED=<k>  k for the matched-rho arm; REQUIRED for sparse, since it comes
#                     from the probe's measured l0_frac (rho * D). No default: guessing
#                     it would silently turn the matched control into a second tight arm.
#   SEEDS="0 1 2"     seeds (default "0 1 2"); DRYRUN=1; NEVALS=50; WINDOWS=3
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; fi

SEEDS=${SEEDS:-"0 1 2"}
NEVALS=${NEVALS:-50}
WINDOWS=${WINDOWS:-4}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
# namespaced on purpose: a bare D is commonly already exported in a login shell,
# and inheriting one would silently corrupt every run name below
PROJ_D=${PROJ_D:-384}

# cls  -> vit_scratch       : num_patches=1, the predictor sees ONE token per frame
# patch-> vit_scratch_patch : num_patches=(img/patch)^2 = 256 tokens, each with a position
# This was the hardcoded literal "cls" in the two submit lines below. It is a knob because
# TBT is about a POPULATION of columns and at num_patches=1 there is no population to test:
# LinearDynamicsPredictor has no positional embedding and no per-patch parameters, so its
# whole operator is broadcast over a patch axis of size 1.
FEATURE=${FEATURE:-cls}
case "${FEATURE}" in cls|patch) ;; *) echo "FEATURE must be cls|patch" >&2; exit 1 ;; esac
# The run name carries it, so a patch arm and a cls arm can never collide in one dir or be
# silently compared -- the same reason PRECISION is in the name.
FEAT_TAG=""; [ "${FEATURE}" != "cls" ] && FEAT_TAG="_${FEATURE}"

# Set once here, for every arm including the controls. bf16 perturbs numerics, so
# an arm that differed in precision would confound it with the intervention; the
# value is chosen from the probe wall-clock (3.9h/epoch in fp32 on A100).
PRECISION=${PRECISION:-bf16}
export PRECISION

# Must stay under train_slurm.sbatch's cpus-per-task (30), leaving cores for the
# main process and the GPU feed. train.sh otherwise defaults to 20, which was
# oversubscribing the old 16-CPU allocation and thrashing the dataloader.
WORKERS=${WORKERS:-24}

# One project for the whole campaign so the arms are comparable in a single wandb
# workspace. train.py sets name=<arm>/s<seed>, group=<arm> and job_type=<gate>, so
# grouping by "group" gives one row per arm with a seed band.
WANDB_PROJECT=${WANDB_PROJECT:-PiWM-pushT}
export WANDB_PROJECT

# CANARY=1: one short single-seed smoke run per arm before the 8-seed wave is queued.
# Rule 6 of the campaign ("canary before wave") needs a real GPU, a real dataloader and a
# real checkpoint -- the submit host has no GPU and CPU tests cannot answer "is the new
# term live on REAL data" or "is the new parameter in a checkpoint ON DISK". So it goes
# through this launcher like everything else, with three differences and no others:
#   * the run name is PREFIXED, so a canary can never collide with the real run dir and
#     never joins its wandb group (train.py derives group= from the run dir name)
#   * one 4h window instead of WINDOWS, one epoch instead of EPOCHS
#   * env.dataset.n_rollout=200 -> ~315 train steps/epoch at batch 64 (of 1,981,721
#     windows), and save_every_x_min=1 so a checkpoint exists within the first minute
# OVERRIDES is EXPORTED because that is how it reaches train.sh: sbatch --export=ALL
# passes the submitting shell's environment through train_slurm.sbatch.
CANARY_PREFIX=""
if [ "${CANARY:-0}" = "1" ]; then
    CANARY_PREFIX="CANARY-"
    WINDOWS=1
    EPOCHS=${EPOCHS:-1}
    OVERRIDES="${OVERRIDES:-} env.dataset.n_rollout=${CANARY_ROLLOUTS:-200} training.save_every_x_min=1"
    export OVERRIDES
    echo "=== CANARY mode: prefix='${CANARY_PREFIX}' windows=1 epochs=${EPOCHS} overrides='${OVERRIDES}'"
fi

# arm -> "PREDICTOR REG_WEIGHT MUP_LR <extra env assignments...>"
# HPs are the reproduce_pusht.sh sparse table entries for that predictor at D=384.
declare -A ARMS
declare -A ORDER
# arm -> "LINK TARGET_P". These are POSITIONAL args to train.sh, so they cannot ride
# along in the extra-env field above; the dense LeWM control is the only arm that
# overrides the sparse default (reprelu, p=1).
declare -A ARM_LINK
# arm -> "cls" | "patch". FEATURE is a whole-invocation setting, but wave23 crosses
# (tokens) x (decoder), so the patch cell has to live in the SAME wave as the cls cells
# or the 2x2 cannot be launched (and read) as one object. Unset => the global FEATURE,
# so every existing wave composes exactly as before.
declare -A ARM_FEAT

sparse_arms() {
    : "${KWTA_MATCHED:?set KWTA_MATCHED=<k> from the measured rho of the probe (k = round(rho*D))}"
    ORDER[sparse]="LpWM-base PiWM-sparse-matched PiWM-sparse-2pct"
    ARMS[LpWM-base]="mlp_var 0.1 5e-4"
    ARMS[PiWM-sparse-matched]="mlp_var 0.1 5e-4 KWTA_K=${KWTA_MATCHED}"
    ARMS[PiWM-sparse-2pct]="mlp_var 0.1 5e-4 KWTA_K=$(python -c "print(round(0.02*${PROJ_D}))")"
}

# ltv gate factorized into input x normalization; (magnitude, sigmoid) is upstream.
gate_arms() {
    ORDER[gate]="LpWM-ltv PiWM-gate-sup-sigmoid PiWM-gate-mag-softmax PiWM-gate-sup-softmax"
    ARMS[LpWM-ltv]="ltv 1.0 5e-4"
    ARMS[PiWM-gate-sup-sigmoid]="ltv 1.0 5e-4 GATE_INPUT=support"
    ARMS[PiWM-gate-mag-softmax]="ltv 1.0 5e-4 GATE_NORM=softmax"
    ARMS[PiWM-gate-sup-softmax]="ltv 1.0 5e-4 GATE_INPUT=support GATE_NORM=softmax"
}

# LpWM-ltv is the J=1 arm: identical config, so it is deliberately not retrained.
union_arms() {
    ORDER[union]="LpWM-ltv PiWM-union4 PiWM-union4-entropy"
    ARMS[LpWM-ltv]="ltv 1.0 5e-4"
    ARMS[PiWM-union4]="ltv 1.0 5e-4 N_HEADS=4 HEAD_ENT=0.0"
    ARMS[PiWM-union4-entropy]="ltv 1.0 5e-4 N_HEADS=4 HEAD_ENT=${HEAD_ENT:-0.1}"
}

# Wave 2: does PushT discriminate a representation change AT ALL? PushT's own state
# has participation ratio ~4.3, so the healthy arms' effective dim of 15-18 is already
# 3.5-4x over-complete. If forcing the code far below that does not move CEM success,
# no gate in this campaign is measuring anything.
wave2_arms() {
    local K; K=$(python -c "print(round(0.02*${PROJ_D}))")
    ORDER[wave2]="PiWM-sparse-2pct LeWM-ltv PiWM-union4-kwta8"
    # k = round(0.02*D): the SDR-canonical sparsity, far below the task dimension.
    ARMS[PiWM-sparse-2pct]="mlp_var 0.1 5e-4 KWTA_K=${K}"
    # The DENSE control (LeWM): identity link + Gaussian target. Reproduces the
    # sparse-vs-dense claim inside OUR pipeline, which has never been checked here.
    ARMS[LeWM-ltv]="ltv 0.1 5e-4"
    ARM_LINK[LeWM-ltv]="identity 2"
    # Union head AT k-WTA sparsity. A 4-way union is 88.7% ON at rho=0.42 (saturated,
    # carries nothing) but 7.8% at 2% -- separating "unions do not help" from "the
    # union ran at an operating point where it could not help".
    ARMS[PiWM-union4-kwta8]="ltv 1.0 5e-4 N_HEADS=4 KWTA_K=${K}"
}

# wave3: the HIGH-POWER gate round. Exactly three arms -- the control, the Step 3
# proposal, and its repair -- so a paired contrast is available on one substrate.
#
# Deliberately NOT ORDER[gate]: that has four arms, so running it at extra seeds
# would also submit PiWM-gate-sup-sigmoid, which nothing here asks about.
#
# PiWM-gate-both feeds the gate [z ; 1[z>0]] -- support AND magnitude. Step 3 gates
# on 1[z>0] alone, which is a deterministic function of z, so by the data-processing
# inequality I(supp;Y) <= I(z;Y): support gating is bounded ABOVE by magnitude gating
# and can only win as an inductive bias. Measured on trained codes, binarising
# discards 76% of per-unit bits and keeps 63% of the one-step predictive information.
# "both" makes the gate input a strict SUPERSET, removing that bound.
wave3_arms() {
    ORDER[wave3]="LpWM-ltv PiWM-gate-sup-softmax PiWM-gate-mag-softmax PiWM-gate-both"
    ARMS[LpWM-ltv]="ltv 1.0 5e-4"
    ARMS[PiWM-gate-sup-softmax]="ltv 1.0 5e-4 GATE_INPUT=support GATE_NORM=softmax"
    # The magnitude@softmax cell. gate-both is (both + softmax) but the control
    # LpWM-ltv is (magnitude + sigmoid), so "both vs control" confounds the gate_input
    # repair with the -0.067 softmax nuisance. With this arm the ladder is
    # support -> magnitude -> both at FIXED normalisation, and (both - mag) isolates
    # the only open question: does the support add value once it costs nothing?
    ARMS[PiWM-gate-mag-softmax]="ltv 1.0 5e-4 GATE_NORM=softmax"
    ARMS[PiWM-gate-both]="ltv 1.0 5e-4 GATE_INPUT=both GATE_NORM=softmax"
}

# wave4: does preventing code death rescue the union head? PiWM-union4 ends at
# rho=0.0000 / effective_dim 0.0 because min_j L_j admits z==0 as a GLOBAL optimum
# (every head is then exactly right). No aggregator change removes that -- min, mean
# and softmin are all 0 there -- so the repair has to make a dead code expensive.
#
# var_gamma=0.2 is calibrated, not the VICReg default: healthy per-dim std(z) here is
# 0.45-0.49, so gamma=1.0 would be always-on at full strength and would rescale the
# code rather than floor it, turning the control into a different model.
# VAR_SPACE=z is load-bearing: on pre-link u an all-negative code still has full
# spread, so the floor is SATISFIED by exactly the failure it must prevent.
wave4_arms() {
    ORDER[wave4]="LpWM-ltv-vfloor PiWM-union4-vfloor PiWM-kwta8-J1"
    ARMS[LpWM-ltv-vfloor]="ltv 1.0 5e-4 LAMB_VAR=1.0 VAR_GAMMA=0.2 VAR_SPACE=z"
    ARMS[PiWM-union4-vfloor]="ltv 1.0 5e-4 N_HEADS=4 LAMB_VAR=1.0 VAR_GAMMA=0.2 VAR_SPACE=z"
    # J=1 + ltv + k-WTA(k=8): the control that never existed. Without it,
    # PiWM-union4-kwta8 = 0.00 is fully explained by the k-WTA main effect and
    # attributes NOTHING to the union head.
    ARMS[PiWM-kwta8-J1]="ltv 1.0 5e-4 KWTA_K=8"
}

# wave5 (run with PROJ_D=2048): the SDR-regime test. PiWM-sparse-2pct put k-WTA at 2%
# of D=384 -> w=8 active units at n=384, which is outside Numenta's viable SDR band
# (n=2048-10000, w=10-40) on BOTH axes. "Sparsity hurts" was therefore never measured
# on an actual SDR. At D=2048, 2% gives w=41 -- in band. LpWM-ltv-d2048 is the dense
# control that separates width from sparsity.
# wave6/7: the WIDTH-vs-LEARNING-RATE factorial.
#
# LpWM-ltv-d2048 (median 0.690) beat LpWM-ltv (0.380), but the two differ in TWO ways,
# not one. models/mup.py:57 sets used_lr = base_lr * base_width / fan_in with base_width
# pinned at 384, so every predictor code-reading matrix (fan_in == D) gets
#   D=384  -> 5e-4 * 384/384  = 5.000e-4
#   D=2048 -> 5e-4 * 384/2048 = 9.375e-5   (5.3x lower)
# Confirmed in both runs' printed muP schema. So the "width helps" result is confounded
# with a 5.3x LR reduction, and a too-high LR would ALSO explain the control's
# catastrophic-zero seeds (LpWM-ltv spans 0.00-0.66 over 13 seeds).
#
# These two arms complete the 2x2. Zero code change: mup_lr is the 3rd ARMS field.
#   wave6 (PROJ_D=384):  base_lr 9.375e-5 -> matrices at 9.375e-5, matching d2048's rate
#   wave7 (PROJ_D=2048): base_lr 2.667e-3 -> matrices at 5e-4, matching d384's rate
# CAVEAT: base_lr also sets vector-like params (biases/LayerNorm), which muP holds at
# base_lr regardless of fan_in. So the match is exact for matrices and off by the same
# factor for biases. Matrices dominate; an exact match would need a per-group override.
wave6_arms() {
    ORDER[wave6]="LpWM-ltv-lr9e5"
    ARMS[LpWM-ltv-lr9e5]="ltv 1.0 9.375e-5"
}

wave7_arms() {
    ORDER[wave7]="LpWM-ltv-d2048-hilr"
    ARMS[LpWM-ltv-d2048-hilr]="ltv 1.0 2.667e-3"
}

# wave12: THE COLUMNS ARM. Launch with FEATURE=patch.
#
# The campaign has been testing theories about cortical columns on a model that has
# exactly one token. conf/encoder/vit_scratch.yaml sets feature: cls, so
# vit_encoder.py:55-56 gives num_patches=1 and the predictor sees ONE token per frame;
# LinearDynamicsPredictor has no positional embedding and no per-patch parameters
# (infojepa_modules.py:545-567,612-638), so its operator is broadcast over a patch axis
# of size 1. There is no population of columns to vote, and no location to have a frame
# in. FEATURE=patch selects vit_scratch_patch -> 256 tokens, each with a position.
#
# Config-only: same predictor, reg_weight, mup_lr, link, target_p as LpWM-ltv, so the
# already-evaluated LpWM-ltv s3-s15 (n=13, 2 epochs) is the matched control for free.
# The run name carries "_patch" (FEAT_TAG), and analysis/figures.run_arm maps it to a
# SEPARATE arm, so a patch run can never pool with a cls run.
# wave13: THE REFERENCE-FRAME ARM, with its own matched control.
#
# TBT pairs sensory input with an allocentric LOCATION signal. This model throws that
# signal away: action_conditioning=adaln returns at visual_world_model.py:174 before
# obs["proprio"] is read, so the agent pose [x, y, vx, vy] is loaded and moved to GPU
# every batch while proprio_encoder receives ZERO gradient and is still optimised and
# checkpointed. USE_POSE=true adds its embedding to the action embedding -- the only
# conditioning-shaped tensor the predictor consumes.
#
# BOTH arms carry MUP_INPUT_FIX=true, and that is what makes the contrast single-factor:
# the pose module's input layer has fan_in=4 and, under the unfixed muP rule, would train
# at 96x base_lr (4.8e-2). Turning pose on without the fix would measure an optimizer
# blow-up, not a reference frame. The control isolates the LR fix on its own, so
# LpWM-ltv-mupfix vs the existing LpWM-ltv also prices the muP bug for free.
wave13_arms() {
    ORDER[wave13]="LpWM-ltv-mupfix PiWM-refframe"
    ARMS[LpWM-ltv-mupfix]="ltv 1.0 5e-4 MUP_INPUT_FIX=true"
    ARMS[PiWM-refframe]="ltv 1.0 5e-4 MUP_INPUT_FIX=true USE_POSE=true"
}

# ---------------------------------------------------------------------------------
# LeVJEPA (arXiv 2608.27395) combination waves. That paper trains a single encoder with
# an invariance loss + SIGReg, which constrains embeddings to an ISOTROPIC GAUSSIAN --
# "the distribution shown to minimize worst-case downstream probing risk" -- and thereby
# excludes collapse with a provable guarantee, dispensing with the EMA target encoder,
# the stop-gradient and the capacity-limited predictor.
#
# Two of those three we already match: this model has no EMA target, and
# conf/train_rdmreg.yaml sets detach_target: False, so there is no stop-gradient either.
# What we do NOT match is the target distribution: all 160 campaign runs used
# regularizer=rdmreg, and 157 of them used link=reprelu + target_p=1 -- a SPARSE,
# rectified target, the opposite of isotropic Gaussian. SIGReg is already implemented
# here (infojepa_modules.py:26, Epps-Pulley on 1024 random projections over 17 knots --
# the paper's exact hyperparameters) and has never once been used.
#
# Measured statistic scale, which is why reg_weight cannot be carried over from RDMReg:
#   SIGReg  at N(0,I) 1.10 | rank-1 collapse 10.8 | dead code 25.7
#   RDMReg  healthy 0.032  |                      | dead code 0.51
# A dead code cost RDMReg 0.51 -- payable, which is exactly how PiWM-union4 reached
# rho=0.0000. SIGReg charges 25.7 for the same code.
wave14_arms() {
    ORDER[wave14]="LeWM-ltv-p2 PiWM-sigreg PiWM-sigreg-w0p5"
    # Single-factor control: SAME isotropic-Gaussian target (identity link, p=2), but the
    # OLD regularizer. Isolates SIGReg-vs-RDMReg from the change of target distribution.
    ARMS[LeWM-ltv-p2]="ltv 0.1 5e-4"
    ARM_LINK[LeWM-ltv-p2]="identity 2"
    ARMS[PiWM-sigreg]="ltv 0.05 5e-4 REGULARIZER=sigreg"
    ARM_LINK[PiWM-sigreg]="identity 2"
    # reg_weight is the one nuisance parameter here; a 10x probe guards against a null
    # that is really just a mis-scaled coefficient.
    ARMS[PiWM-sigreg-w0p5]="ltv 0.5 5e-4 REGULARIZER=sigreg"
    ARM_LINK[PiWM-sigreg-w0p5]="identity 2"
}

# C2: token dropping. LAUNCH WITH FEATURE=patch -- with feature=cls there is exactly one
# token and nothing to drop, which is why this was untestable before wave12. The paper
# reports ImageNet accuracy rising MONOTONICALLY with the drop ratio (33.9% -> 47.6% at
# rho=0.95). Note this is INPUT-space sparsity; our own latent-code sparsity (k-WTA)
# zeroed planning at every density and width, so the two are worth separating.
wave15_arms() {
    ORDER[wave15]="PiWM-drop95"
    ARMS[PiWM-drop95]="ltv 1.0 5e-4 TOKEN_DROP=0.95"
}

# C3: remove the last architectural asymmetry we still have -- the capacity-limited
# predictor. LTV is deliberately low-rank; ar_adaln is the deep AdaLN transformer.
# Paired against PiWM-sigreg, this is single-factor (predictor only).
wave16_arms() {
    ORDER[wave16]="PiWM-sigreg-arpred"
    ARMS[PiWM-sigreg-arpred]="ar_adaln 0.05 5e-4 REGULARIZER=sigreg"
    ARM_LINK[PiWM-sigreg-arpred]="identity 2"
}

# C4: block-causal temporal attention. Verified: perturbing the LAST frame leaves frame
# 0's embedding bit-identical, so the mask really is causal across frames. Today
# encode_obs folds t into the batch and encodes every frame INDEPENDENTLY, so the encoder
# has no temporal structure at all -- for a world model, which must be causal, that is a
# structural gap rather than a tuning choice.
wave17_arms() {
    ORDER[wave17]="PiWM-blockcausal"
    ARMS[PiWM-blockcausal]="ltv 1.0 5e-4 BLOCK_CAUSAL=true"
}

# =================================================================================
# wave20: THE CAUSAL FACTORIAL. The action is causally near-inert in the base objective:
# across 9 trained arms, changing a_t moves the PREDICTED latent by 0.02-0.1% of its own
# magnitude (d_state/d_action = 37x for LpWM-ltv), and that displacement is the strongest
# predictor of CEM success measured in this project (Spearman +0.81, p=0.0079) -- stronger
# than rel_mse, while the RATIO d_action/|z| predicts nothing (+0.28, p=0.47).
# CEM plans ONLY by comparing action sequences, so a flat landscape over actions makes the
# planner near-random however accurate the marginal prediction is.
#
# V1 incr_norm : per-sample increment normalisation. NB "loss on the increment" is a NO-OP
#                (the z_t cancels exactly) and a batch-level normaliser only rescales; only
#                PER-SAMPLE weighting changes gradient direction, by stopping frames with
#                large autonomous motion from dominating.
# V2 act_info  : InfoNCE over in-batch action negatives -> max I(a_t ; z_t+1 | z_t). The
#                first intervention here that changes what the model must DISTINGUISH
#                rather than what its code's marginal looks like.
# V3 path_int  : learned path integration of the location signal. PiWM-refframe RECEIVED
#                pose and was a null (-0.063, p=0.169); TBT requires the frame to be UPDATED
#                BY THE MOVEMENT, which is the causal half we omitted.
#
# V3 arms carry MUP_INPUT_FIX because they re-activate the proprio encoder, whose input
# layer sits at 96x base_lr without it -- so their matched control is LpWM-ltv-mupfix
# (already n=12), while the V1/V2 arms pair against LpWM-ltv (n=13). Both controls exist.
wave20_arms() {
    ORDER[wave20]="PiWM-incr PiWM-actinfo PiWM-incr-actinfo PiWM-pathint PiWM-actinfo-pathint"
    ARMS[PiWM-incr]="ltv 1.0 5e-4 INCR_NORM=true"
    ARMS[PiWM-actinfo]="ltv 1.0 5e-4 ACT_INFO=0.1"
    ARMS[PiWM-incr-actinfo]="ltv 1.0 5e-4 INCR_NORM=true ACT_INFO=0.1"
    ARMS[PiWM-pathint]="ltv 1.0 5e-4 PATH_INT=true USE_POSE=true MUP_INPUT_FIX=true"
    ARMS[PiWM-actinfo-pathint]="ltv 1.0 5e-4 ACT_INFO=0.1 PATH_INT=true USE_POSE=true MUP_INPUT_FIX=true"
}

# wave21: DE-CONFOUND link vs target_p. The archive has 173 runs at (reprelu, p=1) and 35 at
# (identity, p=2) and ZERO off-diagonal, so "sparse vs dense" has never actually been tested
# -- it is perfectly confounded with rectification. And target_p is measurably INERT at
# D=384: swd(Gaussian, Laplace) = 1.9947 +- 0.0170 against a null of 1.9987 +- 0.0112, the
# alternative BELOW the null, because a 1-D random projection of a D=384 Laplace sample is
# Gaussian (excess kurtosis -0.002, Shapiro p=0.396 -- Diaconis-Freedman). RDMReg only ever
# looks at 1-D projections, so it cannot see p=1 vs p=2, and rho~0.5 comes from the ReLU
# link rather than the sparse prior.
#
# reg_weight follows the LINK (1.0 sparse / 0.1 dense, the paper's own swept cells), so
# within each row only target_p varies -- which is exactly the inert-knob question.
# ROUND 4. The campaign varied the link, the target, the regulariser, the loss, sparsity,
# width, LR, the encoder, the conditioning and the head count -- and used predictor=ltv in
# 36 of 39 arms. wave22 varies the map g(z,a) itself, the one axis nobody touched.
#   P2  linear_var / linear_pa: two cells of the {additive,multiplicative} x {1,3 lags}
#       factorial that are fully implemented and have NEVER been run.
#   P1  lie: the action acts as a GROUP ELEMENT (block-diagonal 2x2 rotations). 10.8x
#       fewer predictor params than ltv; P1b enlarges the group to similitudes.
#   P3  actgain: the one-scalar falsifier -- if merely reweighting the additive action term
#       recovers what lie recovers, the structure is unnecessary.
#   P4  ctrb: -logdet of the H-step controllability Gramian. Run on linear_var, where the
#       linearisation is EXACT; its control LpWM-linvar is an arm in this same wave.
#   P5  actinfo-cond: V2's objective with negatives from p(a|z_t) instead of p(a).
# Controls: LpWM-ltv (n=16) and PiWM-actinfo (n=8) already exist; LpWM-linvar is in-wave.
wave22_arms() {
    ORDER[wave22]="${WAVE22_ARMS:-LpWM-linvar PiWM-multact PiWM-lie PiWM-lie-sim PiWM-actgain-b03 PiWM-actgain-b30 PiWM-ctrb PiWM-actinfo-cond PiWM-actinfo-cond-sigreg}"
    ARMS[LpWM-linvar]="linear_var 1.0 5e-4"
    ARMS[PiWM-multact]="linear_pa 1.0 5e-4"
    ARMS[PiWM-lie]="lie 1.0 5e-4"
    ARMS[PiWM-lie-sim]="lie 1.0 5e-4 LIE_SIM=true"
    ARMS[PiWM-actgain-b03]="ltv 1.0 5e-4 ACT_GAIN=0.3"
    ARMS[PiWM-actgain-b30]="ltv 1.0 5e-4 ACT_GAIN=3.0"
    ARMS[PiWM-ctrb]="linear_var 1.0 5e-4 CTRB_W=0.01"
    ARMS[PiWM-actinfo-cond]="ltv 1.0 5e-4 ACT_INFO=0.1 ACT_INFO_NEG=knn"
    ARMS[PiWM-actinfo-cond-sigreg]="ltv 0.05 5e-4 ACT_INFO=0.1 ACT_INFO_NEG=knn REGULARIZER=sigreg"
    ARM_LINK[PiWM-actinfo-cond-sigreg]="identity 2"
}

# ROUND 5, wave23. Four objectives that all touch the same three files, so they are one
# wave. Every arm here is paired with a control that shares its seed, its window count and
# its construction order, because RNG consumption differs the moment a module is built:
# the decoder is instantiated (train.py init_models) BEFORE mup_init_(encoder), so a
# decoder run is NOT seed-matched to a no-decoder run even with the gradient detached.
#
#  T1  decode / decode-detach / decode-w1
#      Nothing in this system has ever forced the latent to retain visual content
#      (has_decoder: False everywhere, and the decoder branch that exists passed
#      z_emb.detach()). Measured on CPU: with the detach, 0 of 144 encoder parameters
#      receive gradient from decoder_recon_loss; without it, 144/144. decode-detach is
#      THE control -- it also builds the decoder and also adds the pixel loss, so only
#      the gradient path differs. decode-w1 is the nuisance-weight guard cell.
#  T2  patchdecode / patchdecode-detach   [FEATURE=patch, per-arm]
#      The missing (tokens) x (decoder) cell. PiWM-columns (256 tokens, no decoder) was a
#      null: +0.072, 95% CI [-0.064, +0.207], n=12 -- spatial capacity with nothing asking
#      for spatial content. PatchHead is the first parameter in this model that
#      distinguishes token i from token j (verified: perturbing token i changes ONLY its
#      own 14x14 block), and at 226,380 params it is 138x SMALLER than T1's decoder, so a
#      T2 win cannot be read as decoder capacity.
#  T3  contact / contact-shuf / contact-g05
#      Weight each transition by the visual change proprio does NOT explain: the agent's
#      position is a model INPUT, so its disc can be masked exactly and the residual is,
#      to first order, the block. Measured offline at training batch composition (60
#      episodes, 3,264 transitions): the masked residual separates static from
#      block-moving transitions by 3690x against the raw pixel difference's 2.3x, moves
#      the weight mass on moving transitions from 0.36 (uniform) to 0.66, and correlates
#      with true block motion at rho=+0.95 (rotation included) -- all without states.pth.
#      contact-shuf is the ESS-matched control: the SAME weights, permuted, so effective
#      compute (ESS/N = 0.447, measured identical) is held and only the alignment dies.
#  T6  jump5 / overshoot5
#      num_pred IS K. Over one action row the block's displacement is exactly 0.000 px in
#      48% of transitions (median 0.084 px); over a 5-row option the zero fraction falls
#      to 12.7% and the median rises to 24.6 px. Window count falls 1,981,721 ->
#      1,608,021 (verified), and CEM's whole goal_H=5 horizon becomes ONE predictor call
#      (verified), so nothing compounds. overshoot5 holds the window and the horizon and
#      keeps the compounding, which is the attribution.
#
# Matched baseline for T3 and T6 is LpWM-ltv (n=16, already trained -- deliberately not
# retrained here); T2 also reads against PiWM-columns (n=12, already evaluated).
wave23_arms() {
    ORDER[wave23]="${WAVE23_ARMS:-PiWM-decode PiWM-decode-detach PiWM-decode-w1 PiWM-patchdecode PiWM-patchdecode-detach PiWM-contact PiWM-contact-shuf PiWM-contact-g05 PiWM-jump5 PiWM-overshoot5}"
    # T1. AUX_DECODER, not has_decoder: plan.py:502-515 rebuilds a decoder for any run
    # whose train_cfg has has_decoder=True and raises ValueError when the checkpoint holds
    # a state_dict and env.decoder_path is null -- which is EVERY checkpoint this repo
    # writes -- so has_decoder=true would train fine and then be impossible to evaluate.
    ARMS[PiWM-decode]="ltv 1.0 5e-4 AUX_DECODER=true DECODE_GRAD=true LAMB_DECODE=0.1"
    ARMS[PiWM-decode-detach]="ltv 1.0 5e-4 AUX_DECODER=true DECODE_GRAD=false LAMB_DECODE=0.1"
    ARMS[PiWM-decode-w1]="ltv 1.0 5e-4 AUX_DECODER=true DECODE_GRAD=true LAMB_DECODE=1.0"
    # T2. Same decoder plumbing, 256 tokens, per-patch head. FEATURE is per-arm so the
    # 2x2 launches as one wave; the run name carries "_patch" either way.
    ARMS[PiWM-patchdecode]="ltv 1.0 5e-4 AUX_DECODER=true DECODER=patch_head DECODE_GRAD=true LAMB_DECODE=0.1"
    ARMS[PiWM-patchdecode-detach]="ltv 1.0 5e-4 AUX_DECODER=true DECODER=patch_head DECODE_GRAD=false LAMB_DECODE=0.1"
    ARM_FEAT[PiWM-patchdecode]="patch"
    ARM_FEAT[PiWM-patchdecode-detach]="patch"
    # T3.
    ARMS[PiWM-contact]="ltv 1.0 5e-4 CONTACT_GAMMA=1.0"
    ARMS[PiWM-contact-shuf]="ltv 1.0 5e-4 CONTACT_GAMMA=1.0 CONTACT_SHUF=true"
    ARMS[PiWM-contact-g05]="ltv 1.0 5e-4 CONTACT_GAMMA=0.5"
    # T6. NUM_PRED changes the DATASET window (num_frames = num_hist + num_pred), so both
    # arms set it and their batch composition is identical.
    ARMS[PiWM-jump5]="ltv 1.0 5e-4 NUM_PRED=5"
    ARMS[PiWM-overshoot5]="ltv 1.0 5e-4 NUM_PRED=5 OVERSHOOT=true"
}

# ROUND 5, wave24. T4 (+ V4's head, which rides the same job).
#
# T4  PiWM-vp / PiWM-vp-mc / PiWM-vp-geom
#     V(z, g) by in-sample expectile TD (tau = 0.7) on HINDSIGHT goals: any z_{t+k} in a
#     window is a valid goal with label -k, so the labels are free and fully
#     self-supervised -- latents and step counts only, no reward and no states.pth.
#     The motivation is the LEAF SCORE, not d_action: the latent distance CEM minimises is
#     ranked against the true task distance at only Spearman +0.398 (n = 296), and block
#     motion is a tail (median displacement per model step exactly 0.000 px, 48.1% of
#     transitions exactly zero, top 1% carrying 34.8% of all motion), so ||z - g||^2 is
#     dominated by directions along which nothing task-relevant happened. -E[steps] is
#     invariant to any monotone reparametrisation of the latent, so it cannot be gamed by
#     the scale of the code.
#       PiWM-vp-mc   isolates TD from "any learned scalar that correlates with progress"
#                    (hindsight MC regression to -k; no bootstrap, no target net).
#       PiWM-vp-geom isolates temporal structure from "a smooth MLP reparametrisation of
#                    the distance CEM already uses" (same head, same optimiser, regressed
#                    to ||z - g|| in units of one mean latent step).
#     Both controls are single-token VALUE_MODE flips on identical code paths, and all
#     three consume the RNG stream identically (_hindsight_pairs runs in every mode), so
#     the encoder/predictor trajectory is matched.
#
# V4  rides PiWM-vp via POLICY_W=1.0. a_t = pi(z_t, z_g) by behaviour cloning on the SAME
#     latent, so one checkpoint yields the CEM number, the V5 number and the V4 number on
#     bit-identical encoder/predictor weights -- this run's own CEM eval IS V4's control.
#     Both heads read DETACHED latents, which is what makes that true.
#
# Matched baseline for the encoder/predictor is LpWM-ltv (n=16, already trained and
# deliberately not retrained). The heads' own controls are in-wave.
wave24_arms() {
    ORDER[wave24]="${WAVE24_ARMS:-PiWM-vp PiWM-vp-mc PiWM-vp-geom}"
    ARMS[PiWM-vp]="ltv 1.0 5e-4 VALUE_W=1.0 POLICY_W=1.0"
    ARMS[PiWM-vp-mc]="ltv 1.0 5e-4 VALUE_W=1.0 POLICY_W=1.0 VALUE_MODE=mc"
    ARMS[PiWM-vp-geom]="ltv 1.0 5e-4 VALUE_W=1.0 POLICY_W=1.0 VALUE_MODE=geom"
}

# ROUND 6, wave25. Five GRIDS. Not five arms: every cell below brackets its own failure
# mode around a MEASURED quantity, because the audit that opened this round found that six
# of the nine round-3/4/5 proposals were SINGLE-SHOT on their strength knob -- V1 had no
# knob at all (its epsilon was the literal 1e-4 at visual_world_model.py:1091, 414x below
# the median increment) and T4 would have died on its primary setting (val/rho_k 0.443
# against a 0.6 gate) and was saved only by a variant. A grid cannot be killed by one
# unlucky choice of scalar, and a grid's SHAPE is itself the result.
#
# The harness has no native grid, so a grid is N hand-written ARMS[...] lines and N names
# in ORDER -- exactly as PiWM-actgain-b03/-b30 and PiWM-decode/-w1 already are.
#
# Death condition, PRE-REGISTERED for every arm in this wave (conf/train_rdmreg.yaml):
# err/rel_mse >= 0.5. An arm at or above it is not a world model and its CEM number is
# evidence about the objective breaking prediction, not about the objective.
#
#  R6  support-w0p03 / -w0p1 / -w0p3            control: LpWM-ltv (support_w=0, n=16)
#      Optimise the ONE logged quantity analysis/screen_objective.py endorses:
#      jacc/S_model = 1 - soft_jaccard(z_pred, target), raw Spearman -0.769 against CEM,
#      partial (rho and rel_mse removed) -0.549, monotone over healthy predictors
#      (0.407 / 0.365 / 0.249 / 0.058). soft_jaccard appeared ZERO times in models/ before
#      this round -- it has only ever been a diagnostic. Grid centred on PARITY with the
#      MSE, measured on the trained baseline over 4 x 32 real valid windows: S_model =
#      0.0866 against z_loss = 0.00578, so parity is support_w = 0.0667 and the grid
#      0.03 / 0.1 / 0.3 is [0.45x, 4.5x] parity.
#  R2  consist-w0p03 / -w0p1 / -w0p3 (src=cem)  + consist-w0p1-data (DISTRIBUTION control)
#      || P_K(z, a) - chain(P_1, z, a) ||^2 at actions the CEM PROPOSAL draws, which no
#      training arm in ~60 has ever done: both sides are the model's own output, so a
#      counterfactual action needs no recorded next frame. Measured on the baseline:
#      consist_loss 0.1107 (cem) vs 0.1915 (data) against z_loss 0.00578, i.e. the trained
#      model's 5-step jump already disagrees with its own 5-step chain by ~19x its 1-step
#      error; parity is consist_w = 0.052, so 0.03 / 0.1 / 0.3 brackets it. The -data cell
#      is the matched control at the SAME weight: same term, same shapes, same predictor
#      calls, same private RNG, dataset actions instead of the planner's Gaussian -- so
#      the contrast is the DISTRIBUTION and nothing else.
#  R3  sam-r0p01 / -r0p03 / -r0p1               control: LpWM-ltv (sam_rho=0)
#      min_theta max_{||da|| <= rho} L(z, a+da), one SAM ascent step per action ROW, so
#      rho is in units of one normalised action and directly comparable to CEM's
#      var_scale = 1. Measured sharpness (L(a+d)-L(a))/L(a) on the baseline: 0.061 at
#      rho=0.03, 0.236 at 0.1, 0.997 at 0.3. The grid runs one step LOWER than that table
#      (0.01 / 0.03 / 0.1) because the falsifier here is a COLLAPSE, not a null:
#      d_action -> 0 is the unconstrained minimiser of the inner max, and rho=0.3 already
#      doubles the loss inside the ball. sam_d_action is logged every epoch for exactly
#      this; the baseline sits at 0.345 and CEM's relation to it is an inverted U.
#  R4  incr-eps0p001 / -eps0p01 / -eps0p041     + -eps0p041-clip10 (bounded-weight cell)
#      V1 COMPLETED. PiWM-incr scored -0.383 [-0.497,-0.268] with 8/8 seeds dead at the
#      hardcoded eps=1e-4; its per-sample weight 1/(increment + eps) then spans four
#      orders of magnitude and ESS/N = 0.063 -- six percent of a batch, and V1 had no way
#      to leave that cell. This grid walks the knee: ESS/N = 0.242 / 0.591 / 0.813 at the
#      three eps (the round's pre-launch measurement), and the clip10 cell bounds the
#      SPAN (max/min <= 100) at the top eps instead, separating "the weighting was too
#      sharp" from "the weighting was wrong". eps = 0.041 is the MEDIAN increment, so the
#      top cell is the one where the weight is flat by construction.
#  R1  jump2/overshoot2, jump3/overshoot3, jump8/overshoot8
#      T6's K sweep. num_pred IS K and K is T6's whole hypothesis, yet only K=5 has ever
#      run (PiWM-jump5 / PiWM-overshoot5, already in flight at 8 seeds -- deliberately not
#      repeated here). Each K carries its OWN matched overshoot control at the same K,
#      because NUM_PRED changes the dataset window (num_frames = num_hist + num_pred) and
#      therefore the window count and batch composition: a jump arm compared against an
#      overshoot arm at a different K would confound the horizon with the data. Within a
#      K the two differ only in whether error compounds, which is the attribution T6
#      claims. K=2 and K=3 sit below the planner's goal_H=5, K=8 above it.
#
# Controls: LpWM-ltv (n=16, trained) is the matched zero-strength control for R6, R3 and
# R4, and is deliberately NOT retrained here. R2's control is in-wave (-data); R1's are
# in-wave (the three overshoot cells) plus the existing K=5 pair.
wave27_arms() {
    # ROUND 7, REDESIGNED -- the DIMENSION-MATCHED cls-vs-patch grid.
    #
    # THE CONFOUND THIS FIXES. A patch arm carries num_patches x proj_dim latent values; a cls
    # arm carries proj_dim. At the campaign's defaults that is 256 x 384 = 98304 against 384 --
    # a 256x capacity gap. So "PiWM-columns vs LpWM-ltv" (+0.072) and the orientation probe
    # (9.58 deg vs 15.72) BOTH confound the feature with a 256x capacity change, and
    # LpWM-ltv-d2048 -- the only width control ever run -- is 5.3x, nowhere near matched.
    #
    # num_patches = (img_size/patch_size)^2 and img_size is 224, so patch_size is the token
    # knob: 14 -> 256, 56 -> 16, 112 -> 4, 224 -> 1. encoder.proj_dim is settable independently
    # of the ViT width (scripts/train.sh PROJ_DIM), so the two can be matched on TOTAL LATENT:
    #
    #     total dims | cls                        | patch
    #     -----------|----------------------------|---------------------------------
    #        384     | LpWM-ltv          (exists) | LpWM-ltv-p1   patch_size=224, 1 token
    #       1536     | LpWM-ltv-d1536             | PiWM-cols-p4  patch_size=112, 4 tokens
    #       6144     | LpWM-ltv-d6144             | PiWM-cols-p16 patch_size=56, 16 tokens
    #      98304     | infeasible                 | PiWM-columns      (exists)
    #
    # Each row is a within-capacity cls-vs-patch contrast; each column is a capacity ladder at
    # fixed feature. That separates "tokens carry orientation" from "more latent is better",
    # which nothing in rounds 1-7 could.
    #
    # PROJ_D is invocation-wide, so the cls rungs are launched with their own PROJ_D and carry
    # it in the ARM NAME -- reusing "LpWM-ltv" at a different PROJ_D would file them under the
    # baseline's key in collect_evals and silently pool two architectures.
    ORDER[wave27]="${WAVE27_ARMS:-LpWM-ltv-p1 PiWM-cols-p4 PiWM-cols-p16}"
    ARMS[LpWM-ltv-p1]="ltv 1.0 5e-4 PATCH_SIZE=224";   ARM_FEAT[LpWM-ltv-p1]="patch"
    ARMS[PiWM-cols-p4]="ltv 1.0 5e-4 PATCH_SIZE=112";  ARM_FEAT[PiWM-cols-p4]="patch"
    ARMS[PiWM-cols-p16]="ltv 1.0 5e-4 PATCH_SIZE=56";  ARM_FEAT[PiWM-cols-p16]="patch"
}

wave28_arms() {
    # The cls half of the same grid. Launched separately because PROJ_D is invocation-wide:
    #   PROJ_D=1536 scripts/run_campaign.sh wave28   -> LpWM-ltv-d1536
    #   PROJ_D=6144 scripts/run_campaign.sh wave28   -> LpWM-ltv-d6144
    # The arm name must match the PROJ_D or collect_evals pools them; the launcher checks.
    ORDER[wave28]="${WAVE28_ARMS:-LpWM-ltv-d${PROJ_D}}"
    ARMS[LpWM-ltv-d${PROJ_D}]="ltv 1.0 5e-4"
}

wave26_arms() {
    # ROUND 7 -- the representation, not the objective.
    #
    # Six rounds and ~120 contrasts changed the LOSS on a fixed representation:
    # `feature=cls`, a SINGLE 384-d token for the whole image. They produced exactly one
    # positive, and that one (plan-time consensus) is not a world-model result at all.
    # diary/2026-09-04 §7.4 says why, from a probe that needs no dynamics and no contrast:
    #
    #   ridge-decode the block pose from ONE FROZEN FRAME, median angular error, against
    #   the best CONSTANT prediction (14.51 deg -- PushT's T settles into a canonical pose):
    #       LpWM-ltv        cls              15.72 deg   WORSE than the constant bound
    #       LpWM-ltv-d2048  cls, 5x width    14.64 deg   at the bound -- width is not it
    #       PiWM-columns    patch             9.58 deg   the only arm that beats it
    #       PiWM-drop95     patch, 95% drop  16.01 deg   back at the bound
    #
    # The baseline latent cannot see the variable PushT is about. No loss defined on it can
    # recover information it does not contain, which is the null result 120 contrasts kept
    # producing. And drop95 shows it is the TOKENS, not the word "patch".
    #
    # S1 is therefore a DOSE-RESPONSE INSIDE THE PATCH FAMILY, which is the cleanest causal
    # design available here: it never compares patch against cls, so it cannot be confounded
    # by the feature. If angular error and CEM both degrade monotonically as tokens are
    # removed, tokens -> orientation -> planning is causal rather than correlational. The
    # endpoints already exist (TOKEN_DROP 0.0 = PiWM-columns, 0.95 = PiWM-drop95), so this
    # buys the four interior points of a six-point curve.
    #
    # Every arm here is feature=patch. NOTHING in this wave changes the objective.
    ORDER[wave26]="${WAVE26_ARMS:-PiWM-tok25 PiWM-tok50 PiWM-tok75 PiWM-tok90}"
    ARMS[PiWM-tok25]="ltv 1.0 5e-4 TOKEN_DROP=0.25";  ARM_FEAT[PiWM-tok25]="patch"
    ARMS[PiWM-tok50]="ltv 1.0 5e-4 TOKEN_DROP=0.50";  ARM_FEAT[PiWM-tok50]="patch"
    ARMS[PiWM-tok75]="ltv 1.0 5e-4 TOKEN_DROP=0.75";  ARM_FEAT[PiWM-tok75]="patch"
    ARMS[PiWM-tok90]="ltv 1.0 5e-4 TOKEN_DROP=0.90";  ARM_FEAT[PiWM-tok90]="patch"
    # S2 and S4 are NOT new arms and must not be. Renaming PiWM-columns or PiWM-patchdecode
    # would create a fresh key in collect_evals and the extra seeds would not pool with the
    # existing ones -- which is the entire point of running them. They are launched as MORE
    # SEEDS OF THE EXISTING ARMS, from their own waves:
    #
    #   S2  SEEDS="0 1 2"        WAVE22_ARMS="PiWM-columns"     run_campaign.sh wave22
    #       The single-factor patch-vs-cls contrast is +0.072 [-0.064, +0.207] at n=12, a
    #       positive-leaning NULL and the largest n of any treated arm. columns holds seeds
    #       3-15 and the baseline holds 0-15, so 0-2 are the cheapest points that widen it.
    #
    #   S4  SEEDS="11 12 13 14"  WAVE23_ARMS="PiWM-patchdecode PiWM-patchdecode-detach" wave23
    #       patchdecode carries the campaign's highest non-consensus arm MEAN (0.520 against
    #       the baseline's 0.357) and is +0.140 [-0.047, +0.327] against its own detach
    #       control at n=8. Its control needs the same seeds or the pairing does not widen.
}

wave25_arms() {
    ORDER[wave25]="${WAVE25_ARMS:-PiWM-support-w0p03 PiWM-support-w0p1 PiWM-support-w0p3 PiWM-consist-w0p03 PiWM-consist-w0p1 PiWM-consist-w0p3 PiWM-consist-w0p1-data PiWM-sam-r0p01 PiWM-sam-r0p03 PiWM-sam-r0p1 PiWM-incr-eps0p001 PiWM-incr-eps0p01 PiWM-incr-eps0p041 PiWM-incr-eps0p041-clip10 PiWM-jump2 PiWM-overshoot2 PiWM-jump3 PiWM-overshoot3 PiWM-jump8 PiWM-overshoot8}"
    # R6. The '0p03' spelling of 0.03 follows PiWM-sigreg-w0p5: a '.' in a run dir is
    # legal but analysis/figures.py's _ARM_STRIP and every glob in the analysis path are
    # easier to read without one, and the arm token must survive collect_evals.py intact.
    ARMS[PiWM-support-w0p03]="ltv 1.0 5e-4 SUPPORT_W=0.03"
    ARMS[PiWM-support-w0p1]="ltv 1.0 5e-4 SUPPORT_W=0.1"
    ARMS[PiWM-support-w0p3]="ltv 1.0 5e-4 SUPPORT_W=0.3"
    # R2. CONSIST_SRC is passed EXPLICITLY on the cem cells even though cem is the config
    # default, so the run's own .hydra/overrides.yaml records which distribution it was
    # trained on rather than leaving it to be inferred from the default of the day.
    ARMS[PiWM-consist-w0p03]="ltv 1.0 5e-4 CONSIST_W=0.03 CONSIST_SRC=cem"
    ARMS[PiWM-consist-w0p1]="ltv 1.0 5e-4 CONSIST_W=0.1 CONSIST_SRC=cem"
    ARMS[PiWM-consist-w0p3]="ltv 1.0 5e-4 CONSIST_W=0.3 CONSIST_SRC=cem"
    ARMS[PiWM-consist-w0p1-data]="ltv 1.0 5e-4 CONSIST_W=0.1 CONSIST_SRC=data"
    # R3.
    ARMS[PiWM-sam-r0p01]="ltv 1.0 5e-4 SAM_RHO=0.01"
    ARMS[PiWM-sam-r0p03]="ltv 1.0 5e-4 SAM_RHO=0.03"
    ARMS[PiWM-sam-r0p1]="ltv 1.0 5e-4 SAM_RHO=0.1"
    # R4. INCR_NORM=true is what turns the weighting ON; INCR_EPS alone is inert, which is
    # why every cell carries both (the flag that once made three variants silently
    # identical to baseline was exactly this kind of missing second leg).
    ARMS[PiWM-incr-eps0p001]="ltv 1.0 5e-4 INCR_NORM=true INCR_EPS=1e-3"
    ARMS[PiWM-incr-eps0p01]="ltv 1.0 5e-4 INCR_NORM=true INCR_EPS=1e-2"
    ARMS[PiWM-incr-eps0p041]="ltv 1.0 5e-4 INCR_NORM=true INCR_EPS=4.1e-2"
    ARMS[PiWM-incr-eps0p041-clip10]="ltv 1.0 5e-4 INCR_NORM=true INCR_EPS=4.1e-2 INCR_CLIP=10"
    # R1. Both cells of each K set NUM_PRED, so their dataset windows are identical.
    ARMS[PiWM-jump2]="ltv 1.0 5e-4 NUM_PRED=2"
    ARMS[PiWM-overshoot2]="ltv 1.0 5e-4 NUM_PRED=2 OVERSHOOT=true"
    ARMS[PiWM-jump3]="ltv 1.0 5e-4 NUM_PRED=3"
    ARMS[PiWM-overshoot3]="ltv 1.0 5e-4 NUM_PRED=3 OVERSHOOT=true"
    ARMS[PiWM-jump8]="ltv 1.0 5e-4 NUM_PRED=8"
    ARMS[PiWM-overshoot8]="ltv 1.0 5e-4 NUM_PRED=8 OVERSHOOT=true"
}

wave21_arms() {
    ORDER[wave21]="LpWM-ltv-relu-p2 LpWM-ltv-ident-p1"
    ARMS[LpWM-ltv-relu-p2]="ltv 1.0 5e-4"
    ARM_LINK[LpWM-ltv-relu-p2]="reprelu 2"
    ARMS[LpWM-ltv-ident-p1]="ltv 0.1 5e-4"
    ARM_LINK[LpWM-ltv-ident-p1]="identity 1"
}

wave12_arms() {
    ORDER[wave12]="PiWM-columns"
    ARMS[PiWM-columns]="ltv 1.0 5e-4"
}

wave5_arms() {
    local K; K=$(python -c "print(round(0.02*${PROJ_D}))")
    ORDER[wave5]="LpWM-ltv-d${PROJ_D} PiWM-sdr-d${PROJ_D}-k${K}"
    ARMS[LpWM-ltv-d${PROJ_D}]="ltv 1.0 5e-4"
    ARMS[PiWM-sdr-d${PROJ_D}-k${K}]="ltv 1.0 5e-4 KWTA_K=${K}"
}

submit_arm() {  # $1 = arm name, $2 = seed
    local arm=$1 seed=$2
    read -r pred rw mlr extra <<< "${ARMS[$arm]}"
    # per-arm encoder feature, defaulting to the invocation-wide one. ftag reproduces
    # FEAT_TAG exactly when ARM_FEAT is unset, so no existing run name moves.
    local feat="${ARM_FEAT[$arm]:-${FEATURE}}"
    local ftag=""; [ "${feat}" != "cls" ] && ftag="_${feat}"
    # precision is in the run name so a mixed-precision comparison is visible
    # rather than silent if PRECISION is ever changed mid-campaign
    local run="${CANARY_PREFIX:-}${arm}_pd${PROJ_D}${ftag}_${PRECISION}_s${seed}"
    local dir="${CKPT_BASE}/outputs/${run}"

    if [ "${EVAL:-0}" = "1" ]; then
        if [ ! -f "${dir}/DONE" ]; then
            echo "  SKIP eval ${run}: not finished training yet"
            return
        fi
        # sbatch, NOT inline. scripts/plan.sh runs `python plan.py` in the calling
        # shell -- plan.py's submitit path is only reached from train.py's epoch-end
        # hook -- so calling it here ran every CEM eval on the submit host, which has
        # no GPU. plan_slurm.sbatch gives it a real allocation.
        if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -q "^eval_${run}$"; then
            echo "  eval in flight already, skipping: ${run}"
            return
        fi
        # Already evaluated? The DONE sentinel says training finished, NOT that the
        # eval ran. Without this, a blanket `EVAL=1 run_campaign.sh <gate>` resubmits
        # completed evals, and collect_evals.py (sorted(), newest-timestamp-wins)
        # silently REPLACES the recorded success rate with a fresh noisy draw.
        if compgen -G "plan_outputs/*_${run}_gH*/logs.json" >/dev/null 2>&1; then
            echo "  already evaluated, skipping: ${run}"
            return
        fi
        echo "  eval ${run}"
        [ "${DRYRUN:-0}" = "1" ] && { echo "    [dry-run] sbatch --job-name=eval_${run} scripts/plan_slurm.sbatch"; return; }
        RUN_NAME="${run}" SEED="${seed}" NEVALS="${NEVALS}" MAXITER=10 \
            sbatch --job-name="eval_${run}" scripts/plan_slurm.sbatch | sed 's/^/    /'
        return
    fi

    if [ -f "${dir}/DONE" ]; then
        echo "  done already: ${run}"
        return
    fi
    # Never submit a second chain for a run that already has jobs in flight: both
    # would write ${dir}/checkpoints/model_latest.pth and corrupt each other's
    # resume point. This is the case when re-running the launcher to repair one
    # broken arm while the rest of the campaign is still training.
    if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -q "^${run}_w[0-9]*$"; then
        echo "  in flight already, skipping: ${run}"
        return
    fi
    read -r lnk tp <<< "${ARM_LINK[$arm]:-reprelu 1}"
    echo "  submit ${run}  (pred=${pred} rw=${rw} mup_lr=${mlr} link=${lnk} p=${tp} ${extra})"
    # shellcheck disable=SC2086
    [ "${DRYRUN:-0}" = "1" ] && { DRYRUN=1 env RUN_NAME="${run}" PREDICTOR="${pred}" \
        PROJ_DIM="${PROJ_D}" MUP=1 MUP_LR="${mlr}" REG_WEIGHT="${rw}" MU=0 SEED="${seed}" \
        REGULARIZER=rdmreg WINDOWS="${WINDOWS}" ${extra} \
        scripts/submit_until_done.sh pusht 5 3 "${EPOCHS:-2}" 64 "${lnk}" "${feat}" "${tp}" b "${WORKERS}" | sed 's/^/    /'; return; }
    env RUN_NAME="${run}" PREDICTOR="${pred}" PROJ_DIM="${PROJ_D}" MUP=1 MUP_LR="${mlr}" \
        REG_WEIGHT="${rw}" MU=0 SEED="${seed}" REGULARIZER=rdmreg WINDOWS="${WINDOWS}" \
        ${extra} \
        scripts/submit_until_done.sh pusht 5 3 "${EPOCHS:-2}" 64 "${lnk}" "${feat}" "${tp}" b "${WORKERS}" | sed 's/^/    /'
}

[ $# -gt 0 ] || { sed -n '2,25p' "$0"; exit 1; }

SUBMITTED=""
for gate in "$@"; do
    case "${gate}" in
        sparse|step2) sparse_arms; gate=sparse ;;
        gate|step3)   gate_arms;   gate=gate   ;;
        union|step4)  union_arms;  gate=union  ;;
        wave2)        wave2_arms;  gate=wave2  ;;
        wave3)        wave3_arms;  gate=wave3  ;;
        wave4)        wave4_arms;  gate=wave4  ;;
        wave5)        wave5_arms;  gate=wave5  ;;
        wave6)        wave6_arms;  gate=wave6  ;;
        wave7)        wave7_arms;  gate=wave7  ;;
        wave12)       wave12_arms; gate=wave12 ;;
        wave13)       wave13_arms; gate=wave13 ;;
        wave20)       wave20_arms; gate=wave20 ;;
        wave21)       wave21_arms; gate=wave21 ;;
        wave22)       wave22_arms; gate=wave22 ;;
        wave23)       wave23_arms; gate=wave23 ;;
        wave24)       wave24_arms; gate=wave24 ;;
        wave25)       wave25_arms; gate=wave25 ;;
        wave26)       wave26_arms; gate=wave26 ;;
        wave27)       wave27_arms; gate=wave27 ;;
        wave28)       wave28_arms; gate=wave28 ;;
        wave14)       wave14_arms; gate=wave14 ;;
        wave15)       wave15_arms; gate=wave15 ;;
        wave16)       wave16_arms; gate=wave16 ;;
        wave17)       wave17_arms; gate=wave17 ;;
        *) echo "unknown gate '${gate}' (expected sparse|gate|union|wave2..wave7|wave12..wave17|wave20..wave28)" >&2; exit 1 ;;
    esac
    echo "=== ${gate}: $(echo "${ORDER[$gate]}" | wc -w) arms x $(echo "${SEEDS}" | wc -w) seeds ==="
    for arm in ${ORDER[$gate]}; do
        for seed in ${SEEDS}; do
            case " ${SUBMITTED} " in
                *" ${arm}_s${seed} "*) echo "  shared control, already handled: ${arm}_s${seed}"; continue ;;
            esac
            submit_arm "${arm}" "${seed}"
            SUBMITTED="${SUBMITTED} ${arm}_s${seed}"
        done
    done
    echo
done

echo "Monitor: squeue -u \$USER -o '%.18i %.34j %.8T %.10M %R'"
echo "When all DONE:  EVAL=1 scripts/run_campaign.sh $*"
