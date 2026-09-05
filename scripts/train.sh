#!/bin/bash
# Single training run of the from-scratch JEPA world model (PushT / Wall).
#
# Cluster-agnostic: this just runs `python train.py ...` on the current machine.
# Wrap it in whatever scheduler you use, or run it directly. Prerequisites:
#   - activate the `lpwm` conda env         (see README.md > Installation)
#   - export DATASET_DIR=/path/to/data      (folder containing pusht_noise/ and wall_single/)
#   - optional: `wandb login`, or `export WANDB_MODE=offline` to skip logging
# PushT/Wall are pure-Python (pymunk/pygame, numpy) -- no simulator install needed.
#
# Usage:
#   scripts/train.sh <env> <frameskip> <num_hist> <epochs> <batch> <link> <feature> [target_p] [agg] [num_workers]
#     <env>     : pusht | wall
#     <link>    : reprelu (sparse LpWM) | identity (dense LeWM)
#     <feature> : cls | patch
#     [target_p]: 1 (sparse rectified-Laplace) | 2 (dense Gaussian);  [agg]: b (default) | btp | bp | bt
#
# Method-knob env-var overrides (all optional):
#   PREDICTOR (ar_adaln|ar_adaln_d1|mlp_var|ltv|linear_var|linear_wb), PROJ_DIM (latent dim D),
#   SUPPORT_W (R6), CONSIST_W/CONSIST_SRC/CONSIST_K/CONSIST_SIGMA (R2), SAM_RHO (R3),
#   INCR_EPS/INCR_CLIP (R4, with INCR_NORM=true),
#   MU (sparsity), MUP=1 MUP_LR=1e-4, REG_WEIGHT, LAMB_VAR, LAMB_COV, VAR_SPACE, REGULARIZER
#   (rdmreg|sigreg|none), TRAIN_ENCODER, SEED, RUN_NAME, WANDB_PROJECT, SAVE_EVERY, DEBUG=1,
#   and CKPT_BASE (where run dirs are written; default ./runs).
set -euo pipefail
ENV=${1:?usage: train.sh <env> <frameskip> <num_hist> <epochs> <batch> <link> <feature> [target_p] [agg] [num_workers]}
FRAMESKIP=${2:?need frameskip}; NUM_HIST=${3:?need num_hist}; EPOCHS=${4:?need epochs}
BATCH=${5:?need batch}; LINK=${6:?need link: reprelu|identity}; FEATURE=${7:?need feature: cls|patch}
case "${LINK}" in reprelu) DEFP=1.0;; identity) DEFP=2.0;; *) DEFP=1.0;; esac
TARGET_P=${8:-${DEFP}}; AGG=${9:-b}; NUM_WORKERS=${10:-20}
case "${FEATURE}" in cls) ENCODER=vit_scratch;; patch) ENCODER=vit_scratch_patch;;
  *) echo "feature must be cls|patch" >&2; exit 1;; esac
ENCODER=${ENCODER_OVERRIDE:-${ENCODER}}

REPO=$(cd "$(dirname "$0")/.." && pwd)
# local secrets + paths; already-exported values win so sbatch/CLI can override
if [ -f "${REPO}/.env" ]; then
    _pre_ds=${DATASET_DIR:-}; _pre_cb=${CKPT_BASE:-}
    set -a; . "${REPO}/.env"; set +a
    [ -n "${_pre_ds}" ] && DATASET_DIR=${_pre_ds}
    [ -n "${_pre_cb}" ] && CKPT_BASE=${_pre_cb}
fi
: "${DATASET_DIR:?set DATASET_DIR to the dataset root (contains pusht_noise/ and wall_single/)}"
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}

# headless pygame rendering (PushT) + single-process torch (generic; not cluster-specific)
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

EXTRA=""; TAG=""
add(){ EXTRA="${EXTRA} $1"; }
[ -n "${LR:-}" ]         && { add "training.encoder_lr=${LR} training.predictor_lr=${LR} training.action_encoder_lr=${LR} training.link_lr=${LR}"; TAG="${TAG}_lr${LR}"; }
[ -n "${REG_WEIGHT:-}" ] && { add "reg_weight=${REG_WEIGHT}"; TAG="${TAG}_rw${REG_WEIGHT}"; }
[ -n "${LAMB_VAR:-}" ]   && { add "lamb_var=${LAMB_VAR}"; TAG="${TAG}_lv${LAMB_VAR}"; }
[ -n "${LAMB_COV:-}" ]   && { add "lamb_cov=${LAMB_COV}"; TAG="${TAG}_lc${LAMB_COV}"; }
[ -n "${VAR_GAMMA:-}" ] && { add "var_gamma=${VAR_GAMMA}"; TAG="${TAG}_vg${VAR_GAMMA}"; }
[ -n "${MUP_INPUT_FIX:-}" ] && { add "mup_input_lr_fix=${MUP_INPUT_FIX}"; TAG="${TAG}_mupfix"; }
[ -n "${USE_POSE:-}" ]   && { add "use_pose=${USE_POSE}"; TAG="${TAG}_pose"; }
[ -n "${TOKEN_DROP:-}" ] && { add "token_drop=${TOKEN_DROP}"; TAG="${TAG}_drop${TOKEN_DROP}"; }
[ -n "${BLOCK_CAUSAL:-}" ] && { add "block_causal=${BLOCK_CAUSAL}"; TAG="${TAG}_bc"; }
[ -n "${INCR_NORM:-}" ]  && { add "incr_norm=${INCR_NORM}"; TAG="${TAG}_incr"; }
[ -n "${ACT_INFO:-}" ]   && { add "act_info=${ACT_INFO}"; TAG="${TAG}_ai${ACT_INFO}"; }
[ -n "${PATH_INT:-}" ]   && { add "path_int=${PATH_INT}"; TAG="${TAG}_pi"; }
[ -n "${VAR_SPACE:-}" ]  && add "var_space=${VAR_SPACE}"
# round 4 (P1-P5); unset => upstream behaviour, bit-identical
[ -n "${LIE_SIM:-}" ]      && { add "lie_sim=${LIE_SIM}"; TAG="${TAG}_sim"; }
[ -n "${ACT_GAIN:-}" ]     && { add "act_gain=${ACT_GAIN}"; TAG="${TAG}_ag${ACT_GAIN}"; }
[ -n "${CTRB_W:-}" ]       && { add "ctrb_w=${CTRB_W}"; TAG="${TAG}_ctrb${CTRB_W}"; }
[ -n "${ACT_INFO_NEG:-}" ] && { add "act_info_neg=${ACT_INFO_NEG}"; TAG="${TAG}_${ACT_INFO_NEG}"; }
# round 5 (T1/T2 decoder); unset => upstream behaviour, bit-identical.
# AUX_DECODER, not HAS_DECODER: plan.py rebuilds a decoder for any has_decoder run and
# raises on every checkpoint this repo writes (state_dict + env.decoder_path: null).
[ -n "${AUX_DECODER:-}" ]  && { add "aux_decoder=${AUX_DECODER} model.train_decoder=${AUX_DECODER} decoder=${DECODER:-transposed_conv}"; TAG="${TAG}_dec${DECODER:-transposed_conv}"; }
[ -n "${DECODE_GRAD:-}" ]  && { add "decode_grad=${DECODE_GRAD}"; TAG="${TAG}_dg${DECODE_GRAD}"; }
[ -n "${LAMB_DECODE:-}" ]  && { add "lamb_decode=${LAMB_DECODE}"; TAG="${TAG}_ld${LAMB_DECODE}"; }
# round 8 (S1). Pixel gradient through the PREDICTOR, not just the encoder.
[ -n "${DECODE_PRED_W:-}" ] && { add "decode_pred_w=${DECODE_PRED_W}"; TAG="${TAG}_dpw${DECODE_PRED_W}"; }
# round 5 (T3 contact weighting). CONTACT_GAMMA=0 is the uniform upstream objective.
[ -n "${CONTACT_GAMMA:-}" ] && { add "contact_gamma=${CONTACT_GAMMA}"; TAG="${TAG}_cg${CONTACT_GAMMA}"; }
[ -n "${CONTACT_SHUF:-}" ]  && { add "contact_shuffle=${CONTACT_SHUF}"; TAG="${TAG}_cshuf"; }
# round 5 (T6 option model). NUM_PRED is K; it changes the DATASET window too
# (num_frames = num_hist + num_pred), which is why both arms must set it.
[ -n "${NUM_PRED:-}" ]     && { add "num_pred=${NUM_PRED}"; TAG="${TAG}_K${NUM_PRED}"; }
[ -n "${OVERSHOOT:-}" ]    && { add "overshoot=${OVERSHOOT}"; TAG="${TAG}_os"; }
# round 5 (T4 value head / V4 BC policy). Unset => the head is not even built, so this
# is bit-identical to upstream. ACT_DIM_RAW is derived from FRAMESKIP, not hardcoded, and
# is written into the config so plan.py can rebuild the policy head from the run's saved
# hydra.yaml alone -- without it V4 would plan with a fresh init (the path_int defect).
[ -n "${VALUE_W:-}" ]      && { add "value_w=${VALUE_W}"; TAG="${TAG}_v${VALUE_W}"; }
[ -n "${VALUE_MODE:-}" ]   && { add "value_mode=${VALUE_MODE}"; TAG="${TAG}_${VALUE_MODE}"; }
[ -n "${VALUE_TAU:-}" ]    && { add "value_tau=${VALUE_TAU}"; TAG="${TAG}_tau${VALUE_TAU}"; }
[ -n "${POLICY_W:-}" ]     && { add "policy_w=${POLICY_W} act_dim_raw=$((2 * FRAMESKIP))"; TAG="${TAG}_bc"; }
# round 6 (R6 support / R2 consistency / R3 action-SAM / R4 V1's epsilon). Unset =>
# upstream behaviour, bit-identical on both mlp_var and ltv. Every one of these is a GRID
# knob: the round's rule is that no arm is single-shot on its own strength parameter.
[ -n "${SUPPORT_W:-}" ]    && { add "support_w=${SUPPORT_W}"; TAG="${TAG}_sup${SUPPORT_W}"; }
[ -n "${CONSIST_W:-}" ]    && { add "consist_w=${CONSIST_W}"; TAG="${TAG}_con${CONSIST_W}"; }
[ -n "${CONSIST_SRC:-}" ]  && { add "consist_src=${CONSIST_SRC}"; TAG="${TAG}_${CONSIST_SRC}"; }
[ -n "${CONSIST_K:-}" ]    && { add "consist_k=${CONSIST_K}"; TAG="${TAG}_ck${CONSIST_K}"; }
[ -n "${CONSIST_SIGMA:-}" ] && { add "consist_sigma=${CONSIST_SIGMA}"; TAG="${TAG}_cs${CONSIST_SIGMA}"; }
[ -n "${SAM_RHO:-}" ]      && { add "sam_rho=${SAM_RHO}"; TAG="${TAG}_sam${SAM_RHO}"; }
[ -n "${INCR_EPS:-}" ]     && { add "incr_eps=${INCR_EPS}"; TAG="${TAG}_ie${INCR_EPS}"; }
[ -n "${INCR_CLIP:-}" ]    && { add "incr_clip=${INCR_CLIP}"; TAG="${TAG}_ic${INCR_CLIP}"; }
# round 8 (T2 rung 1). conf/train_rdmreg.yaml:95 sets detach_target: False, so the encoder
# receives gradient THROUGH its own prediction target (visual_world_model.py:1129, :1456) and
# the objective is a two-player game rather than a regression -- the hazard the code documents
# at :581. A JEPA needs a stop-gradient. The ctor default is already True; only the config
# overrides it, and no arm in ~160 runs has ever contrasted the two.
[ -n "${DETACH_TARGET:-}" ] && { add "detach_target=${DETACH_TARGET}"; TAG="${TAG}_dt${DETACH_TARGET}"; }
# round 8 (T4). Masking and dilation over the LAG axis. Unset => upstream exactly:
# p=0 draws no mask, and no dilation means lag slot k reads z_{t-k} as it always has.
# '+' because these keys live on the predictor OBJECT, not in any predictor yaml -- hydra
# requires append syntax for a key the selected config group does not already define, and
# every predictor config would otherwise need the same two lines.
[ -n "${LAG_MASK_P:-}" ]   && { add "+predictor.lag_mask_p=${LAG_MASK_P}"; TAG="${TAG}_lm${LAG_MASK_P}"; }
# quoted: hydra reads a bare comma as a sweep, so "1,2,5" must arrive as one scalar.
[ -n "${LAG_DILATION:-}" ] && { add "+predictor.lag_dilation='${LAG_DILATION}'"; TAG="${TAG}_ld"; }
[ -n "${PREDICTOR:-}" ]  && { add "predictor=${PREDICTOR}"; TAG="${TAG}_${PREDICTOR}"; }
[ -n "${MU:-}" ]         && { add "mu=${MU}"; TAG="${TAG}_mu${MU}"; }
[ "${MUP:-0}" = "1" ]    && { add "mup=true"; TAG="${TAG}_mup"; }
[ -n "${MUP_LR:-}" ]     && { add "training.mup_lr=${MUP_LR}"; TAG="${TAG}_mlr${MUP_LR}"; }
[ -n "${SEED:-}" ]       && { add "training.seed=${SEED}"; TAG="${TAG}_seed${SEED}"; }
[ -n "${PROJ_DIM:-}" ]   && { add "encoder.proj_dim=${PROJ_DIM} action_emb_dim=${PROJ_DIM}"; TAG="${TAG}_pd${PROJ_DIM}"; }
# PATCH_SIZE sets the ViT grid, hence the TOKEN COUNT: num_patches = (img_size/patch_size)^2.
# At img_size=224 that is 14 -> 256 tokens, 56 -> 16, 112 -> 4, 224 -> 1. This is the only knob
# that makes a DIMENSION-MATCHED cls-vs-patch comparison possible: a patch arm carries
# num_patches x proj_dim latent values, so at proj_dim 384 the default patch arm carries 98304
# against a cls arm's 384 -- a 256x capacity gap that "columns vs LpWM-ltv" silently confounds
# with the feature itself (diary/2026-09-05 §2c).
[ -n "${PATCH_SIZE:-}" ] && { add "encoder.patch_size=${PATCH_SIZE}"; TAG="${TAG}_ps${PATCH_SIZE}"; }
[ -n "${SAVE_EVERY:-}" ] && add "training.save_every_x_epoch=${SAVE_EVERY}"
[ -n "${TRAIN_ENCODER:-}" ] && add "model.train_encoder=${TRAIN_ENCODER}"
[ -n "${WANDB_PROJECT:-}" ] && add "wandb_project=${WANDB_PROJECT}"
# Pi-WM intervention flags; unset => upstream behaviour (bit-identical)
[ -n "${KWTA_K:-}" ]     && { add "kwta_k=${KWTA_K}"; TAG="${TAG}_k${KWTA_K}"; }
[ -n "${GATE_INPUT:-}" ] && { add "gate_input=${GATE_INPUT}"; TAG="${TAG}_gi${GATE_INPUT}"; }
[ -n "${GATE_NORM:-}" ]  && { add "gate_norm=${GATE_NORM}"; TAG="${TAG}_gn${GATE_NORM}"; }
[ -n "${N_HEADS:-}" ]    && { add "n_heads=${N_HEADS}"; TAG="${TAG}_J${N_HEADS}"; }
[ -n "${HEAD_ENT:-}" ]   && { add "head_entropy_coef=${HEAD_ENT}"; TAG="${TAG}_ent${HEAD_ENT}"; }
[ -n "${PRECISION:-}" ] && { add "precision=${PRECISION}"; TAG="${TAG}_${PRECISION}"; }
[ "${DEBUG:-0}" = "1" ]  && add "debug=True"
# free-form hydra overrides, e.g. OVERRIDES="env.dataset.n_rollout=40 training.save_every_x_min=1"
[ -n "${OVERRIDES:-}" ]  && add "${OVERRIDES}"
REGULARIZER=${REGULARIZER:-rdmreg}

STAMP=$(date +%Y%m%d-%H%M%S); RAND=$(python3 -c 'import secrets; print(secrets.token_hex(3))')
RUNDIR=${CKPT_BASE}/outputs/lpwm_${LINK}_${FEATURE}_${ENV}_p${TARGET_P}_${AGG}${TAG}_${STAMP}_${RAND}
[ -n "${RUN_NAME:-}" ] && RUNDIR=${CKPT_BASE}/outputs/${RUN_NAME}

cd "${REPO}"
# exec so python replaces this shell: SIGUSR1 from the sbatch preemption trap must
# reach train.py's handler, not an intermediate bash that would just die on it.
exec python train.py --config-name train_rdmreg.yaml \
    env="${ENV}" frameskip="${FRAMESKIP}" num_hist="${NUM_HIST}" \
    encoder="${ENCODER}" link="${LINK}" regularizer="${REGULARIZER}" \
    target_p="${TARGET_P}" agg="${AGG}" \
    training.epochs="${EPOCHS}" training.batch_size="${BATCH}" env.num_workers="${NUM_WORKERS}" \
    ckpt_base_path="${CKPT_BASE}" hydra.run.dir="${RUNDIR}" ${EXTRA}
