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
[ -n "${VAR_SPACE:-}" ]  && add "var_space=${VAR_SPACE}"
[ -n "${PREDICTOR:-}" ]  && { add "predictor=${PREDICTOR}"; TAG="${TAG}_${PREDICTOR}"; }
[ -n "${MU:-}" ]         && { add "mu=${MU}"; TAG="${TAG}_mu${MU}"; }
[ "${MUP:-0}" = "1" ]    && { add "mup=true"; TAG="${TAG}_mup"; }
[ -n "${MUP_LR:-}" ]     && { add "training.mup_lr=${MUP_LR}"; TAG="${TAG}_mlr${MUP_LR}"; }
[ -n "${SEED:-}" ]       && { add "training.seed=${SEED}"; TAG="${TAG}_seed${SEED}"; }
[ -n "${PROJ_DIM:-}" ]   && { add "encoder.proj_dim=${PROJ_DIM} action_emb_dim=${PROJ_DIM}"; TAG="${TAG}_pd${PROJ_DIM}"; }
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
