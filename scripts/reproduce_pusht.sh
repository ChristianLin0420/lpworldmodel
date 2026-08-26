#!/bin/bash
# Reproduce the PushT sparsity-vs-linearity grid: LpWM (sparse) vs LeWM (dense
# SWD-Gaussian) x the predictor-complexity ladder x latent dim D.
#
# Cluster-agnostic driver: for each cell it runs scripts/train.sh then scripts/plan.sh
# SEQUENTIALLY on the local machine (no scheduler). On a cluster, wrap each cell in
# your own job submission instead. Each cell pins the (reg_weight, mup_lr) used in the
# paper; Deep-AdaLN(k) (ar_adaln) uses the robust (0.5, 1e-4) default (see README).
#
#   bash scripts/reproduce_pusht.sh                # dry-run: print all cells + commands
#   RUN=1 bash scripts/reproduce_pusht.sh          # actually run (train + eval per cell)
#   ARM_LIST=sparse RUN=1 bash scripts/reproduce_pusht.sh                    # just sparse
#   PRED_LIST="mlp_var" PD_LIST="4096" RUN=1 bash scripts/reproduce_pusht.sh # narrow
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN="${HERE}/train.sh"; PLAN="${HERE}/plan.sh"

ARM_LIST=${ARM_LIST:-"sparse dense"}
PRED_LIST=${PRED_LIST:-"ar_adaln ar_adaln_d1 ltv mlp_var linear_var linear_wb"}
PD_LIST=${PD_LIST:-"384 768 1536 2048 4096"}
EPOCHS=${EPOCHS:-2}
NEVALS=${NEVALS:-50}; MAXITER=${MAXITER:-10}
DRYRUN=1; [ "${RUN:-0}" = "1" ] && DRYRUN=0

# Per-cell (reg_weight mup_lr), keyed "arm:pred:D".
# ar_adaln (Deep-AdaLN) uses the robust fine-grained default (0.5, 1e-4) for every arm/D.
declare -A HP
# --- sparse (LpWM, mu=0) ---
HP[sparse:ar_adaln:384]="0.5 1e-4";    HP[sparse:ar_adaln:768]="0.5 1e-4";   HP[sparse:ar_adaln:1536]="0.5 1e-4";  HP[sparse:ar_adaln:2048]="0.5 1e-4";  HP[sparse:ar_adaln:4096]="0.5 1e-4"
HP[sparse:ar_adaln_d1:384]="0.1 5e-4"; HP[sparse:ar_adaln_d1:768]="0.1 5e-4"; HP[sparse:ar_adaln_d1:1536]="0.1 5e-4"; HP[sparse:ar_adaln_d1:2048]="1.0 5e-4"; HP[sparse:ar_adaln_d1:4096]="1.0 5e-4"
HP[sparse:mlp_var:384]="0.1 5e-4";     HP[sparse:mlp_var:768]="1.0 5e-4";    HP[sparse:mlp_var:1536]="1.0 5e-4";   HP[sparse:mlp_var:2048]="1.0 5e-4";   HP[sparse:mlp_var:4096]="1.0 5e-4"
HP[sparse:ltv:384]="1.0 5e-4";         HP[sparse:ltv:768]="1.0 5e-4";        HP[sparse:ltv:1536]="1.0 5e-4";       HP[sparse:ltv:2048]="1.0 5e-4";       HP[sparse:ltv:4096]="1.0 5e-4"
HP[sparse:linear_var:384]="0.1 5e-5";  HP[sparse:linear_var:768]="0.1 5e-5"; HP[sparse:linear_var:1536]="0.1 5e-5"; HP[sparse:linear_var:2048]="0.1 5e-4"; HP[sparse:linear_var:4096]="0.1 5e-4"
HP[sparse:linear_wb:384]="1.0 5e-4";   HP[sparse:linear_wb:768]="1.0 5e-4";  HP[sparse:linear_wb:1536]="1.0 5e-4"; HP[sparse:linear_wb:2048]="1.0 5e-4"; HP[sparse:linear_wb:4096]="1.0 5e-4"
# --- dense (LeWM, SWD-Gaussian) ---
HP[dense:ar_adaln:384]="0.5 1e-4";     HP[dense:ar_adaln:768]="0.5 1e-4";    HP[dense:ar_adaln:1536]="0.5 1e-4";   HP[dense:ar_adaln:2048]="0.5 1e-4";   HP[dense:ar_adaln:4096]="0.5 1e-4"
HP[dense:ar_adaln_d1:384]="0.1 5e-4";  HP[dense:ar_adaln_d1:768]="0.1 5e-4"; HP[dense:ar_adaln_d1:1536]="0.1 5e-4"; HP[dense:ar_adaln_d1:2048]="1.0 5e-4"; HP[dense:ar_adaln_d1:4096]="0.1 5e-4"
HP[dense:mlp_var:384]="0.01 5e-5";     HP[dense:mlp_var:768]="0.01 5e-5";    HP[dense:mlp_var:1536]="0.01 5e-5";   HP[dense:mlp_var:2048]="0.01 5e-5";   HP[dense:mlp_var:4096]="0.1 5e-4"
HP[dense:ltv:384]="0.1 5e-4";          HP[dense:ltv:768]="10.0 5e-5";        HP[dense:ltv:1536]="0.1 5e-4";        HP[dense:ltv:2048]="0.1 5e-5";        HP[dense:ltv:4096]="0.1 5e-5"
HP[dense:linear_var:384]="0.1 5e-4";   HP[dense:linear_var:768]="0.1 5e-4";  HP[dense:linear_var:1536]="0.1 5e-4"; HP[dense:linear_var:2048]="0.1 5e-4"; HP[dense:linear_var:4096]="0.1 5e-4"
HP[dense:linear_wb:384]="0.1 5e-4";    HP[dense:linear_wb:768]="0.1 5e-4";   HP[dense:linear_wb:1536]="0.1 5e-4";  HP[dense:linear_wb:2048]="0.1 5e-4";  HP[dense:linear_wb:4096]="0.1 5e-4"

count=0
for ARM in ${ARM_LIST}; do
  case "${ARM}" in
    sparse) LINK=reprelu;  TP=1;;
    dense)  LINK=identity; TP=2;;
    *) echo "ARM must be sparse|dense" >&2; exit 1;;
  esac
  for PRED in ${PRED_LIST}; do
    for PD in ${PD_LIST}; do
      key="${ARM}:${PRED}:${PD}"
      cell="${HP[$key]:-}"
      if [ -z "${cell}" ]; then echo "NO HP for ${key}" >&2; exit 1; fi
      read -r RW MLR <<< "${cell}"
      count=$((count+1))
      run="repro_${ARM}_pusht_${PRED}_pd${PD}"
      MU_ENV=""; [ "${ARM}" = "sparse" ] && MU_ENV="MU=0"
      if [ "${DRYRUN}" = "1" ]; then
        printf '[%02d] %-42s reg_weight=%-5s mup_lr=%-5s link=%-8s p=%s\n' "${count}" "${run}" "${RW}" "${MLR}" "${LINK}" "${TP}"
        printf '     train: PREDICTOR=%s PROJ_DIM=%s MUP=1 MUP_LR=%s REG_WEIGHT=%s %s RUN_NAME=%s scripts/train.sh pusht 5 3 %s 64 %s cls %s b\n' \
               "${PRED}" "${PD}" "${MLR}" "${RW}" "${MU_ENV}" "${run}" "${EPOCHS}" "${LINK}" "${TP}"
        printf '     eval:  scripts/plan.sh plan_lewm.yaml %s latest %s %s\n' "${run}" "${NEVALS}" "${MAXITER}"
      else
        echo "[run ${count}] ${run}  (reg_weight=${RW} mup_lr=${MLR})"
        env PREDICTOR="${PRED}" PROJ_DIM="${PD}" MUP=1 MUP_LR="${MLR}" REG_WEIGHT="${RW}" \
            REGULARIZER=rdmreg RUN_NAME="${run}" ${MU_ENV} \
            "${TRAIN}" pusht 5 3 "${EPOCHS}" 64 "${LINK}" cls "${TP}" b
        "${PLAN}" plan_lewm.yaml "${run}" latest "${NEVALS}" "${MAXITER}"
      fi
    done
  done
done
echo "----"
[ "${DRYRUN}" = "1" ] && echo "DRY-RUN: ${count} cells (train + eval each). Set RUN=1 to execute." \
                      || echo "Ran ${count} cells (train + eval each)."
