#!/usr/bin/env bash
# 批量渲染所有 scene 的俯瞰图，分 N_WORKERS 组并行
set -eo pipefail

WORKER_ID="${1:-0}"   # 保存 $1，之后 source activate 前会清掉 $@

SCENES=(
  00800-TEEsavR23oF 00802-wcojb4TFT35 00803-k1cupFYWXJ6 00808-y9hTuugGdiq
  00810-CrMo8WxCyVb 00813-svBbv1Pavdk 00814-p53SfW6mjZe 00815-h1zeeAwLh9Z
  00820-mL8ThkuaVTM 00821-eF36g7L6Z9M 00823-7MXmsvcQjpJ 00824-Dd4bFSTQ8gi
  00827-BAbdmeyTvMZ 00829-QaLdnwvtxbs 00831-yr17PDCnDDW 00832-qyAac8rV8Zk
  00835-q3zU7Yy5E5s 00839-zt1RVoi7PcG 00843-DYehNKdT76V 00844-q5QZSEeHe5g
  00847-bCPU9suPUw9 00848-ziup5kvtCCR 00849-a8BtkwhxdRV 00853-5cdEh9F2hJL
  00861-GLAQ4DNUx5U 00862-LT9Jq6dN3Ea 00869-MHPLjHsuG27 00871-VBzV5z6i1WS
  00873-bxsVRursffK 00876-mv2HUxq3B53 00877-4ok3usBNeis 00878-XB4GS9ShBRE
  00880-Nfvxx8J5NCo 00890-6s7QHgap2fW 00891-cvZr5TUy5C5 00894-HY1NcmCgn3n
)

N_WORKERS=4
OUT_DIR="visual-output/topdown_rgb"
LOG_DIR="visual-output/topdown_logs"
mkdir -p "${LOG_DIR}"

# Activate env — clear $@ first so source activate 不会把 WORKER_ID 当环境名
set --
source "/opt/conda/bin/activate"
set +u
conda activate mtu3d
set -u

export PYTHONPATH="/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

total=${#SCENES[@]}
for i in "${!SCENES[@]}"; do
    if (( i % N_WORKERS != WORKER_ID )); then continue; fi
    scene="${SCENES[$i]}"
    out_png="${OUT_DIR}/${scene}/topdown_rgb.png"
    if [ -f "${out_png}" ]; then
        echo "[worker${WORKER_ID}] SKIP ${scene} (already done)"
        continue
    fi
    echo "[worker${WORKER_ID}] ($((i+1))/${total}) Rendering ${scene} ..."
    python3 visual-scripts/vis_topdown_rgb.py \
        --scene_name "${scene}" \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/${scene}.log"
    echo "[worker${WORKER_ID}] done ${scene}"
done
echo "[worker${WORKER_ID}] ALL DONE"
