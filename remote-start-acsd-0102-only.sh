#!/usr/bin/env bash
set -euo pipefail

cd /home/chenlin/krona/anchor-nav
mkdir -p output_logs/tmux

echo "GPU_STATUS_BEGIN"
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
echo "GPU_STATUS_END"

ts="$(date +%Y%m%d-%H%M%S)"
acsd_session="acsd-all-0102-adaptive-level-${ts}"
acsd_log="output_logs/tmux/${acsd_session}.log"
acsd_wrapper="/tmp/${acsd_session}.sh"

cat > "${acsd_wrapper}" <<'EOS'
#!/usr/bin/env bash
set -eo pipefail

cd /home/chenlin/krona/anchor-nav
source /opt/conda/bin/activate
set +u
conda activate mtu3d
set -u

export CUDA_VISIBLE_DEVICES="${ACSD_GPU:-2}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"
VLM_MODEL="${VLM_MODEL:-gpt-4o-mini}"
ACSD_TOP_K="${ACSD_TOP_K:-12}"
ACSD_SEED="${ACSD_SEED:-1234}"
ACSD_CORRECTION_MARGIN="${ACSD_CORRECTION_MARGIN:-0.005}"
ACSD_OBJECT_CORRECTION_MIN_BASELINE_SCORE="${ACSD_OBJECT_CORRECTION_MIN_BASELINE_SCORE:-0.80}"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${TS}-detailed-vlm-anchors-object-only-adaptive-level-policy-v31-0102-instance-exact-target-rescue-margin005-min080-tmux-all-${ACSD_SESSION}"
OUT_DIR="output_logs/anchor/acsd_all_0.1_0.2/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo ">>> ACSD refine1 all-task 0.1-0.2 run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> VLM model: ${VLM_MODEL}"
echo ">>> ACSD seed: ${ACSD_SEED}"
echo ">>> ACSD correction_margin: ${ACSD_CORRECTION_MARGIN}"
echo ">>> ACSD object_correction_min_baseline_score: ${ACSD_OBJECT_CORRECTION_MIN_BASELINE_SCORE}"
echo ">>> ACSD decomposition: VLM-extracted target/room/object anchors"
echo ">>> ACSD correction: object-only origin-anchor rerank; exploration frontiers pass through unchanged"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=detailed"
  echo "task_levels=object,room,region,instance"
  echo "slice_range=0.1-0.2"
  echo "vlm_model=${VLM_MODEL}"
  echo "acsd_top_k=${ACSD_TOP_K}"
  echo "seed=${ACSD_SEED}"
  echo "correction_margin=${ACSD_CORRECTION_MARGIN}"
  echo "object_correction_min_baseline_score=${ACSD_OBJECT_CORRECTION_MIN_BASELINE_SCORE}"
  echo "decomposition=vlm_only"
  echo "correction_scope=object_only"
  echo "frontier_correction=false"
  echo "policy_patch=instance_low_raw_exact_target_rescue"
  echo "schedule=single_shard_single_pipeline"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} > "${OUT_DIR}/run_args.txt"

python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-acsd-refine1.py \
  --start_ratio 0.1 \
  --end_ratio 0.2 \
  --task_levels "object,room,region,instance" \
  --navigation_data_path "${NAVIGATION_DATA_PATH}" \
  --acsd_top_k "${ACSD_TOP_K}" \
  --vlm_model "${VLM_MODEL}" \
  --seed "${ACSD_SEED}" \
  --correction_margin "${ACSD_CORRECTION_MARGIN}" \
  --object_correction_min_baseline_score "${ACSD_OBJECT_CORRECTION_MIN_BASELINE_SCORE}" \
  --output_log_dir "${OUT_DIR}"

echo ">>> Done ACSD all-task [0.1,0.2]. Output: ${OUT_DIR}"
EOS

chmod +x "${acsd_wrapper}"

tmux new-session -d -s "${acsd_session}" \
  "ACSD_SESSION='${acsd_session}' ACSD_GPU='${ACSD_GPU:-2}' bash '${acsd_wrapper}' > '${acsd_log}' 2>&1"

echo "STARTED_ACSD_SESSION=${acsd_session}"
echo "STARTED_ACSD_LOG=${acsd_log}"
echo "STARTED_ACSD_WRAPPER=${acsd_wrapper}"

sleep 2
echo "TMUX_0102_BEGIN"
tmux ls 2>/dev/null | grep -E "(baseline-0102|acsd-all-0102)" || true
echo "TMUX_0102_END"
echo "ACSD_LOG_HEAD_BEGIN"
sed -n '1,28p' "${acsd_log}" 2>/dev/null || true
echo "ACSD_LOG_HEAD_END"
