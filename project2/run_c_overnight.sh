#!/usr/bin/env bash
# 에이전트 C 학습: seed 0 -> 16 -> 32 한 시드씩 이어서 (팀장 결정 2026-11-19)
# 시드당 약 2.2~2.5시간(B 기준), 전체 약 7~8시간.
# 주의: 실행 중 Claude 앱을 닫으면 WSL이 종료되어 학습이 죽음 (최소화는 괜찮음)
#
# 실행 (저장소 루트에서):   bash project2/run_c_overnight.sh
# 일부 시드만:              bash project2/run_c_overnight.sh 16 32
set -u
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/scobots/bin/python
LOGDIR=/mnt/d/my_Assignment/IoT_Project2/logs
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=(0 16 32); fi
SUM="$LOGDIR/train_C_overnight.log"

echo "########## C 학습 시작: $(date "+%F %T")  seeds=${SEEDS[*]} ##########" | tee -a "$SUM"
for s in "${SEEDS[@]}"; do
  dir="resources/checkpoints/Pong_seed${s}_reward-env_oc_pruned-external_C-mix"
  if [ -e "$dir" ]; then
    echo ">>> seed$s 건너뜀: $dir 이미 있음 (지우거나 이름을 바꾼 뒤 다시 실행)" | tee -a "$SUM"
    continue
  fi
  echo ">>> C seed$s 시작: $(date "+%F %T")" | tee -a "$SUM"
  SCOBI_EXCLUDE_OBJECTS=Enemy PYTHONPATH=. "$PY" -u -W ignore project2/train_c.py \
      -g Pong -s "$s" -env 8 -r env -p external > "$LOGDIR/train_C_seed${s}.log" 2>&1
  rc=$?
  echo ">>> C seed$s 종료: $(date "+%F %T")  exit=$rc" | tee -a "$SUM"
done
echo "C_ALL_DONE: $(date "+%F %T")" | tee -a "$SUM"
