#!/usr/bin/env bash
# 에이전트 D2 학습 + 평가 (팀장 결정 2026-11-21: D2-a 비율, D와 같은 기준, seed 0·16·32 연속)
#   seed 0 → 16 → 32 학습 → 끝나면 평가(e3, 20판, 평가 시드 84, best 모델) + 판정 출력
# 시드당 약 2.6~2.8시간(C·D 기준). 주의: 실행 중 Claude 앱을 닫으면 WSL이 꺼져 학습이 죽음 (최소화는 괜찮음)
#
# 실행 (저장소 루트):  bash project2/run_d2.sh            # seed 0, 16, 32
#                      bash project2/run_d2.sh 32         # 일부 시드만
set -u
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/scobots/bin/python
LOGDIR=/mnt/d/my_Assignment/IoT_Project2/logs
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=(0 16 32); fi
SUM="$LOGDIR/train_D2.log"

echo "########## D2 학습 시작: $(date "+%F %T")  seeds=${SEEDS[*]} ##########" | tee -a "$SUM"
for s in "${SEEDS[@]}"; do
  dir="resources/checkpoints/Pong_seed${s}_reward-env_oc_pruned-external_D2-freeze"
  if [ -e "$dir" ]; then
    echo ">>> seed$s 학습 건너뜀: $dir 이미 있음" | tee -a "$SUM"
  else
    echo ">>> D2 seed$s 학습 시작: $(date "+%F %T")" | tee -a "$SUM"
    SCOBI_EXCLUDE_OBJECTS=Enemy PYTHONPATH=. "$PY" -u -W ignore project2/train_d2.py \
        -g Pong -s "$s" -env 8 -r env -p external > "$LOGDIR/train_D2_seed${s}.log" 2>&1
    echo ">>> D2 seed$s 학습 종료: $(date "+%F %T")  exit=$?" | tee -a "$SUM"
  fi
done

echo ">>> D2 평가 시작: $(date "+%F %T")" | tee -a "$SUM"
for s in "${SEEDS[@]}"; do
  if [ -e "project2/results/diag_ckpt/e3_D2_seed${s}.jsonl" ]; then
    echo ">>> seed$s 평가 건너뜀: 이미 있음" | tee -a "$SUM"; continue
  fi
  PYTHONPATH=. "$PY" -u -W ignore project2/hackatari_diag_ckpt.py --agent D2 --seed "$s" --ckpts best \
      --conds base Lazy LazyNoBallX LazyOld Perturb --episodes 20 --tag e3 > "$LOGDIR/diag_e3_D2_seed${s}.log" 2>&1 &
done
wait
PYTHONPATH=. "$PY" project2/hackatari_diag_ckpt_analyze.py > "$LOGDIR/analyze_D2.log" 2>&1
sed -n "/### D2 (/,/최종/p" "$LOGDIR/analyze_D2.log" | tee -a "$SUM"
echo "D2_DONE: $(date "+%F %T")" | tee -a "$SUM"
