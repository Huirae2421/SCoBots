#!/usr/bin/env bash
# 확장 E10: 체크포인트 다시 고르기 (학습 없음). 설명은 project2/select_ckpt.py
set -u
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/scobots/bin/python
L=/mnt/d/my_Assignment/IoT_Project2/logs
SUM="$L/select_E10.log"
R(){ PYTHONPATH=. "$PY" -u -W ignore project2/hackatari_diag_ckpt.py "$@"; }
CK=$(seq 1 20)
echo "### E10 시작 $(date "+%F %T")" | tee -a "$SUM"
for job in "D 0" "D 16" "D 32" "D2 0" "D2 16" "D2 32"; do
  set -- $job
  if [ -e "project2/results/diag_ckpt/sel_$1_seed$2.jsonl" ]; then
    echo "sel $1 s$2 이미 있음" | tee -a "$SUM"; continue
  fi
  R --agent "$1" --seed "$2" --ckpts $CK --conds base Freeze --episodes 10 --eval-seed 42 --tag sel > "$L/sel_$1_seed$2.log" 2>&1 &
done
wait
echo "### 1단계(고르기 평가) 끝 $(date "+%F %T")" | tee -a "$SUM"
PYTHONPATH=. "$PY" project2/select_ckpt.py pick > "$L/select_E10_pick.txt"
cat "$L/select_E10_pick.txt" | tee -a "$SUM"
grep "^CHOICE" "$L/select_E10_pick.txt" | while read -r _ ag s ck; do
  if [ ! -e "project2/results/diag_ckpt/self_${ag}_seed${s}.jsonl" ]; then
    R --agent "$ag" --seed "$s" --ckpts "$ck" --conds base Lazy LazyNoBallX --episodes 20 --eval-seed 84 --tag self > "$L/self_${ag}_seed${s}.log" 2>&1 &
  fi
done
wait
# while 서브셸 안의 백그라운드 작업은 위 wait 로 기다려지지 않으므로 실제 프로세스가 끝날 때까지 확인
while ps -eo args | grep -q "[h]ackatari_diag_ckpt.py --agent .* --tag self"; do sleep 20; done
echo "### 2단계(최종 평가) 끝 $(date "+%F %T")" | tee -a "$SUM"
PYTHONPATH=. "$PY" project2/select_ckpt.py report | tee -a "$SUM"
echo "E10_DONE $(date "+%F %T")" | tee -a "$SUM"
