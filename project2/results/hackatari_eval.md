# HackAtari 평가 결과

| 항목 | 내용 |
|---|---|
| 스크립트 | `project2/hackatari_eval.py --episodes 20` |
| 실행일 | 2026-11-12 (리더 PC) |
| 에이전트 | A = `Pong_seed{0,16,32}_reward-env_oc` (적 포함) / B = `Pong_seed{0,16,32}_reward-env_oc_pruned-external` (적 제거, `SCOBI_EXCLUDE_OBJECTS=Enemy`) |
| 모델 | 각 체크포인트의 `best_model` + `best_vecnormalize.pkl`, 결정적 행동 |
| 평가 | 조건별 20에피소드, 평가 시드 84, 보상 = 게임 점수 |
| 조건 | base (수정 없음) / NoEnemy (`hidden_enemy`) / LazyEnemy (`lazy_enemy`) |
| HackAtari | k4ntz/HackAtari @ `8ca2bdf` |
| 환경 | Python 3.8, gymnasium 0.28.1, ale-py 0.10.1, stable-baselines3 2.0.0, ocatari 2.0.0, torch 2.4.1+cpu |

## 시드별 결과
에피소드 보상의 평균 ± 표준편차 (20에피소드, 표준편차 ddof=0)

| 에이전트 | 시드 | base | NoEnemy | LazyEnemy |
|---|---|---|---|---|
| A | 0 | 17.10 ± 1.87 | -21.00 ± 0.00 | -21.00 ± 0.00 |
| A | 16 | 16.20 ± 2.54 | -21.00 ± 0.00 | -20.35 ± 0.57 |
| A | 32 | 19.40 ± 1.62 | -21.00 ± 0.00 | -20.15 ± 1.06 |
| B | 0 | 17.80 ± 2.25 | 17.80 ± 2.25 | 17.15 ± 1.98 |
| B | 16 | 19.35 ± 2.03 | 19.35 ± 2.03 | 2.40 ± 6.32 |
| B | 32 | 19.40 ± 1.36 | 19.40 ± 1.36 | 5.40 ± 6.63 |

## 시드 평균
시드별 평균값 3개의 평균 ± 표준편차 (ddof=0)

| 에이전트 | base | NoEnemy | LazyEnemy |
|---|---|---|---|
| A | 17.57 ± 1.35 | -21.00 ± 0.00 | -20.50 ± 0.36 |
| B | 18.85 ± 0.74 | 18.85 ± 0.74 | 8.32 ± 6.37 |
