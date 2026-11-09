# SCoBots-Pong — 강화학습 에이전트의 '지름길 학습' 재현 및 분석

> 개념 병목(Concept Bottleneck) 기반 강화학습 에이전트가 Pong에서 보이는 **목표 오정렬(지름길) 현상**을 재현하고, 그 원인과 교정 효과를 분석하는 프로젝트

---

## 프로젝트 소개

**SCoBots-Pong**은 NeurIPS 2024 논문 *"Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents"* (Delfosse et al.)의 핵심 발견인 **Pong 지름길(shortcut) 학습**을 CPU 환경에서 재현하고 확장한 프로젝트입니다.

Pong에서 내장 상대(적) AI는 공을 따라다니기 때문에, **적 패들의 위치가 공의 위치를 거의 그대로 대신**합니다. 그 결과 에이전트는 정작 봐야 할 **공 대신 '적 패들'을 보고** 플레이하는 오정렬을 학습합니다. 표준 Pong 점수만 보면 멀쩡해 보이지만, 적을 없애면 무너지는 숨은 결함입니다.

논문의 교정법은 단순합니다 — **적 객체를 아예 제거**하면 에이전트는 공을 추적할 수밖에 없습니다. 본 프로젝트는 **적을 포함한 에이전트(A)** 와 **적을 제거한 에이전트(B)** 를 각각 학습해, 지름길의 존재·원인·교정 효과를 **점수 / 결정트리 / 상관분석 / 개입실험**으로 검증합니다.

외부 LLM·GPU 없이 **CPU만으로 로컬(WSL2)에서 독립 실행**됩니다.

---

## 주요 내용

- **에이전트 2종 × 3시드 학습**: A(적 포함) / B(적 제거)를 각각 seed 0·16·32로 20M 스텝 학습 (PPO)
- **표준 Pong 재현**: 두 에이전트 점수가 논문 Table 1 범위 안에 들어오는지 검증
- **지름길 증명(개입실험)**: 관측에서 적 피처만 중립화했을 때 A가 붕괴하는지 측정
- **결정트리 해석(VIPER)**: 학습된 정책을 결정트리로 증류해, **트리가 적 위치로 분기**하는지 확인
- **상관 분석**: 적 패들 ↔ 공의 수직위치 상관계수를 재현
- **학습 시각화**: 3시드 학습곡선·비교표를 그래프로 출력

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| Language | Python 3.8 |
| 강화학습 | Stable-Baselines3 2.0.0 (PPO, MLP 2×64) |
| 객체 탐지 | OCAtari |
| 환경 | ale-py 0.10.1, gymnasium 0.28.1 (ALE/Pong-v5) |
| 트리 증류 | VIPER (scikit-learn DecisionTree) |
| 분석·시각화 | SciPy(상관), matplotlib(차트) |
| 실행 환경 | **CPU 전용** (WSL2 / Ubuntu) |
| 기반 코드 | k4ntz/SCoBots (MIT License) — 포크 후 수정 |

---

## 프로젝트 구조

```
scobots/                       # k4ntz/SCoBots 포크 (MIT License)
├── README.md                  # 이 문서
├── README_SCoBots.md          # 원본 SCoBots README (보존)
├── scobi/  utils/             # 원본 SCoBots 코드
│   ├── scobi/core.py          # [수정] 환경변수로 객체 제외(적 제거용)
│   ├── utils/viper.py         # [수정] CUDA → CPU 폴백
│   └── viper_extract.py       # [수정] device=cpu
├── eval.py  train.py  ...     # 원본 (평가·학습)
├── resources/checkpoints/     # 학습된 에이전트 (A / B, seed 0·16·32)
└── project2/                  # 본 프로젝트 구현 코드
    ├── train_long.py          # 학습 스크립트 (train.py + 옵션 추가)
    ├── correlation.py         # 적↔공 상관계수
    ├── plot_curves.py         # A·B 학습곡선
    └── viper_interpret.py     # 결정트리 분기 피처 분석
```

> 원본 파일 3곳을 수정했으며(위 `[수정]`), 나머지 새 파일은 모두 `project2/`에 모았습니다.

---

## 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/<본인>/SCoBots.git
cd SCoBots

# 2. 환경 구성 (WSL2 + conda, Python 3.8)
conda create -n scobots python=3.8 -y
conda activate scobots
pip install -r requirements.txt

# 3. 분석 실행 (저장소 루트에서, import 때문에 PYTHONPATH 지정)
PYTHONPATH=. python project2/correlation.py       # 상관계수
PYTHONPATH=. python project2/plot_curves.py        # 학습곡선
PYTHONPATH=. python project2/viper_interpret.py    # 결정트리 해석
```

- 요구 환경: Windows + WSL2(Ubuntu), Python 3.8
- 학습된 모델(체크포인트)은 `resources/checkpoints/` 아래에 있어야 합니다.

---

## 팀 구성

| 역할 | 이름 | 학번 |
|---|---|---|
| 팀장 | 김희래 | 20211873 |
| 팀원 | 이호민 | 20222040 |
| 팀원 | 전도현 | 20231993 |
