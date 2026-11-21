"""
viper_interpret.py — VIPER 결정트리가 '무엇을 보고 판단하는지' 분석

[코드 설명]
viper_extract.py 로 추출된 결정트리(.viper, sklearn DecisionTreeClassifier)를 불러오고,
관측 벡터의 각 입력이 어떤 개념인지(예: Enemy1.y, Ball1.y) 이름표를 가져옵니다.
트리가 적(Enemy) 위치로 분기하면 지름길의 '트리 수준 증거'가 됩니다.
(논문 Fig.4: A는 적·자기 패들의 수직위치로 판단, 공은 대체로 무시)

[확장] 확장 단계에서 학습한 C·D 의 트리도 같은 방식으로 분석합니다. (TODO 3)
"""
import argparse
import glob
import os
import subprocess
import sys
import numpy as np
from joblib import load
from sklearn.tree import export_text
from scobi import Environment

AGENTS = {
    "A": {
        "label": "A (적 포함: default)",
        "ckpt": "Pong_seed0_reward-env_oc",
        "focus": "default_focus_Pong-v5.yaml",
    },
    "B": {
        "label": "B (적 제거: pruned-external)",
        "ckpt": "Pong_seed0_reward-env_oc_pruned-external",
        "focus": "pruned_pong.yaml",
    },
}

# ================= 확장 단계 =================
# C·D — B처럼 적을 뺀 입력 (SCOBI_EXCLUDE_OBJECTS=Enemy 필요)
AGENTS["C0"] = {"label": "C seed0", "ckpt": "Pong_seed0_reward-env_oc_pruned-external_C-mix", "focus": "pruned_pong.yaml"}
AGENTS["C16"] = {"label": "C seed16", "ckpt": "Pong_seed16_reward-env_oc_pruned-external_C-mix", "focus": "pruned_pong.yaml"}
AGENTS["D16"] = {"label": "D seed16", "ckpt": "Pong_seed16_reward-env_oc_pruned-external_D-freeze", "focus": "pruned_pong.yaml"}
AGENTS["D32"] = {"label": "D seed32", "ckpt": "Pong_seed32_reward-env_oc_pruned-external_D-freeze", "focus": "pruned_pong.yaml"}

# ===== TODO 3 (확장) =====
"""
AGENTS 에 확장 단계에서 학습한 C·D 4개가 추가되어 있습니다.
  C0, D16  : LazyEnemy 에서도 점수가 유지된 에이전트
  C16, D32 : LazyEnemy 에서 점수가 크게 떨어진 에이전트
  → 넷 다 B 처럼 적을 뺀 입력이라 SCOBI_EXCLUDE_OBJECTS=Enemy 로 실행해야 합니다.
    (--agent C0 처럼 실행하면 자동으로 설정됩니다)

1) viper_extract.py 로 4개의 트리를 추출해 주세요.
   ex)
   SCOBI_EXCLUDE_OBJECTS=Enemy python viper_extract.py -i resources/checkpoints/Pong_seed16_reward-env_oc_pruned-external_D-freeze -r viper
   → 나머지 3개도 같은 형식으로 추출해 주세요.
2) analyze_agent 로 4개를 분석하고, 점수가 유지된 쪽과 떨어진 쪽의 트리가 어떻게 다른지 비교해 주세요.
   ex)
   python project2/viper_interpret.py --agent D16
   → 상위 피처나 분기 규칙에서 두 묶음의 차이를 print 로 정리해 주세요.
3) 추출한 *_best.viper 파일도 함께 커밋해 주세요.
"""


def analyze_agent(agent_key: str):
    cfg = AGENTS[agent_key]
    ckpt = cfg["ckpt"]
    focus = cfg["focus"]

    print("=" * 60)
    print(f"분석 대상: 에이전트 {cfg['label']}")
    print("=" * 60)

    paths = glob.glob(
        f"resources/viper_extracts/extract_output/{ckpt}-*extraction/*_best.viper"
    )
    assert len(paths) == 1, f"트리 후보가 {len(paths)}개입니다(1개여야 함): {paths}"
    tree_path = paths[0]
    clf = load(tree_path)
    print("트리:", tree_path.split("/")[-1], "| 노드수:", clf.tree_.node_count)

    env = Environment(
        "ALE/Pong-v5",
        focus_dir=f"resources/checkpoints/{ckpt}",
        focus_file=focus,
        silent=True,
        refresh_yaml=False,
    )
    env.reset(seed=0)
    feat_names = [str(f) for f in env.get_vector_entry_descriptions()]
    assert clf.n_features_in_ == len(feat_names), (
        "트리와 피처 이름 개수가 다릅니다 — 다른 에이전트의 트리이거나, "
        "B인데 SCOBI_EXCLUDE_OBJECTS=Enemy 를 빼먹었는지 확인하세요"
    )

    # ===== TODO 1: 중요도 순 정렬 및 출력 =====
    print("\n[1] 주요 피처 중요도 (0 초과)")
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]

    for i in order:
        if imp[i] > 0:
            print(f"{feat_names[i]:<30} : {imp[i]:.4f}")

    # ===== TODO 2: Enemy vs Ball 의존도 비교 및 판정 =====
    print("\n[2] 객체별 중요도 합산 및 지름길 여부 판정")
    enemy_features = [feat_names[i] for i in order if imp[i] > 0 and "Enemy" in feat_names[i]]
    enemy_sum = sum(imp[i] for i in order if "Enemy" in feat_names[i])
    ball_sum = sum(imp[i] for i in order if "Ball" in feat_names[i])
    player_sum = sum(imp[i] for i in order if "Player" in feat_names[i])

    print(f"- 사용된 Enemy 관련 피처: {enemy_features if enemy_features else '없음'}")
    print(f"- Enemy 피처 중요도 합 : {enemy_sum:.4f}")
    print(f"- Ball  피처 중요도 합 : {ball_sum:.4f}")
    print(f"- Player피처 중요도 합 : {player_sum:.4f}")

    if enemy_sum > ball_sum:
        print("=> 판정: 지름길 징후: 적 위치에 의존")
    else:
        print("=> 판정: 공/자기 패들 기반")

    # ===== 추가 분석: 실제 결정트리 상위 분기 규칙 출력 =====
    print("\n[3] 결정트리 상위 분기 규칙 (Depth 3까지)")
    print(export_text(clf, feature_names=feat_names, max_depth=3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=list(AGENTS) + ["both"], default=None)
    args = parser.parse_args()

    # SCOBI_EXCLUDE_OBJECTS=Enemy 환경변수를 직접 지정해 실행한 경우 B만 단독 분석
    if args.agent is None:
        target = "B" if "Enemy" in os.environ.get("SCOBI_EXCLUDE_OBJECTS", "") else "both"
    else:
        target = args.agent

    if target == "both":
        analyze_agent("A")
        env_b = os.environ.copy()
        env_b["SCOBI_EXCLUDE_OBJECTS"] = "Enemy"
        subprocess.run([sys.executable, __file__, "--agent", "B"], env=env_b, check=True)
    elif target != "A":  # B 와 확장 C·D 는 적을 뺀 입력
        if "Enemy" not in os.environ.get("SCOBI_EXCLUDE_OBJECTS", ""):
            env_b = os.environ.copy()
            env_b["SCOBI_EXCLUDE_OBJECTS"] = "Enemy"
            subprocess.run([sys.executable, __file__, "--agent", target], env=env_b, check=True)
        else:
            analyze_agent(target)
    else:
        analyze_agent("A")