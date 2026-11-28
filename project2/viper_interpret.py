"""
viper_interpret.py — VIPER 결정트리가 '무엇을 보고 판단하는지' 분석

[코드 설명]
viper_extract.py 로 추출된 결정트리(.viper, sklearn DecisionTreeClassifier)를 불러오고,
관측 벡터의 각 입력이 어떤 개념인지(예: Enemy1.y, Ball1.y) 이름표를 가져옵니다.
트리가 적(Enemy) 위치로 분기하면 지름길의 '트리 수준 증거'가 됩니다.
(논문 Fig.4: A는 적·자기 패들의 수직위치로 판단, 공은 대체로 무시)

[확장] 확장 단계에서 학습한 C·D 의 트리도 같은 방식으로 분석합니다. (TODO 3)
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

CD_MAINTAINED = ["C0", "D16"]  # LazyEnemy에서도 점수가 유지된 에이전트
CD_DROPPED = ["C16", "D32"]    # LazyEnemy에서 점수가 크게 떨어진 에이전트


def _get_root_rule(clf, feat_names: list[str]) -> str:
    """루트 노드(depth 0)의 첫 분기 조건 문자열 반환"""
    tree = clf.tree_
    if tree.feature[0] < 0:
        return "Leaf (분기 없음)"
    fname = feat_names[tree.feature[0]]
    thresh = tree.threshold[0]
    return f"{fname} <= {thresh:.2f}"


def analyze_agent(agent_key: str) -> dict:
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
    print("트리:", os.path.basename(tree_path), "| 노드수:", clf.tree_.node_count, "| 깊이:", clf.get_depth())

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

    top_features = []
    for i in order:
        if imp[i] > 0:
            print(f"{feat_names[i]:<30} : {imp[i]:.4f}")
            top_features.append((feat_names[i], float(imp[i])))

    # ===== TODO 2: Enemy vs Ball 의존도 비교 및 판정 =====
    print("\n[2] 객체별 중요도 합산 및 지름길 여부 판정")
    enemy_features = [feat_names[i] for i in order if imp[i] > 0 and "Enemy" in feat_names[i]]
    enemy_sum = float(sum(imp[i] for i in order if "Enemy" in feat_names[i]))
    ball_sum = float(sum(imp[i] for i in order if "Ball" in feat_names[i]))
    player_sum = float(sum(imp[i] for i in order if "Player" in feat_names[i]))

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
    tree_rules_d3 = export_text(clf, feature_names=feat_names, max_depth=3)
    print(tree_rules_d3)

    return {
        "key": agent_key,
        "label": cfg["label"],
        "tree_file": os.path.basename(tree_path),
        "node_count": int(clf.tree_.node_count),
        "depth": int(clf.get_depth()),
        "feat_names": feat_names,
        "imp_map": {feat_names[i]: float(imp[i]) for i in range(len(feat_names))},
        "top_features": top_features,
        "enemy_sum": enemy_sum,
        "ball_sum": ball_sum,
        "player_sum": player_sum,
        "root_rule": _get_root_rule(clf, feat_names),
        "tree_rules_d2": export_text(clf, feature_names=feat_names, max_depth=2),
    }


def compare_cd_agents():
    """LazyEnemy 점수 유지군(C0, D16) vs 하락군(C16, D32) 트리 비교 분석"""
    results = {}
    for key in CD_MAINTAINED + CD_DROPPED:
        results[key] = analyze_agent(key)

    print("\n" + "=" * 76)
    print(" [TODO 3 요약] LazyEnemy 점수 유지군(C0, D16) vs. 하락군(C16, D32) 트리 비교")
    print("=" * 76)

    header = f"{'구분':<10} {'에이전트':<12} {'노드수':>6} {'깊이':>4} {'Ball합':>8} {'Player합':>9}  {'루트 분기(Depth 0)'}"
    print(header)
    print("-" * 76)

    for group_name, keys in [("점수 유지", CD_MAINTAINED), ("점수 하락", CD_DROPPED)]:
        for k in keys:
            r = results[k]
            print(
                f"{group_name:<10} {r['label']:<12} {r['node_count']:>6} {r['depth']:>4} "
                f"{r['ball_sum']:>8.4f} {r['player_sum']:>9.4f}  {r['root_rule']}"
            )

    print("\n[에이전트별 상위 3개 핵심 피처]")
    for group_name, keys in [("점수 유지", CD_MAINTAINED), ("점수 하락", CD_DROPPED)]:
        print(f"● {group_name} ({', '.join(keys)})")
        for k in keys:
            top3_str = ", ".join(f"{name}({val:.3f})" for name, val in results[k]["top_features"][:3])
            print(f"  - {results[k]['label']:<10}: {top3_str}")

    maint_ball = np.mean([results[k]["ball_sum"] for k in CD_MAINTAINED])
    drop_ball = np.mean([results[k]["ball_sum"] for k in CD_DROPPED])
    maint_player = np.mean([results[k]["player_sum"] for k in CD_MAINTAINED])
    drop_player = np.mean([results[k]["player_sum"] for k in CD_DROPPED])
    maint_nodes = np.mean([results[k]["node_count"] for k in CD_MAINTAINED])
    drop_nodes = np.mean([results[k]["node_count"] for k in CD_DROPPED])

    print("\n[두 묶음의 구조·의존도 차이 요약]")
    print(f"1) 객체 중요도 평균 — 유지군(Ball {maint_ball:.4f} / Player {maint_player:.4f}) "
          f"vs 하락군(Ball {drop_ball:.4f} / Player {drop_player:.4f})")
    print(f"2) 평균 트리 크기   — 유지군 {maint_nodes:.1f} nodes vs 하락군 {drop_nodes:.1f} nodes")

    maint_common = set(f for f, _ in results["C0"]["top_features"][:3]) & set(
        f for f, _ in results["D16"]["top_features"][:3]
    )
    drop_common = set(f for f, _ in results["C16"]["top_features"][:3]) & set(
        f for f, _ in results["D32"]["top_features"][:3]
    )
    print(f"3) 유지군 공통 상위 피처(Top-3 교집합): {sorted(maint_common) if maint_common else '없음'}")
    print(f"4) 하락군 공통 상위 피처(Top-3 교집합): {sorted(drop_common) if drop_common else '없음'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=list(AGENTS) + ["both", "cd"], default=None)
    args = parser.parse_args()

    if args.agent is None:
        target = "B" if "Enemy" in os.environ.get("SCOBI_EXCLUDE_OBJECTS", "") else "both"
    else:
        target = args.agent

    if target == "both":
        analyze_agent("A")
        env_b = os.environ.copy()
        env_b["SCOBI_EXCLUDE_OBJECTS"] = "Enemy"
        subprocess.run([sys.executable, __file__, "--agent", "B"], env=env_b, check=True)
    elif target == "cd":
        if "Enemy" not in os.environ.get("SCOBI_EXCLUDE_OBJECTS", ""):
            env_b = os.environ.copy()
            env_b["SCOBI_EXCLUDE_OBJECTS"] = "Enemy"
            subprocess.run([sys.executable, __file__, "--agent", "cd"], env=env_b, check=True)
        else:
            compare_cd_agents()
    elif target != "A":
        if "Enemy" not in os.environ.get("SCOBI_EXCLUDE_OBJECTS", ""):
            env_b = os.environ.copy()
            env_b["SCOBI_EXCLUDE_OBJECTS"] = "Enemy"
            subprocess.run([sys.executable, __file__, "--agent", target], env=env_b, check=True)
        else:
            analyze_agent(target)
    else:
        analyze_agent("A")