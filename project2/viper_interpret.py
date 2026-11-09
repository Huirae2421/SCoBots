"""
viper_interpret.py — VIPER 결정트리가 '무엇을 보고 판단하는지' 분석

[코드 설명]
viper_extract.py 로 추출된 결정트리(.viper, sklearn DecisionTreeClassifier)를 불러오고,
관측 벡터의 각 입력이 어떤 개념인지(예: Enemy1.y, Ball1.y) 이름표를 가져옵니다.
트리가 적(Enemy) 위치로 분기하면 지름길의 '트리 수준 증거'가 됩니다.
(논문 Fig.4: A는 적·자기 패들의 수직위치로 판단, 공은 대체로 무시)
"""
import glob
import numpy as np
from joblib import load
from scobi import Environment

# 분석할 체크포인트 (A = 적 포함). B를 보려면 아래 두 값을 B용으로 바꾸세요.
CKPT = "Pong_seed0_reward-env_oc"
FOCUS = "default_focus_Pong-v5.yaml"

tree_path = glob.glob(
    f"resources/viper_extracts/extract_output/{CKPT}*-extraction/*_best.viper")[0]
clf = load(tree_path)
print("트리:", tree_path.split("/")[-1], "| 노드수:", clf.tree_.node_count)

env = Environment("ALE/Pong-v5", focus_dir=f"resources/checkpoints/{CKPT}",
                  focus_file=FOCUS, silent=True, refresh_yaml=False)
env.reset(seed=0)
feat_names = env.get_vector_entry_descriptions()

# ===== TODO 1 =====
# clf.feature_importances_ 를 이용해, 트리가 실제로 쓰는 피처를
# 중요도 높은 순으로 feat_names 와 함께 출력해 주세요.
# 힌트: order = np.argsort(clf.feature_importances_)[::-1]
#       중요도 0 보다 큰 것만 이름과 함께 출력


# ===== TODO 2 =====
# 위 상위 피처 이름 중 "Enemy" 가 들어간 것이 있는지 확인해,
# 있으면 "지름길 징후: 적 위치에 의존", 없으면 "공/자기 패들 기반" 으로 출력해 주세요.
