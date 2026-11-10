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

"""
[실행 전 준비] 저장소 루트에서 결정트리를 먼저 추출해야 합니다.
  PYTHONPATH=. python viper_extract.py -i resources/checkpoints/Pong_seed0_reward-env_oc -r viper -e 2
  - B 는 트리를 뽑을 때와 이 스크립트를 실행할 때 모두 앞에 SCOBI_EXCLUDE_OBJECTS=Enemy 를 붙여 주세요.
"""
tree_path = glob.glob(
    f"resources/viper_extracts/extract_output/{CKPT}*-extraction/*_best.viper")[0]
clf = load(tree_path)
print("트리:", tree_path.split("/")[-1], "| 노드수:", clf.tree_.node_count)

env = Environment("ALE/Pong-v5", focus_dir=f"resources/checkpoints/{CKPT}",
                  focus_file=FOCUS, silent=True, refresh_yaml=False)
env.reset(seed=0)
feat_names = env.get_vector_entry_descriptions()

# ===== TODO 1 =====
"""
위에서 불러온 값들이 저장되어 있습니다.
  clf                      : 결정트리 (sklearn DecisionTreeClassifier)
  clf.feature_importances_ : 각 입력 피처의 중요도 배열 (합이 1)
  feat_names               : 각 입력 피처의 이름 리스트 (예: "Ball1.y", "LT(Enemy1, Player1).y")
  → 같은 번호 i 끼리 짝입니다. feat_names[i] 의 중요도가 clf.feature_importances_[i] 입니다.

중요도가 높은 순서로 번호를 정렬하고, 중요도가 0 보다 큰 피처만 이름과 함께 출력해 주세요.
ex)
imp = clf.feature_importances_
order = np.argsort(imp)[::-1]        # 중요도 큰 순서의 번호
i = order[0]
print(feat_names[i], f"{imp[i]:.3f}")  # 1등 피처 하나 출력
→ for 문으로 order 를 돌면서, imp[i] > 0 인 것만 같은 형식으로 출력해 주세요.
"""


# ===== TODO 2 =====
"""
트리가 적(Enemy) 정보에 얼마나 의존하는지 판단해 주세요.
1) 상위 피처 이름 중 "Enemy" 가 들어간 것이 있는지 확인해 주세요.
   ex)
   if "Enemy" in str(feat_names[i]):
       ...
2) 이름 포함 여부만 보지 말고, "Enemy" 가 들어간 피처들의 중요도를 모두 더한 값도 구해서 같이 출력해 주세요.
   ex)
   enemy_sum = 0.0
   → "Ball" 이 들어간 피처들의 중요도 합(ball_sum)도 구해서 enemy_sum 과 비교해 보세요.
3) 위 결과를 근거로 "지름길 징후: 적 위치에 의존" 또는 "공/자기 패들 기반" 을 출력해 주세요.
"""
