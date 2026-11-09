"""
correlation.py — 적 패들과 공의 수직위치 상관 분석

[코드 설명]
Pong을 랜덤 에이전트로 10만 프레임 돌리면서, 10프레임마다 적 패들(Enemy)과
공(Ball)의 세로 위치(y)를 모읍니다. 내장 적 AI가 공을 따라다니기 때문에 두 값은
거의 함께 움직이며, 이 상관이 높다는 점이 'A 에이전트가 공 대신 적을 봐도 되는'
지름길(shortcut)의 근본 원인입니다.
(논문 A.7 기준: Pearson 99.6% / Spearman 99.5% / Kendall 96.4%)
"""
from scobi import Environment
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt

env = Environment("ALE/Pong-v5",
    focus_dir="resources/checkpoints/Pong_seed0_reward-env_oc",
    focus_file="default_focus_Pong-v5.yaml", silent=True, refresh_yaml=False)
env.reset(seed=0)

enemy_y, ball_y = [], []
for i in range(100000):
    _, _, term, trunc, _ = env.step(env.action_space.sample())
    objs = {o.category: o for o in env.oc_env.objects if o is not None}
    if i % 10 == 0 and "Enemy" in objs and "Ball" in objs:
        enemy_y.append(objs["Enemy"].y)
        ball_y.append(objs["Ball"].y)
    if term or trunc:
        env.reset()

enemy_y, ball_y = np.array(enemy_y), np.array(ball_y)
print("수집 샘플 수:", len(enemy_y))


# ===== TODO 1 =====
# pearsonr / spearmanr / kendalltau 로 세 상관계수를 계산해 출력해 주세요.
# (논문 A.7 값과 비교: Pearson 0.996 / Spearman 0.995 / Kendall 0.964)


# ===== TODO 2 =====
# plt.scatter(ball_y, enemy_y) 로 산점도를 그려 correlation.png 로 저장해 주세요.
# (거의 직선이 나오면 적이 공을 따라다닌다는 '시각 증거')
