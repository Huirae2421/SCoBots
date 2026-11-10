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
env.action_space.seed(0)  # 랜덤 행동도 시드 고정 → 실행할 때마다 같은 결과

enemy_y, ball_y = [], []
ball_x, enemy_x, player_x = [], [], []
for i in range(100000):
    _, _, term, trunc, _ = env.step(env.action_space.sample())
    objs = {o.category: o for o in env.oc_env.objects if o is not None}
    if i % 10 == 0 and "Enemy" in objs and "Ball" in objs and "Player" in objs:
        enemy_y.append(objs["Enemy"].y)
        ball_y.append(objs["Ball"].y)
        ball_x.append(objs["Ball"].x)
        enemy_x.append(objs["Enemy"].x)
        player_x.append(objs["Player"].x)
    if term or trunc:
        env.reset()

enemy_y, ball_y = np.array(enemy_y), np.array(ball_y)
ball_x, enemy_x, player_x = np.array(ball_x), np.array(enemy_x), np.array(player_x)
print("수집 샘플 수:", len(enemy_y))

# ===== TODO 1 =====
"""
위에서 모은 값들이 numpy 배열로 저장되어 있습니다.
  enemy_y, ball_y   : 적 패들 / 공의 세로 위치 (y)
  ball_x            : 공의 가로 위치 (x)
  enemy_x, player_x : 적 패들 / 내 패들의 가로 위치 (x)

1) 산점도를 한 번 그려 보고, 직선에서 벗어난 점들이 어떤 상황에서 생기는지 살펴보세요.
2) 공이 '경기 중'이 아닌 샘플을 빼는 조건(True/False 배열)을 만들어 주세요.
   ex)
   keep = (ball_y != 0)
   print("제외한 샘플 수:", (~keep).sum())
   → 공이 두 패들 사이(enemy_x ~ player_x)에 있는지도 같은 방식으로 조건에 추가해 보세요.
3) 남은 샘플(keep 이 True 인 것)로 세 상관계수를 구해 논문 값과 비교해 주세요.
   ex)
   r, _ = pearsonr(ball_y[keep], enemy_y[keep])
   print(f"Pearson : {r:.3f}  (논문 0.996)")
   → spearmanr(논문 0.995), kendalltau(논문 0.964) 도 같은 형식으로 구해 주세요.
4) 무엇을, 왜 뺐는지 print 로 함께 남겨 주세요.
"""


# ===== TODO 2 =====
"""
ball_y 를 x축, enemy_y 를 y축으로 산점도를 그려 correlation.png 로 저장해 주세요.
ex)
plt.figure(figsize=(6, 6))
plt.scatter(ball_y, enemy_y, s=2, alpha=0.3)
plt.xlabel("Ball y")
plt.ylabel("Enemy y")
plt.savefig("correlation.png")
→ keep 을 만든 뒤에는 ball_y[keep], enemy_y[keep] 으로도 그려서
  제외 전/후를 비교해 보세요. (plt.title 로 상관계수도 같이 표시해 주세요)
"""
