"""
correlation.py — 적 패들과 공의 수직위치 상관 분석

[코드 설명]
Pong을 랜덤 에이전트로 10만 프레임 돌리면서, 10프레임마다 적 패들(Enemy)과
공(Ball)의 세로 위치(y)를 모읍니다. 내장 적 AI가 공을 따라다니기 때문에 두 값은
거의 함께 움직이며, 이 상관이 높다는 점이 'A 에이전트가 공 대신 적을 봐도 되는'
지름길(shortcut)의 근본 원인입니다.
(논문 A.7 기준: Pearson 99.6% / Spearman 99.5% / Kendall 96.4%)

[확장] 같은 측정을 LazyEnemy 환경에서도 해서 base 와 비교합니다. (TODO 3)
"""
from scobi import Environment
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

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
# 1) 제외 전 전체 샘플의 상관 (참고용)
r_all = pearsonr(ball_y, enemy_y)[0]
print(f"[제외 전] Pearson : {r_all:.3f}")

# 2) 공이 '경기 중'이 아닌 샘플 제외
#  - 공이 화면에 없음 : 득점 직후 등 공이 사라지면 OCAtari가 (0, 0)을 돌려줌
#  - 적 패들 미검출   : 적 x 가 0 으로 들어오는 프레임
#  - 공이 두 패들 사이가 아님 : 득점 직후 공이 패들 뒤(골라인 쪽)로 빠져 있는 구간.
#    랜덤 에이전트는 실점이 잦아 공이 내 패들 뒤(x > player_x)에 머무는 샘플이 많고,
#    이때 적 AI는 공을 따라가지 않으므로 산점도에서 직선을 벗어난 점이 됨
no_ball = (ball_y == 0) & (ball_x == 0)
no_enemy = (enemy_x == 0)
between = (ball_x > enemy_x) & (ball_x < player_x)
keep = ~no_ball & ~no_enemy & between

print("\n[제외한 샘플]")
print(f"  공 미검출 (ball x,y = 0)          : {no_ball.sum()}")
print(f"  적 패들 미검출 (enemy_x = 0)      : {(no_enemy & ~no_ball).sum()}")
print(f"  공이 두 패들 사이가 아님          : {(~between & ~no_ball & ~no_enemy).sum()}")
print(f"  → 합계 {(~keep).sum()} 개 제외, {keep.sum()} 개 사용 (전체 {len(keep)})")

# 3) 남은 샘플로 세 상관계수 계산
r_p, _ = pearsonr(ball_y[keep], enemy_y[keep])
r_s, _ = spearmanr(ball_y[keep], enemy_y[keep])
r_k, _ = kendalltau(ball_y[keep], enemy_y[keep])
print("\n[제외 후 상관계수]")
print(f"  Pearson : {r_p:.3f}  (논문 0.996)")
print(f"  Spearman: {r_s:.3f}  (논문 0.995)")
print(f"  Kendall : {r_k:.3f}  (논문 0.964)")

# 남은 이탈점: 적 AI는 공을 약간 늦게 따라가므로, 공이 위/아래 벽에 튕겨 방향이
# 갑자기 바뀌는 화면 가장자리에서 공-적 y 차이가 커짐 (경기 중 상황이므로 제외하지 않음)
gap = np.abs(enemy_y - ball_y)[keep]
by = ball_y[keep]
print("\n참고) 남은 샘플의 공 높이별 평균 |적 y - 공 y|")
for lo, hi, name in ((0, 60, "위쪽 벽 근처"), (60, 175, "가운데"), (175, 256, "아래쪽 벽 근처")):
    m = (by >= lo) & (by < hi)
    print(f"  {name:8s} (공 y {lo}~{hi}): {gap[m].mean():.2f}  (n={m.sum()})")


# ===== TODO 2 =====
out_dir = os.path.dirname(os.path.abspath(__file__))
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
axes[0].scatter(ball_y, enemy_y, s=2, alpha=0.3)
axes[0].set_title(f"All samples (n={len(ball_y)})\nPearson={r_all:.3f}")
axes[1].scatter(ball_y[keep], enemy_y[keep], s=2, alpha=0.3, color="tab:orange")
axes[1].set_title(f"Ball in play (n={keep.sum()})\n"
                  f"Pearson={r_p:.3f}  Spearman={r_s:.3f}  Kendall={r_k:.3f}")
for ax in axes:
    ax.set_xlabel("Ball y")
    ax.set_ylabel("Enemy y")
    ax.grid(alpha=0.3)
fig.suptitle("Pong: Enemy paddle y vs Ball y (random agent, 100k frames)")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "correlation.png"), dpi=150)
print("저장:", os.path.relpath(os.path.join(out_dir, "correlation.png")))


# ================= 확장 단계 =================
def make_lazy_env():
    # scobi 가 게임을 만들 때 HackAtari(lazy_enemy) 를 쓰도록 바꿔 끼움
    import scobi.environments.ocgym as ocgym
    from hackatari.core import HackAtari
    ocgym.make = lambda name, *a, notify=False, **kw: HackAtari(name, modifs=["lazy_enemy"], mode="ram", **kw)
    env = Environment("ALE/Pong-v5",
        focus_dir="resources/checkpoints/Pong_seed0_reward-env_oc",
        focus_file="default_focus_Pong-v5.yaml", silent=True, refresh_yaml=False)
    env.reset(seed=0)
    env.action_space.seed(0)
    return env


# ===== TODO 3 (확장: LazyEnemy) =====
"""
make_lazy_env() 는 LazyEnemy 환경을 만들어 주는 함수입니다.
  LazyEnemy : 공이 내 쪽으로 올 때 적 패들이 멈추는 변형 (HackAtari lazy_enemy)
  → 사용법은 위의 env 와 같습니다. (env.step, env.oc_env.objects 등)

1) 위의 수집 루프를 LazyEnemy 환경에서도 돌려서 enemy_y, ball_y 등을 모아 주세요.
   ex)
   env_lazy = make_lazy_env()
   → 위 for 문을 함수로 묶어 두 환경에서 같이 쓰면 편합니다.
2) TODO 1 과 같은 keep 조건으로 상관계수를 구해, base 결과와 나란히 출력해 주세요.
3) 산점도를 correlation_lazy.png 로 저장해 주세요.
"""
