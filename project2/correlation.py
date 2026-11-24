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

def collect(env, n_frames=100000, every=10):
    """랜덤 행동으로 n_frames 만큼 돌리며 every 프레임마다 위치를 모아 dict 로 반환.
    ball_dx : 바로 전 프레임 대비 공의 x 변화 (+ 면 내 쪽(오른쪽), - 면 적 쪽으로 이동)"""
    keys = ["enemy_y", "ball_y", "ball_x", "enemy_x", "player_x", "ball_dx"]
    d = {k: [] for k in keys}
    prev_ball_x = 0
    for i in range(n_frames):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        objs = {o.category: o for o in env.oc_env.objects if o is not None}
        cur_ball_x = objs["Ball"].x if "Ball" in objs else 0
        if i % every == 0 and "Enemy" in objs and "Ball" in objs and "Player" in objs:
            d["enemy_y"].append(objs["Enemy"].y)
            d["ball_y"].append(objs["Ball"].y)
            d["ball_x"].append(objs["Ball"].x)
            d["enemy_x"].append(objs["Enemy"].x)
            d["player_x"].append(objs["Player"].x)
            d["ball_dx"].append(cur_ball_x - prev_ball_x)
        prev_ball_x = cur_ball_x
        if term or trunc:
            env.reset()
            prev_ball_x = 0
    return {k: np.array(v) for k, v in d.items()}


env = Environment("ALE/Pong-v5",
    focus_dir="resources/checkpoints/Pong_seed0_reward-env_oc",
    focus_file="default_focus_Pong-v5.yaml", silent=True, refresh_yaml=False)
env.reset(seed=0)
env.action_space.seed(0)  # 랜덤 행동도 시드 고정 → 실행할 때마다 같은 결과

base = collect(env)
enemy_y, ball_y = base["enemy_y"], base["ball_y"]
ball_x, enemy_x, player_x = base["ball_x"], base["enemy_x"], base["player_x"]
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
def in_play(d):
    # TODO 1 과 같은 keep 조건: 공·적이 검출되고, 공이 두 패들 사이에 있는 샘플
    no_ball = (d["ball_y"] == 0) & (d["ball_x"] == 0)
    no_enemy = (d["enemy_x"] == 0)
    between = (d["ball_x"] > d["enemy_x"]) & (d["ball_x"] < d["player_x"])
    return ~no_ball & ~no_enemy & between


def corr3(by, ey):
    return pearsonr(by, ey)[0], spearmanr(by, ey)[0], kendalltau(by, ey)[0]


print("\n" + "=" * 60)
print("[확장] LazyEnemy 환경에서 같은 측정 (랜덤 에이전트, 10만 프레임, 시드 0)")
env_lazy = make_lazy_env()
lazy = collect(env_lazy)
print("수집 샘플 수:", len(lazy["enemy_y"]))

results = {}
for name, d in (("base", base), ("Lazy", lazy)):
    k = in_play(d)
    toward_me = k & (d["ball_dx"] > 0)      # 공이 내 쪽으로 이동 → Lazy 적은 멈춤
    toward_enemy = k & (d["ball_dx"] < 0)   # 공이 적 쪽으로 이동 → Lazy 적도 따라감
    results[name] = {
        "keep": k,
        "all": corr3(d["ball_y"][k], d["enemy_y"][k]),
        "me": corr3(d["ball_y"][toward_me], d["enemy_y"][toward_me]),
        "enemy": corr3(d["ball_y"][toward_enemy], d["enemy_y"][toward_enemy]),
        "gap": np.abs(d["enemy_y"] - d["ball_y"])[k].mean(),
        "n": (k.sum(), toward_me.sum(), toward_enemy.sum()),
    }

print("\n[경기 중 샘플의 공 y – 적 y 상관]   (Pearson / Spearman / Kendall)")
print(f"{'':22s}{'base':>24s}{'Lazy':>24s}")
for key, label in (("all", "전체"), ("enemy", "공이 적 쪽으로 이동"), ("me", "공이 내 쪽으로 이동")):
    cells = ["{:.3f} / {:.3f} / {:.3f}".format(*results[n][key]) for n in ("base", "Lazy")]
    print(f"  {label:18s}{cells[0]:>24s}{cells[1]:>24s}")
nb, nl = results["base"]["n"], results["Lazy"]["n"]
print(f"  {'샘플 수 (전체/내쪽/적쪽)':18s}{str(nb):>24s}{str(nl):>24s}")
print(f"  {'평균 |적 y - 공 y|':18s}{results['base']['gap']:>24.2f}{results['Lazy']['gap']:>24.2f}")
print("→ base 는 공 방향과 상관없이 적이 공 높이를 따라가지만, Lazy 는 공이 내 쪽으로 올 때 적이 멈춰"
      "\n  그 구간의 상관이 크게 떨어짐. 즉 '적 위치 ≈ 공 위치'라는 지름길이 Lazy 에서는 성립하지 않음.")

# 산점도: base / Lazy 를 나란히, 공 진행 방향으로 색 구분
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
for ax, (name, d) in zip(axes, (("base", base), ("Lazy", lazy))):
    k = results[name]["keep"]
    for mask, color, lab in ((k & (d["ball_dx"] < 0), "tab:blue", "ball → enemy"),
                             (k & (d["ball_dx"] > 0), "tab:red", "ball → player")):
        ax.scatter(d["ball_y"][mask], d["enemy_y"][mask], s=2, alpha=0.3, color=color, label=lab)
    p, sp, kt = results[name]["all"]
    ax.set_title(f"{name} (ball in play, n={k.sum()})\n"
                 f"Pearson={p:.3f}  Spearman={sp:.3f}  Kendall={kt:.3f}")
    ax.set_xlabel("Ball y")
    ax.set_ylabel("Enemy y")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", markerscale=6)
fig.suptitle("Enemy paddle y vs Ball y: base vs LazyEnemy (random agent, 100k frames)")
fig.tight_layout()
out_lazy = os.path.join(out_dir, "correlation_lazy.png")
fig.savefig(out_lazy, dpi=150)
print("저장:", os.path.relpath(out_lazy))
