"""
plot_curves.py — A와 B의 학습곡선 비교

[코드 설명]
학습 중 저장된 평가 기록(evaluations.npz)을 읽어, 타임스텝에 따른 평가 보상을
A(적 포함)와 B(적 제거)에 대해 그립니다. 세 시드(0/16/32)를 묶어 평균과
표준편차 밴드로 나타내면, 두 에이전트가 표준 Pong에서 비슷하게 잘 학습된다는
점이 보입니다.

[확장] 확장 단계에서 새로 학습한 C·D 의 학습곡선을 추가로 그립니다. (TODO 3)
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_CKPTS = [f"Pong_seed{s}_reward-env_oc" for s in (0, 16, 32)]
B_CKPTS = [f"Pong_seed{s}_reward-env_oc_pruned-external" for s in (0, 16, 32)]


def load_curve(ckpt):
    # 체크포인트의 평가 기록에서 (타임스텝, 에피소드 평균 보상)을 반환
    d = np.load(f"resources/checkpoints/{ckpt}/evaluations.npz")
    return d["timesteps"], d["results"].mean(axis=1)


t = load_curve(A_CKPTS[0])[0]
A_curves = np.stack([load_curve(c)[1] for c in A_CKPTS])
B_curves = np.stack([load_curve(c)[1] for c in B_CKPTS])

plt.figure(figsize=(8, 5))

# ===== TODO 1 =====
A_mean, A_std = A_curves.mean(axis=0), A_curves.std(axis=0)
B_mean, B_std = B_curves.mean(axis=0), B_curves.std(axis=0)

plt.plot(t, A_mean, label="A (with enemy)", color="tab:blue")
plt.fill_between(t, A_mean - A_std, A_mean + A_std, alpha=0.25, color="tab:blue")
plt.plot(t, B_mean, label="B (enemy removed)", color="tab:orange")
plt.fill_between(t, B_mean - B_std, B_mean + B_std, alpha=0.25, color="tab:orange")

print(f"평가 횟수: {len(t)}회 (마지막 {t[-1]:,} 스텝)")
print(f"[마지막 평가] A: {A_mean[-1]:.2f} ± {A_std[-1]:.2f}   "
      f"시드별 {np.round(A_curves[:, -1], 2).tolist()}")
print(f"[마지막 평가] B: {B_mean[-1]:.2f} ± {B_std[-1]:.2f}   "
      f"시드별 {np.round(B_curves[:, -1], 2).tolist()}")
# 마지막 한 번의 평가는 에피소드 8개라 흔들림이 있어서, 마지막 5회 평균도 함께 출력
print(f"[마지막 5회 평균] A: {A_mean[-5:].mean():.2f}   B: {B_mean[-5:].mean():.2f}")
# 처음으로 평균 보상 15점을 넘긴 시점 (학습 속도 비교용)
for name, m in (("A", A_mean), ("B", B_mean)):
    hit = np.where(m >= 15)[0]
    print(f"{name} 평균 보상 15점 첫 도달: "
          + (f"{t[hit[0]]:,} 스텝" if len(hit) else "도달하지 못함"))


# ===== TODO 2 =====
plt.axhline(0, color="gray", lw=0.8, ls="--")
plt.xlabel("Timesteps")
plt.ylabel("Mean evaluation reward (8 episodes)")
plt.title("Pong learning curves: A vs B (mean ± std over seeds 0/16/32)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves.png")
plt.savefig(out, dpi=150)
print("저장:", os.path.relpath(out))


# ================= 확장 단계 =================
C_CKPTS = [f"Pong_seed{s}_reward-env_oc_pruned-external_C-mix" for s in (0, 16, 32)]
D_CKPTS = [f"Pong_seed{s}_reward-env_oc_pruned-external_D-freeze" for s in (0, 16, 32)]

# ===== TODO 3 (확장) =====
"""
C_CKPTS, D_CKPTS 에 확장 단계에서 학습한 에이전트의 체크포인트 이름이 저장되어 있습니다.
  C : 학습 상대를 섞어서 학습한 에이전트
  D : C + 멈추는 적까지 섞어서 학습한 에이전트
  → A·B 와 같은 형태(평가 횟수, 시드 3개)라 load_curve 를 그대로 쓸 수 있습니다.

TODO 1·2 에서 그린 A·B 그림에 이어서 C·D 를 추가해, 네 에이전트를 한 그림에 그려 주세요.
ex)
C_curves = np.stack([load_curve(c)[1] for c in C_CKPTS])
→ D_curves 도 같은 형식으로 만들어서 그려 주세요.
→ 그림은 curves_cd.png 로 따로 저장해 주세요.
"""

C_curves = np.stack([load_curve(c)[1] for c in C_CKPTS])
D_curves = np.stack([load_curve(c)[1] for c in D_CKPTS])
assert all(np.array_equal(load_curve(c)[0], t) for c in C_CKPTS + D_CKPTS), "평가 시점이 A·B 와 다름"

agents = [  # (이름, 곡선, 색)
    ("A (with enemy)", A_curves, "tab:blue"),
    ("B (enemy removed)", B_curves, "tab:orange"),
    ("C (B + mixed opponents)", C_curves, "tab:green"),
    ("D (C + freezing enemy)", D_curves, "tab:red"),
]

# 왼쪽: 전체 구간 / 오른쪽: 10M 이후 확대 (초반 음수 구간 때문에 후반 차이가 눌려 보이지 않도록)
fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5.5),
                                      gridspec_kw={"width_ratios": [1.3, 1]})
late = t >= 10_000_000
for name, curves, color in agents:
    m, sd = curves.mean(axis=0), curves.std(axis=0)
    for ax, sel in ((ax_all, slice(None)), (ax_zoom, late)):
        ax.plot(t[sel], m[sel], label=name, color=color, lw=1.4)
        ax.fill_between(t[sel], (m - sd)[sel], (m + sd)[sel], alpha=0.12, color=color)

ax_all.axhline(0, color="gray", lw=0.8, ls="--")
ax_all.set_title("All timesteps")
ax_zoom.set_title("Zoom: 10M – 20M")
for ax in (ax_all, ax_zoom):
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean evaluation reward on base Pong (8 episodes)")
    ax.grid(alpha=0.3)
ax_all.legend(loc="lower right")
fig.suptitle("Pong learning curves: A / B / C / D (mean ± std over seeds 0/16/32, evaluated on base)")
fig.tight_layout()

# 출력: 마지막 평가, 마지막 5회 평균, 15점 첫 도달, 시드별 후반(15M~20M) 평균
print("\n" + "=" * 60)
print("[확장] A·B·C·D 학습곡선 (학습 중 평가는 모두 base 상대)")
tail = t >= 15_000_000
for name, curves, _ in agents:
    m, sd = curves.mean(axis=0), curves.std(axis=0)
    hit = np.where(m >= 15)[0]
    first = f"{t[hit[0]]:,}" if len(hit) else "도달 못함"
    per_seed = " / ".join(f"{v:.2f}" for v in curves[:, tail].mean(axis=1))
    print(f"{name:24s} 마지막 {m[-1]:5.2f} ± {sd[-1]:4.2f} | 마지막 5회 평균 {m[-5:].mean():5.2f}"
          f" | 15점 첫 도달 {first:>10s} | 15M~20M 시드별(0/16/32) {per_seed}")

out_cd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves_cd.png")
fig.savefig(out_cd, dpi=150)
print("저장:", os.path.relpath(out_cd))
