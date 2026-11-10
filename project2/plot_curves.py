"""
plot_curves.py — A와 B의 학습곡선 비교

[코드 설명]
학습 중 저장된 평가 기록(evaluations.npz)을 읽어, 타임스텝에 따른 평가 보상을
A(적 포함)와 B(적 제거)에 대해 그립니다. 세 시드(0/16/32)를 묶어 평균과
표준편차 밴드로 나타내면, 두 에이전트가 표준 Pong에서 비슷하게 잘 학습된다는
점이 보입니다.
"""
import numpy as np
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
"""
위에서 불러온 값들이 저장되어 있습니다.
  t        : 평가한 타임스텝 (x축)
  A_curves : A(적 포함) 시드 3개의 평가 보상, 모양 (3, 평가횟수)
  B_curves : B(적 제거) 시드 3개의 평가 보상, 모양 (3, 평가횟수)

시드 축(axis=0)으로 평균과 표준편차를 구해서
plt.plot 으로 평균 곡선을, plt.fill_between 으로 평균±표준편차 밴드를 그려 주세요.
ex)
A_mean = A_curves.mean(axis=0)
A_std = A_curves.std(axis=0)
plt.plot(t, A_mean, label="A (with enemy)")
plt.fill_between(t, A_mean - A_std, A_mean + A_std, alpha=0.25)
→ B_curves 도 같은 형식으로 그려 주세요.
→ 마지막 평가 시점의 평균 보상(A_mean[-1] 등)도 print 로 출력해 보세요.
"""


# ===== TODO 2 =====
"""
그래프에 축 이름 / 범례 / 제목을 달고 curves.png 로 저장해 주세요.
ex)
plt.xlabel("Timesteps")
→ plt.ylabel, plt.legend, plt.title 도 같은 형식으로 달아 주세요.
plt.savefig("curves.png")
"""
