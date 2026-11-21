"""
hackatari_eval.py — HackAtari 수정 환경(NoEnemy / LazyEnemy)에서 에이전트 재평가

각 에이전트(A=적 포함 / B=적 제거)를 아래 3개 환경에서 평가한다 (논문 Fig.3과 같은 방식):
  - base      : 표준 Pong
  - NoEnemy   : 적이 에이전트 관측에서 사라지지만 계속 플레이 (hackatari hidden_enemy)
  - LazyEnemy : 적이 공을 받아친 뒤 멈춤 (hackatari lazy_enemy)
scobi의 OCAtari 환경 생성을 HackAtari로 바꿔끼워 수정 게임을 적용한다.

[필요 패키지] requirements.txt 외에 HackAtari가 필요하다.
  pip 릴리스(0.0.6)에는 hidden_enemy가 없어 GitHub의 아래 커밋을 쓴다 (의존성은 requirements로 이미 설치됨):
  pip install --no-deps "git+https://github.com/k4ntz/HackAtari.git@8ca2bdf1cd3a9ef34834d9c77943380bcf3a7945"
[검증 환경] Python 3.8, gymnasium 0.28.1, ale-py 0.10.1, stable-baselines3 2.0.0, ocatari 2.0.0, torch 2.4.1+cpu
  (README의 새 설치 조합인 ale-py 0.8.1 에서는 아직 실행 확인 안 함)

실행: cd ~/scobots && PYTHONPATH=. python project2/hackatari_eval.py [--episodes N] [--quick]
결과: project2/results/hackatari_eval.md
"""
import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# --- 글루: scobi의 환경 생성(ocgym.make)을 HackAtari로 교체 ---
import scobi.environments.ocgym as ocgym
from hackatari.core import HackAtari

_MODIFS = []  # run_eval에서 조건별로 설정


def _patched_make(env_name, *args, notify=False, **kwargs):
    # scobi는 OCAtari(env_name, "ram", hud=..., buffer_window_size=2)를 만든다.
    # HackAtari(OCAtari 상속)로 바꾸되 mode="ram"과 나머지 kwargs를 그대로 전달.
    return HackAtari(env_name, modifs=_MODIFS, mode="ram", **kwargs)


ocgym.make = _patched_make

from scobi import Environment  # noqa: E402  (패치 후 import)

EVAL_SEED = 84
SEEDS = [0, 16, 32]
AGENTS = {
    "A": dict(ckpt="Pong_seed{s}_reward-env_oc", focus="default_focus_Pong-v5.yaml", exclude=False),
    "B": dict(ckpt="Pong_seed{s}_reward-env_oc_pruned-external", focus="pruned_pong.yaml", exclude=True),
}
CONDITIONS = {"base": [], "NoEnemy": ["hidden_enemy"], "LazyEnemy": ["lazy_enemy"]}


def run_eval(agent, seed, cond_modifs, episodes):
    global _MODIFS
    _MODIFS = cond_modifs
    cfg = AGENTS[agent]
    base = "resources/checkpoints/" + cfg["ckpt"].format(s=seed)
    # B(적 제거)만 SCOBI_EXCLUDE_OBJECTS=Enemy 필요
    if cfg["exclude"]:
        os.environ["SCOBI_EXCLUDE_OBJECTS"] = "Enemy"
    else:
        os.environ.pop("SCOBI_EXCLUDE_OBJECTS", None)

    env = Environment("ALE/Pong-v5", focus_dir=base, focus_file=cfg["focus"],
                      draw_features=False, reward=0, silent=True, refresh_yaml=False)
    env.reset(seed=EVAL_SEED)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(base + "/best_vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(base + "/best_model")

    rewards, cur, ep = [], 0.0, 0
    obs = venv.reset()
    while ep < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(action)
        cur += float(r[0])
        if done[0]:
            ep += 1
            rewards.append(cur)
            cur = 0.0
            obs = venv.reset()
    venv.close()
    return np.array(rewards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="A seed0 만 5에피소드로 빠른 검증")
    args = ap.parse_args()

    agents, seeds, conds = ["A", "B"], SEEDS, list(CONDITIONS)
    if args.quick:
        agents, seeds, args.episodes = ["A"], [0], 5

    print("=== HackAtari 개입평가 (episodes=%d, seed=%d) ===" % (args.episodes, EVAL_SEED))
    results = {}
    for agent in agents:
        for seed in seeds:
            for cond in conds:
                rw = run_eval(agent, seed, CONDITIONS[cond], args.episodes)
                results[(agent, seed, cond)] = rw
                print("%s seed%-2d %-9s mean=%6.2f  std=%5.2f" %
                      (agent, seed, cond, rw.mean(), rw.std()), flush=True)

    print("\n=== 시드 평균 요약 (mean±std across seeds) ===")
    for agent in agents:
        for cond in conds:
            ms = [results[(agent, s, cond)].mean() for s in seeds]
            print("%s %-9s: %6.2f ± %5.2f" % (agent, cond, np.mean(ms), np.std(ms)))
    print("HACKATARI_EVAL_DONE")


if __name__ == "__main__":
    main()
