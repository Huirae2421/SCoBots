"""
hackatari_eval_oldlazy.py — 수정 전(pre-fix) LazyEnemy로 A/B 재평가

왜: HackAtari 커밋 4dec77c("repair lazy enemy", 2024-11-18)가 lazy_enemy를 고쳤다.
    우리 hackatari_eval.py(@8ca2bdf)는 고친 뒤 버전(적이 실제로 멈춤)이고,
    논문(NeurIPS 2024) Fig.3는 고치기 전 버전으로 평가했을 가능성이 높다.
    → 같은 모델을 '수정 전 lazy_enemy'로 평가해서 논문과의 차이가 버전 때문인지 확인한다.

수정 전 함수(@688e27c, hackatari/games/pong.py)를 그대로 옮겨, 현재 HackAtari 코어의
step_modifs에 1번만 등록한다(수정 전엔 step_modifs에만 등록, 현재 버전은 step+post_detection 2곳).
설치된 패키지는 건드리지 않는다.

  수정 전:  if 공x 증가: 적y ← LAST ;  LAST ← (덮어쓰기 전) 적y   → 적이 한 박자 늦게 계속 따라감(추정)
  수정 후:  공x 증가 동안 LAST 갱신 안 함                           → 적이 멈춤

※ 남는 차이: 코어(프레임 루프 안에서 step_modifs 호출 횟수·frameskip 처리)는 현재 버전 그대로다.

실행 (저장소 루트):
  PYTHONPATH=. python project2/hackatari_eval_oldlazy.py --mechanism          # 적 움직임 확인 (모델 불필요, 수십 초)
  PYTHONPATH=. python project2/hackatari_eval_oldlazy.py --quick              # A seed0 5판
  PYTHONPATH=. python project2/hackatari_eval_oldlazy.py --episodes 20        # A/B × 3시드, LazyEnemyOld
  PYTHONPATH=. python project2/hackatari_eval_oldlazy.py --episodes 20 --conds base LazyEnemy LazyEnemyOld
결과: project2/results/hackatari_eval_oldlazy.md
"""
import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import scobi.environments.ocgym as ocgym
from hackatari.core import HackAtari

EVAL_SEED = 84
SEEDS = [0, 16, 32]
AGENTS = {
    "A": dict(ckpt="Pong_seed{s}_reward-env_oc", focus="default_focus_Pong-v5.yaml", exclude=False),
    "B": dict(ckpt="Pong_seed{s}_reward-env_oc_pruned-external", focus="pruned_pong.yaml", exclude=True),
}
# 조건 → (HackAtari modifs, 수정 전 lazy 추가 여부)
CONDITIONS = {
    "base": ([], False),
    "LazyEnemy": (["lazy_enemy"], False),      # 현재(수정 후) 구현, @8ca2bdf
    "LazyEnemyOld": ([], True),                # 수정 전 구현(@688e27c) 이식
}


def make_old_lazy(env):
    """HackAtari @688e27c hackatari/games/pong.py 의 lazy_enemy 를 그대로 옮김 (전역변수 → 클로저)."""
    st = {"last_enemy_y": 127, "ball_prev_x": 130}

    def lazy_enemy_old():
        ram = env.get_ram()          # ALE getRAM() = 복사본 (set_ram 이후에도 값이 안 바뀜)
        if 0 < ram[11] < 5:
            env.set_ram(21, 127)
            env.set_ram(49, 130)
        if st["ball_prev_x"] < ram[49]:
            env.set_ram(21, st["last_enemy_y"])
        st["ball_prev_x"] = ram[49]
        st["last_enemy_y"] = ram[21]  # 덮어쓰기 전 값으로 갱신 ← 수정 전 버전의 핵심
    return lazy_enemy_old


_COND = ("base", [], False)


def make_hackatari(env_name, modifs, old_lazy, **kwargs):
    env = HackAtari(env_name, modifs=modifs, mode="ram", **kwargs)
    if old_lazy:
        env.step_modifs.append(make_old_lazy(env))
    return env


def _patched_make(env_name, *args, notify=False, **kwargs):
    _, modifs, old_lazy = _COND
    return make_hackatari(env_name, modifs, old_lazy, **kwargs)


ocgym.make = _patched_make
from scobi import Environment  # noqa: E402  (패치 후 import)


def run_eval(agent, seed, cond, episodes):
    global _COND
    modifs, old_lazy = CONDITIONS[cond]
    _COND = (cond, modifs, old_lazy)
    cfg = AGENTS[agent]
    base = "resources/checkpoints/" + cfg["ckpt"].format(s=seed)
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


def mechanism(steps=6000):
    """랜덤 행동으로 조건별 적 움직임 측정: 공x 증가/감소 구간에서 적y가 바뀐 스텝 비율."""
    print("=== 메커니즘 확인 (랜덤 행동 %d스텝, seed %d) ===" % (steps, EVAL_SEED))
    # getRAM 복사 여부 확인
    e = make_hackatari("ALE/Pong-v5", [], False)
    e.reset(seed=EVAL_SEED)
    r0 = e.get_ram(); v = int(r0[21]); e.set_ram(21, (v + 7) % 200)
    print("getRAM 복사본인가:", int(r0[21]) == v and int(e.get_ram()[21]) == (v + 7) % 200)
    e.close()
    rng = np.random.RandomState(0)
    for cond, (modifs, old_lazy) in CONDITIONS.items():
        env = make_hackatari("ALE/Pong-v5", modifs, old_lazy)
        env.reset(seed=EVAL_SEED)
        prev = env.get_ram().copy()
        moved = {+1: [], -1: []}
        for _ in range(steps):
            _, _, term, trunc, _ = env.step(rng.randint(env.action_space.n))
            ram = env.get_ram().copy()
            dx = int(ram[49]) - int(prev[49])
            if 0 < abs(dx) < 20 and ram[54] != 0:          # 경기 중, 공이 실제로 움직인 스텝만
                moved[int(np.sign(dx))].append(int(ram[21]) != int(prev[21]))
            prev = ram
            if term or trunc:
                env.reset(seed=EVAL_SEED); prev = env.get_ram().copy()
        env.close()
        print("%-12s 공x 증가 구간: 적y 바뀐 비율 %.2f (n=%d) | 공x 감소 구간: %.2f (n=%d)" % (
            cond, np.mean(moved[+1]) if moved[+1] else float("nan"), len(moved[+1]),
            np.mean(moved[-1]) if moved[-1] else float("nan"), len(moved[-1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--conds", nargs="+", default=["LazyEnemyOld"], choices=list(CONDITIONS))
    ap.add_argument("--agents", nargs="+", default=["A", "B"], choices=list(AGENTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--quick", action="store_true", help="A seed0 5판")
    ap.add_argument("--mechanism", action="store_true", help="적 움직임만 확인하고 종료")
    ap.add_argument("--out", default="project2/results/hackatari_eval_oldlazy.md")
    args = ap.parse_args()

    if args.mechanism:
        mechanism()
        return
    if args.quick:
        args.agents, args.seeds, args.episodes = ["A"], [0], 5

    print("=== 수정 전 LazyEnemy 평가 (episodes=%d, seed=%d, conds=%s) ===" % (args.episodes, EVAL_SEED, args.conds))
    results = {}
    for agent in args.agents:
        for seed in args.seeds:
            for cond in args.conds:
                rw = run_eval(agent, seed, cond, args.episodes)
                results[(agent, seed, cond)] = rw
                print("%s seed%-2d %-12s mean=%6.2f  std=%5.2f" % (agent, seed, cond, rw.mean(), rw.std()), flush=True)

    hn = lambda x: 100 * (x + 20.7) / (14.6 + 20.7)
    lines = ["# 수정 전 LazyEnemy 평가 결과", "",
             "- 스크립트: `project2/hackatari_eval_oldlazy.py --episodes %d --conds %s`" % (args.episodes, " ".join(args.conds)),
             "- LazyEnemyOld = HackAtari @688e27c 의 lazy_enemy 를 현재 코어(@8ca2bdf)의 step_modifs에 1회 등록",
             "- LazyEnemy = 현재 구현(@8ca2bdf, 4dec77c 수정 반영)",
             "- 평가: 결정적 행동, 평가 시드 %d, 표준편차 ddof=0, HN = 100×(x+20.7)/35.3" % EVAL_SEED, "",
             "| 에이전트 | 시드 | 조건 | 평균 ± 표준편차 | HN(%) |", "|---|---|---|---|---|"]
    for (a, s, c), rw in results.items():
        lines.append("| %s | %d | %s | %.2f ± %.2f | %.0f |" % (a, s, c, rw.mean(), rw.std(), hn(rw.mean())))
    lines += ["", "## 시드 평균", "", "| 에이전트 | 조건 | 평균 ± 표준편차 | HN(%) |", "|---|---|---|---|"]
    print("\n=== 시드 평균 요약 (mean±std across seeds) ===")
    for a in args.agents:
        for c in args.conds:
            ms = [results[(a, s, c)].mean() for s in args.seeds]
            print("%s %-12s: %6.2f ± %5.2f  (HN %.0f%%)" % (a, c, np.mean(ms), np.std(ms), hn(np.mean(ms))))
            lines.append("| %s | %s | %.2f ± %.2f | %.0f |" % (a, c, np.mean(ms), np.std(ms), hn(np.mean(ms))))
    if not args.quick:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("저장:", args.out)
    print("HACKATARI_OLDLAZY_DONE")


if __name__ == "__main__":
    main()
