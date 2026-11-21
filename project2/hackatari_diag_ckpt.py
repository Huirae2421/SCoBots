"""
hackatari_diag_ckpt.py — B(또는 A)의 체크포인트별 · 적 조건별 진단 평가 (확장 E1/E2)

E1: 학습 중간 체크포인트(1M~20M, best)를 base / LazyEnemy 에서 평가 → 언제부터 상대 변화에 약해지나
E2: 최종(best) 모델을 여러 적 조건에서 평가 → 무너지는 원인이 '적 행동 변화' 일반인가, lazy 구현 특유인가

조건
  base         : 수정 없음
  Lazy         : HackAtari lazy_enemy (현재 @8ca2bdf, 적이 실제로 멈춤)
  LazyNoBallX  : Lazy 와 같지만 서브 직후 공 x(ram[49]=130) 덮어쓰기를 뺀 것 (복사본, 설치 패키지 수정 없음)
  LazyOld      : 수정 전 lazy_enemy (@688e27c) — hackatari_eval_oldlazy.py 와 같은 이식
  Perturb      : HackAtari random_perturbation_enemy (적 y에 매 호출 ±5 잡음)

RAM (OCAtari ram/pong.py): 공 x=49, 공 y=54, 적 y=50, 내 패들 y=51. 공 x 증가 = 내 쪽(오른쪽)으로 이동.
기록 (에이전트 스텝 = 4프레임 단위)
  - 판별 점수 · 길이
  - 받아치기: 공 진행 방향이 +→− 로 바뀌면 내가 받아침, −→+ 면 적이 받아침. 실점/득점은 보상 −1/+1.
  - 내 쪽으로 오는 공 '구간'마다 [|dx| 최빈값, |dy| 최빈값, 끝날 때 공 y, 끝날 때 내 패들 y, 결과(1=받아침, 0=실점)]

실행 (저장소 루트, 시드별로 따로 띄워 병렬 가능):
  PYTHONPATH=. python project2/hackatari_diag_ckpt.py --seed 16 --ckpts 1 2 3 4 5 6 8 10 15 20 best --conds base Lazy --episodes 10 --tag e1
  PYTHONPATH=. python project2/hackatari_diag_ckpt.py --seed 16 --ckpts best --conds base Lazy LazyNoBallX LazyOld Perturb --episodes 20 --tag e2
결과: project2/results/diag_ckpt/<tag>_<agent>_seed<s>.jsonl (한 줄 = 체크포인트×조건)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import time
import argparse
from collections import Counter
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import scobi.environments.ocgym as ocgym
from hackatari.core import HackAtari

torch.set_num_threads(1)
EVAL_SEED = 84
AGENTS = {
    "A": dict(ckpt="Pong_seed{s}_reward-env_oc", focus="default_focus_Pong-v5.yaml", exclude=False),
    "B": dict(ckpt="Pong_seed{s}_reward-env_oc_pruned-external", focus="pruned_pong.yaml", exclude=True),
    # C = B + 학습 상대 섞기 (project2/train_c.py). 입력은 B와 같은 18차원.
    "C": dict(ckpt="Pong_seed{s}_reward-env_oc_pruned-external_C-mix", focus="pruned_pong.yaml", exclude=True),
    # D = C + '무작위로 멈추는 적' (project2/train_d.py)
    "D": dict(ckpt="Pong_seed{s}_reward-env_oc_pruned-external_D-freeze", focus="pruned_pong.yaml", exclude=True),
    # D2 = D와 같은 상대, 비율만 base 1/2 · Freeze 1/4 · Perturb 1/8 · LazyOld 1/8 (project2/train_d2.py)
    "D2": dict(ckpt="Pong_seed{s}_reward-env_oc_pruned-external_D2-freeze", focus="pruned_pong.yaml", exclude=True),
}


# ---------- 적 조건 ----------
def lazy_old(env):
    """수정 전 lazy_enemy (@688e27c): step_modifs 에만 1회 등록."""
    st = {"last": 127, "prev_x": 130}

    def f():
        ram = env.get_ram()
        if 0 < ram[11] < 5:
            env.set_ram(21, 127)
            env.set_ram(49, 130)
        if st["prev_x"] < ram[49]:
            env.set_ram(21, st["last"])
        st["prev_x"] = ram[49]
        st["last"] = ram[21]
    return f


def lazy_no_ballx(env):
    """현재 lazy_enemy (@8ca2bdf)에서 공 x 덮어쓰기(ram[49]=130)만 뺀 것. 현재처럼 step+post_detection 2곳 등록."""
    st = {"last": 127, "prev_x": 130}

    def f():
        ram = env.get_ram()
        if 0 < ram[11] < 5:
            env.set_ram(21, 127)
        if st["prev_x"] < ram[49]:
            env.set_ram(21, st["last"])
            tmp = st["last"]
        else:
            tmp = ram[21]
        st["prev_x"] = ram[49]
        st["last"] = tmp
    return f


def lazy_snap(how):
    """개입 실험(E6): 현재 Lazy(@8ca2bdf) 복사본 + '내가 받아친 순간'에만 적 위치를 바꿔 놓음.
    how="ball": 적을 그 순간 공 높이로 (base처럼 적y≈공y 관계 복원)
    how="rand": 적을 무작위 높이로 (대조군: 적을 옮기기만 해서 생기는 효과)
    RAM[21](적 제어) == RAM[50](적 y) 이고 공 y(RAM[54])와 같은 좌표(44~206) — 2026-11-20 측정."""
    def make(env):
        st = {"last": 127, "prev_x": 130, "dir": 0}
        rng = np.random.RandomState(12345)
        env._snaps = 0

        def f():
            ram = env.get_ram()
            if 0 < ram[11] < 5:
                env.set_ram(21, 127)
                env.set_ram(49, 130)
                ram = env.get_ram()
            x = int(ram[49])
            hit = st["dir"] == 1 and 0 < st["prev_x"] - x < 20 and ram[54] != 0
            if x != st["prev_x"]:
                st["dir"] = 1 if x > st["prev_x"] else -1
            if st["prev_x"] < x:
                env.set_ram(21, st["last"])
                tmp = st["last"]
            else:
                tmp = int(ram[21])
            if hit:
                tgt = int(min(max(int(ram[54]), 44), 206)) if how == "ball" else int(rng.randint(44, 207))
                env.set_ram(21, tgt)
                tmp = tgt
                env._snaps += 1
            st["prev_x"] = x
            st["last"] = tmp
        return f
    return make


def freeze_enemy(env):
    """D·D2 학습 상대 Freeze 와 같은 규칙 (project2/train_d.py): 매 스텝 확률 1/30로 멈춤 시작,
    10~60 스텝 동안 적을 그 자리에 고정. step(프레임마다 고정) + post(스텝당 1회 시간 관리) 둘 다 등록.
    난수는 고정 시드(777) → 평가 재현 가능. 모델 선택(E10)에만 쓰고 판정에는 안 씀."""
    st = {"left": 0, "y": 127}
    rng = np.random.RandomState(777)

    def hold():
        if st["left"] > 0:
            env.set_ram(21, st["y"])

    def tick():
        if st["left"] > 0:
            st["left"] -= 1
            env.set_ram(21, st["y"])
        elif rng.rand() < 1 / 30:
            st["left"] = rng.randint(10, 61)
            st["y"] = int(env.get_ram()[21])
    return hold, tick


# 조건 → (HackAtari modifs, 추가 함수 생성기, 등록 위치)
CONDITIONS = {
    "Freeze": ([], freeze_enemy, ("step2", "post2")),
    "LazySnapBall": ([], lazy_snap("ball"), ("step", "post")),
    "LazySnapRand": ([], lazy_snap("rand"), ("step", "post")),
    "base": ([], None, ()),
    "Lazy": (["lazy_enemy"], None, ()),
    "LazyNoBallX": ([], lazy_no_ballx, ("step", "post")),
    "LazyOld": ([], lazy_old, ("step",)),
    "Perturb": (["random_perturbation_enemy"], None, ()),
}

_COND = "base"
_ENVS = []  # 생성된 HackAtari (RAM 읽기용)


def _patched_make(env_name, *args, notify=False, **kwargs):
    modifs, extra, where = CONDITIONS[_COND]
    env = HackAtari(env_name, modifs=modifs, mode="ram", **kwargs)
    if extra is not None:
        fn = extra(env)
        if "step2" in where:              # (step용, post용) 두 함수를 따로 등록하는 조건
            env.step_modifs.append(fn[0])
            env.post_detection_modifs.append(fn[1])
        else:
            if "step" in where:
                env.step_modifs.append(fn)
            if "post" in where:
                env.post_detection_modifs.append(fn)
    _ENVS.append(env)
    return env


ocgym.make = _patched_make
from scobi import Environment  # noqa: E402


def model_paths(base, ck):
    if ck == "best":
        return base + "/best_model", base + "/best_vecnormalize.pkl"
    n = int(float(ck) * 1_000_000)
    d = base + "/training_checkpoints"
    return d + f"/model_{n}_steps", d + f"/model_vecnormalize_{n}_steps.pkl"


def run(agent, seed, ck, cond, episodes):
    global _COND
    _COND = cond
    cfg = AGENTS[agent]
    base = "resources/checkpoints/" + cfg["ckpt"].format(s=seed)
    if cfg["exclude"]:
        os.environ["SCOBI_EXCLUDE_OBJECTS"] = "Enemy"
    else:
        os.environ.pop("SCOBI_EXCLUDE_OBJECTS", None)
    mpath, vpath = model_paths(base, ck)

    _ENVS.clear()
    env = Environment("ALE/Pong-v5", focus_dir=base, focus_file=cfg["focus"],
                      draw_features=False, reward=0, silent=True, refresh_yaml=False)
    hack = _ENVS[-1]
    env.reset(seed=EVAL_SEED)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(vpath, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(mpath)

    scores, lengths = [], []
    cnt = Counter()          # my_ret, my_miss, en_ret, en_miss
    dxh = Counter()          # |dx| 분포 (경기 중, 유효 스텝)
    segs = []                # 내 쪽으로 오는 공 구간
    esegs = []               # 적 쪽으로 가는 공 구간: [시작 적y, 끝 공y, 끝 적y, 시작 공y, 결과(1=적이 받아침, 0=적 실점)]
    track = []               # 공이 내 쪽으로 올 때 (공y, 적y), 5스텝마다
    cur, steps, ep = 0.0, 0, 0
    obs = venv.reset()
    prev = hack.get_ram().copy()
    direction, seg, eseg = 0, None, None
    while ep < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = venv.step(action)
        r = float(r[0])
        cur += r
        steps += 1
        if done[0]:
            # 득점/실점 처리 후 판 종료 (RAM은 이미 reset 이후라 쓰지 않음)
            if r < 0:
                cnt["my_miss"] += 1
                if seg is not None:
                    segs.append(seg + [0])
            elif r > 0:
                cnt["en_miss"] += 1
                if eseg is not None:
                    esegs.append(eseg + [0])
            scores.append(cur)
            lengths.append(steps)
            cur, steps, ep = 0.0, 0, ep + 1
            obs = venv.reset()
            prev = hack.get_ram().copy()
            direction, seg, eseg = 0, None, None
            continue
        ram = hack.get_ram().copy()
        if r < 0:
            cnt["my_miss"] += 1
            if seg is not None:
                segs.append(seg + [0])
            direction, seg, eseg = 0, None, None
        elif r > 0:
            cnt["en_miss"] += 1
            if eseg is not None:
                esegs.append(eseg + [0])
            direction, seg, eseg = 0, None, None
        inplay = ram[54] != 0 and ram[49] > 49 and prev[54] != 0 and prev[49] > 49
        dx = int(ram[49]) - int(prev[49])
        dy = int(ram[54]) - int(prev[54])
        if inplay and 0 < abs(dx) < 20 and r == 0:
            dxh[abs(dx)] += 1
            d = 1 if dx > 0 else -1
            if direction == 1 and d == -1:        # 내가 받아침
                cnt["my_ret"] += 1
                if seg is not None:
                    segs.append(seg + [1])
                seg = None
            elif direction == -1 and d == 1:      # 적이 받아침
                cnt["en_ret"] += 1
                if eseg is not None:
                    esegs.append(eseg + [1])
                eseg = None
            if d == -1:                           # 적 쪽으로 가는 중: [시작 적y, 끝 공y, 끝 적y, 시작 공y]
                if eseg is None:
                    eseg = [int(prev[50]), 0, 0, int(prev[54])]
                eseg[1], eseg[2] = int(ram[54]), int(ram[50])
            elif steps % 5 == 0:                  # 공이 내 쪽으로 오는 동안 (공y, 적y) 표본
                track.append([int(ram[54]), int(ram[50])])
            if d == 1:                            # 내 쪽으로 오는 중: 구간 갱신
                if seg is None:
                    seg = [Counter(), Counter(), 0, 0]
                seg[0][abs(dx)] += 1
                seg[1][abs(dy)] += 1
                seg[2], seg[3] = int(ram[54]), int(ram[51])
            direction = d
        prev = ram
    venv.close()
    # 구간 요약: [|dx| 최빈, |dy| 최빈, 끝 공y, 끝 패들y, 결과]
    segs_out = [[s[0].most_common(1)[0][0], s[1].most_common(1)[0][0], s[2], s[3], s[4]]
                for s in segs if isinstance(s[0], Counter) and s[0]]
    return dict(agent=agent, seed=seed, ckpt=str(ck), cond=cond, episodes=episodes, eval_seed=EVAL_SEED,
                snaps=getattr(hack, "_snaps", None),
                scores=scores, lengths=lengths, counts=dict(cnt),
                dx_hist={int(k): v for k, v in dxh.items()}, segs=segs_out, esegs=esegs, track=track)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="B", choices=list(AGENTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpts", nargs="+", default=["best"])
    ap.add_argument("--conds", nargs="+", default=["base", "Lazy"], choices=list(CONDITIONS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--eval-seed", type=int, default=84)
    args = ap.parse_args()
    global EVAL_SEED
    EVAL_SEED = args.eval_seed

    outdir = "project2/results/diag_ckpt"
    os.makedirs(outdir, exist_ok=True)
    out = f"{outdir}/{args.tag}_{args.agent}_seed{args.seed}.jsonl"
    print(f"=== diag {args.agent} seed{args.seed} ckpts={args.ckpts} conds={args.conds} ep={args.episodes} → {out}", flush=True)
    for ck in args.ckpts:
        for cond in args.conds:
            t0 = time.time()
            res = run(args.agent, args.seed, ck, cond, args.episodes)
            c = res["counts"]
            mr = c.get("my_ret", 0) / max(1, c.get("my_ret", 0) + c.get("my_miss", 0))
            er = c.get("en_ret", 0) / max(1, c.get("en_ret", 0) + c.get("en_miss", 0))
            print("%s s%-2d ck=%-4s %-11s score=%6.2f len=%6.0f my_ret=%.2f en_ret=%.2f (%.0fs)" % (
                args.agent, args.seed, ck, cond, np.mean(res["scores"]), np.mean(res["lengths"]), mr, er,
                time.time() - t0), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
