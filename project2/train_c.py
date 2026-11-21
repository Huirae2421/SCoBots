"""
train_c.py — 확장 E3: 에이전트 C 학습 (B와 같은 설정 + 학습 상대만 섞음)

C = B(18차원 table4 focus, 적 제거, -r env, PPO·20M·시드 동일) 에서 '학습 상대'만 바꾼 에이전트.
  - 학습 환경 8개: 판(episode)마다 상대를 무작위로 고름 (1/3씩)
      base     : 기본 적 AI
      Perturb  : 적 y에 매 호출 ±5 잡음 (HackAtari random_perturbation_enemy와 같은 규칙)
      LazyOld  : 수정 전 lazy_enemy (HackAtari @688e27c, 적이 한 박자 늦게 따라감)
  - 평가 환경(EvalCallback, best_model 선택용)은 B와 똑같이 base만 → 평가용 상대가 모델 선택에 새지 않음
  - 평가에만 쓰는 상대(학습에 절대 안 씀): 현재 Lazy(@8ca2bdf, 적이 멈춤), LazyNoBallX
  - 엄격안 (팀장 결정 2026-11-19)

학습 루프는 B를 학습한 project2/train_long.py 를 그대로 쓰고, 아래 3가지만 바꿔 끼운다(원본 파일 수정 없음).
  1) scobi Environment 생성 시 시드를 기억 → 환경별 난수 (재현 가능, 환경끼리 독립)
  2) ocgym.make → HackAtari + 상대 섞기
  3) 실험 이름 끝에 "_C-mix" → 체크포인트 폴더가 B와 겹치지 않음

실행 (저장소 루트, 오래 걸림 → 밤에):
  SCOBI_EXCLUDE_OBJECTS=Enemy PYTHONPATH=. python project2/train_c.py -g Pong -s 0 -env 8 -r env -p external
확인만 (학습 없음, 환경만 띄워서 섞임·차원·폴더 이름 점검):
  SCOBI_EXCLUDE_OBJECTS=Enemy PYTHONPATH=. python project2/train_c.py -g Pong -s 0 -env 8 -r env -p external --check
"""
import os
import sys
import importlib.util
from collections import Counter

import numpy as np

MODES = ("base", "Perturb", "LazyOld")
EXP_SUFFIX = "_C-mix"
REPORT_EVERY = 200          # 환경마다 이 판 수마다 상대 분포를 로그에 출력

# ---- 실행 옵션 중 train_c 전용(--check) 분리 ----
CHECK = "--check" in sys.argv
if CHECK:
    sys.argv.remove("--check")


def _seed_arg():
    for i, a in enumerate(sys.argv):
        if a in ("-s", "--seed"):
            return int(sys.argv[i + 1])
    raise SystemExit("-s 시드를 지정하세요")


TRAIN_SEED = _seed_arg()
EVAL_SEED_START = (TRAIN_SEED + 42) * 2   # train_long.py의 eval_env_seed 규칙과 같음

if os.environ.get("SCOBI_EXCLUDE_OBJECTS") != "Enemy":
    raise SystemExit("C는 B와 같이 SCOBI_EXCLUDE_OBJECTS=Enemy 가 필요합니다")

# ---- 1) Environment 생성 시드 기억 ----
import scobi.core as scobi_core  # noqa: E402

_CUR_SEED = [None]
_orig_env_init = scobi_core.Environment.__init__


def _env_init(self, env_name, seed=None, *args, **kwargs):
    _CUR_SEED[0] = seed
    _orig_env_init(self, env_name, seed, *args, **kwargs)


scobi_core.Environment.__init__ = _env_init

# ---- 2) ocgym.make → HackAtari + 상대 섞기 ----
import scobi.environments.ocgym as ocgym  # noqa: E402
from hackatari.core import HackAtari  # noqa: E402


class OpponentMixer:
    """한 환경(HackAtari) 안에서 판마다 상대를 고르고, 해당 상대의 수정을 적용."""

    def __init__(self, env, seed):
        self.env = env
        self.rng = np.random.RandomState(10_000 + (seed or 0))
        self.mode = "base"
        self.counts = Counter()
        self.lazy = {"last": 127, "prev_x": 130}

    def on_reset(self):
        self.mode = MODES[self.rng.randint(len(MODES))]
        self.counts[self.mode] += 1
        self.lazy = {"last": 127, "prev_x": 130}
        n = sum(self.counts.values())
        if n % REPORT_EVERY == 0:
            print("[C-mix] env_seed=%s episodes=%d %s" % (
                _CUR_SEED[0], n, dict(self.counts)), flush=True)

    def perturb(self):
        if self.mode != "Perturb":
            return
        ram = self.env.get_ram()
        p = self.rng.randint(-5, 6)
        self.env.set_ram(21, min(max(0, p + int(ram[21])), 250))

    def lazy_old(self):
        if self.mode != "LazyOld":
            return
        ram = self.env.get_ram()
        st = self.lazy
        if 0 < ram[11] < 5:
            self.env.set_ram(21, 127)
            self.env.set_ram(49, 130)
        if st["prev_x"] < ram[49]:
            self.env.set_ram(21, st["last"])
        st["prev_x"] = ram[49]
        st["last"] = ram[21]


MIXERS = []


def _patched_make(env_name, *args, notify=False, **kwargs):
    env = HackAtari(env_name, modifs=[], mode="ram", **kwargs)
    seed = _CUR_SEED[0]
    if seed is not None and seed >= EVAL_SEED_START:
        return env                      # 평가 환경: base 그대로 (B와 같은 모델 선택 기준)
    m = OpponentMixer(env, seed)
    env.reset_modifs.append(m.on_reset)
    env.step_modifs.append(m.perturb)            # HackAtari perturbation처럼 step+post 양쪽
    env.post_detection_modifs.append(m.perturb)
    env.step_modifs.append(m.lazy_old)           # 수정 전 lazy처럼 step에만
    MIXERS.append(m)
    return env


ocgym.make = _patched_make

# ---- 3) 실험 이름에 _C-mix ----
import utils.parser.parser as P  # noqa: E402

_orig_parse = P.parse_train


def _parse_train():
    d = _orig_parse()
    d["exp_name"] = d["exp_name"] + EXP_SUFFIX
    return d


P.parse_train = _parse_train


def check():
    """학습 없이 점검: 폴더 이름, 관측 차원, 상대 섞임, 평가 환경이 base인지."""
    from scobi import Environment
    d = P.parse_train()
    print("실험 이름:", d["exp_name"])
    print("체크포인트 폴더:", "resources/checkpoints/" + d["exp_name"],
          "(이미 있음!)" if os.path.exists("resources/checkpoints/" + d["exp_name"]) else "(새로 생성됨)")
    print("focus:", d["focus_dir"], d["pruned_ff_name"])
    rng = np.random.RandomState(0)
    stats = {m: [] for m in MODES}
    for rank in range(8):
        env = Environment(d["env"], seed=TRAIN_SEED + rank, focus_dir=d["focus_dir"],
                          focus_file=d["pruned_ff_name"], silent=True, refresh_yaml=False)
        if rank == 0:
            print("관측 차원:", env.observation_space.shape, "(B와 같으려면 18)")
        mix = MIXERS[-1]
        hack = env.oc_env
        for ep in range(4):
            env.reset(seed=TRAIN_SEED + rank + 1000 * ep)
            prev = hack.get_ram().copy()
            moved = []
            for _ in range(600):
                _, _, term, trunc, _ = env.step(rng.randint(env.action_space.n))
                r = hack.get_ram().copy()
                dx = int(r[49]) - int(prev[49])
                if r[54] != 0 and r[49] > 49 and prev[54] != 0 and prev[49] > 49 and 0 < dx < 20:
                    moved.append(int(r[50]) != int(prev[50]))
                prev = r
                if term or trunc:
                    break
            if moved:
                stats[mix.mode].append(np.mean(moved))
        print("  train env seed=%d 상대 분포 %s" % (TRAIN_SEED + rank, dict(mix.counts)))
        env.close()
    print("상대별 '공이 내 쪽으로 올 때 적이 움직인 비율' (base≈0.99, LazyOld≈0.98, Perturb는 잡음 때문에 높음):")
    for m in MODES:
        print("  %-8s %s" % (m, "%.2f (판 %d)" % (np.mean(stats[m]), len(stats[m])) if stats[m] else "-"))
    n_before = len(MIXERS)
    ev = Environment(d["env"], seed=EVAL_SEED_START, focus_dir=d["focus_dir"],
                     focus_file=d["pruned_ff_name"], silent=True, refresh_yaml=False)
    print("평가 환경(seed=%d) 섞기 적용 여부:" % EVAL_SEED_START,
          "적용됨 (문제!)" if len(MIXERS) > n_before else "적용 안 됨 = base (정상)")
    ev.close()


if __name__ == "__main__":
    if CHECK:
        check()
    else:
        spec = importlib.util.spec_from_file_location("train_long", "project2/train_long.py")
        tl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tl)
        tl.main()
