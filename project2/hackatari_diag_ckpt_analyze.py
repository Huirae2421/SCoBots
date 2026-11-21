"""
hackatari_diag_ckpt_analyze.py — hackatari_diag_ckpt.py 결과 요약 (E1/E2)

  PYTHONPATH=. python project2/hackatari_diag_ckpt_analyze.py
출력: 화면 + project2/results/diag_ckpt/summary.md

E2 분해: Lazy 의 내 받아치기 성공률 하락을
  (a) 공 종류(|dx|,|dy|) 구성이 바뀐 몫  (b) 같은 종류 공에서 성공률이 떨어진 몫  으로 나눈다.
  기준 = base 의 종류별 성공률. '구성만 Lazy' 예측값 = Σ_k  w_lazy(k) · rate_base(k)
"""
import glob
import json
from collections import defaultdict
import numpy as np

D = "project2/results/diag_ckpt"
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def load(tag):
    rows = []
    for p in sorted(glob.glob(f"{D}/{tag}_*_seed*.jsonl")):
        for l in open(p):
            rows.append(json.loads(l))
    return rows


def rates(r):
    c = r["counts"]
    my = c.get("my_ret", 0) / max(1, c.get("my_ret", 0) + c.get("my_miss", 0))
    en = c.get("en_ret", 0) / max(1, c.get("en_ret", 0) + c.get("en_miss", 0))
    return my, en


def fast(r):
    h = {int(k): v for k, v in r["dx_hist"].items()}
    tot = sum(v for k, v in h.items() if k in (4, 8, 12))
    return sum(v for k, v in h.items() if k in (8, 12)) / max(1, tot)


def balls_per_ep(r):
    return len(r["segs"]) / r["episodes"]


# ---------- E1 ----------
e1 = load("e1")
if e1:
    out("## E1 — 체크포인트별 (10판, 평가 시드 84)")
    by = defaultdict(dict)
    for r in e1:
        by[(r["agent"], r["seed"], r["ckpt"])][r["cond"]] = r
    order = lambda k: 99 if k == "best" else float(k)
    for agent, seed in sorted({(a, s) for a, s, _ in by}):
        out(f"\n### {agent} seed{seed}")
        out("| ckpt | base 점수 | Lazy 점수 | 차이 | base 판길이 | base 빠른공 | Lazy 빠른공 | 내 받기 base→Lazy | 적 받기 base→Lazy |")
        out("|---|---|---|---|---|---|---|---|---|")
        for ck in sorted({k for a, s, k in by if a == agent and s == seed}, key=order):
            d = by[(agent, seed, ck)]
            if "base" not in d or "Lazy" not in d:
                continue
            b, z = d["base"], d["Lazy"]
            bm, be = rates(b)
            zm, ze = rates(z)
            out("| %s | %.1f | %.1f | %.1f | %.0f | %.2f | %.2f | %.2f→%.2f | %.2f→%.2f |" % (
                ck, np.mean(b["scores"]), np.mean(z["scores"]), np.mean(b["scores"]) - np.mean(z["scores"]),
                np.mean(b["lengths"]), fast(b), fast(z), bm, zm, be, ze))

# ---------- E2 (E3 상세에도 재사용) ----------
def cond_tables(rows, title):
    out(title)
    by = defaultdict(dict)
    for r in rows:
        by[(r["agent"], r["seed"])][r["cond"]] = r
    conds = ["base", "Lazy", "LazyNoBallX", "LazyOld", "Perturb"]
    for (agent, seed), d in sorted(by.items()):
        out(f"\n### {agent} seed{seed}")
        out("| 조건 | 점수 | 판길이 | 판당 받아야 할 공 | 빠른공 비율 | 내 받기 | 적 받기 |")
        out("|---|---|---|---|---|---|---|")
        for c in conds:
            if c not in d:
                continue
            r = d[c]
            m, e = rates(r)
            out("| %s | %.2f ± %.2f | %.0f | %.1f | %.2f | %.3f | %.3f |" % (
                c, np.mean(r["scores"]), np.std(r["scores"]), np.mean(r["lengths"]), balls_per_ep(r), fast(r), m, e))

    out("\n### 분해: Lazy 의 내 받아치기 성공률 하락 = 공 구성 변화 몫 + 같은 공에서의 하락 몫")
    out("| 에이전트 seed | base 실제 | Lazy 실제 | Lazy 구성 + base 성공률(예측) | 구성 몫 | 같은 공 하락 몫 | 종류(base에 없던 공 비율) |")
    out("|---|---|---|---|---|---|---|")
    for (agent, seed), d in sorted(by.items()):
        if "base" not in d or "Lazy" not in d:
            continue
        key = lambda s: (s[0], min(s[1], 6))  # (|dx|, |dy| 상한 6)
        rb = defaultdict(list)
        for s in d["base"]["segs"]:
            rb[key(s)].append(s[4])
        zl = [s for s in d["Lazy"]["segs"]]
        base_rate = np.mean([s[4] for s in d["base"]["segs"]])
        lazy_rate = np.mean([s[4] for s in zl])
        pred, unseen = [], 0
        for s in zl:
            k = key(s)
            if k in rb and len(rb[k]) >= 5:
                pred.append(np.mean(rb[k]))
            else:
                unseen += 1
                pred.append(base_rate)   # base에서 거의 없던 종류는 base 전체 평균으로 대체
        pred = np.mean(pred)
        out("| %s s%d | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f |" % (
            agent, seed, base_rate, lazy_rate, pred, base_rate - pred, pred - lazy_rate, unseen / max(1, len(zl))))

    out("\n### 종류별 내 받아치기 성공률 (|dx|, |dy|) — base vs Lazy (n)")
    for (agent, seed), d in sorted(by.items()):
        if "base" not in d or "Lazy" not in d:
            continue
        out(f"\n{agent} seed{seed}")
        out("| (|dx|,|dy|) | base 비율 | Lazy 비율 | base 성공률 (n) | Lazy 성공률 (n) |")
        out("|---|---|---|---|---|")
        agg = {}
        for c in ("base", "Lazy"):
            g = defaultdict(list)
            for s in d[c]["segs"]:
                g[(s[0], min(s[1], 6))].append(s[4])
            agg[c] = g
        nb = sum(len(v) for v in agg["base"].values())
        nz = sum(len(v) for v in agg["Lazy"].values())
        for k in sorted(set(agg["base"]) | set(agg["Lazy"])):
            vb, vz = agg["base"].get(k, []), agg["Lazy"].get(k, [])
            if len(vb) + len(vz) < 10:
                continue
            out("| %s | %.2f | %.2f | %s | %s |" % (
                k, len(vb) / nb, len(vz) / nz,
                "%.2f (%d)" % (np.mean(vb), len(vb)) if vb else "- (0)",
                "%.2f (%d)" % (np.mean(vz), len(vz)) if vz else "- (0)"))

    out("\n### 놓친 공의 위치 — 끝날 때 공 y 3등분 (위/가운데/아래) 비율과 |공y−패들y| 평균")
    out("| 에이전트 seed | 조건 | 놓친 공 수 | 위 | 가운데 | 아래 | |공y−패들y| 평균 |")
    out("|---|---|---|---|---|---|---|")
    for (agent, seed), d in sorted(by.items()):
        allys = [s[2] for c in d for s in d[c]["segs"]]
        if not allys:
            continue
        lo, hi = np.percentile(allys, [33.3, 66.7])
        for c in ("base", "Lazy"):
            if c not in d:
                continue
            miss = [s for s in d[c]["segs"] if s[4] == 0]
            if not miss:
                continue
            ys = np.array([s[2] for s in miss])
            out("| %s s%d | %s | %d | %.2f | %.2f | %.2f | %.1f |" % (
                agent, seed, c, len(miss), np.mean(ys < lo), np.mean((ys >= lo) & (ys <= hi)), np.mean(ys > hi),
                np.mean([abs(s[2] - s[3]) for s in miss])))


e2 = load("e2")
if e2:
    cond_tables(e2, "\n## E2 — 최종(best) 모델, 적 조건별 (20판)")

# ---------- E3: C 판정 (학습 전 고정한 기준, 팀장 결정 2026-11-19) ----------
# 평가에만 쓰는 상대(학습에 안 씀): Lazy(현재), LazyNoBallX
CRIT_LAZY_MEAN = 13.0     # 평가용 Lazy 3시드 평균 ≥ 13  (B 8.32)
CRIT_KEEP = 17.0          # base 각 시드 ≥ 17 (팀장 확정 2026-11-19, 결과 보기 전 고정)
CRIT_LAZY = {"D": 15.0, "D2": 15.0}   # D·D2 최종: 평가용 Lazy 3시드 평균 ≥ 15 (팀장 결정 2026-11-20/21, C 결과 본 뒤 설계라 더 엄격)
D_GATE_BASE, D_GATE_LAZY = 17.0, 13.0   # D seed32 진행 조건 (seed0·16 각각)
e3 = load("e3")
if e3:
    out("\n## E3 — 에이전트 C 판정 (20판, 평가 시드 84)")
    by = defaultdict(dict)
    for r in e3:
        by[(r["agent"], r["seed"])][r["cond"]] = r
    for agent in sorted({a for a, _ in by}):
        seeds = sorted(s for a, s in by if a == agent)
        out(f"\n### {agent} (시드 {seeds})")
        out("| 조건 | " + " | ".join(f"seed{s}" for s in seeds) + " | 평균 |")
        out("|---|" + "---|" * (len(seeds) + 1))
        for c in ["base", "Lazy", "LazyNoBallX", "LazyOld", "Perturb"]:
            vals = [np.mean(by[(agent, s)][c]["scores"]) for s in seeds if c in by[(agent, s)]]
            if len(vals) == len(seeds):
                out(f"| {c} | " + " | ".join("%.2f" % v for v in vals) + " | %.2f |" % np.mean(vals))
        try:
            lazy = [np.mean(by[(agent, s)]["Lazy"]["scores"]) for s in seeds]
            base = [np.mean(by[(agent, s)]["base"]["scores"]) for s in seeds]
            if agent in ("D", "D2") and len(seeds) < 3:
                # D 중간 판정 (팀장 결정 2026-11-20): seed0·16 각각 base ≥ 17, 평가용 Lazy ≥ 13 → seed32 진행
                g = [(s, b_, z_, b_ >= D_GATE_BASE and z_ >= D_GATE_LAZY) for s, b_, z_ in zip(seeds, base, lazy)]
                for s, b_, z_, ok in g:
                    out("- seed%d: base %.2f (≥%.0f), Lazy %.2f (≥%.0f) → %s" % (s, b_, D_GATE_BASE, z_, D_GATE_LAZY, "충족" if ok else "미충족"))
                go = len(g) >= 2 and all(ok for *_, ok in g)
                out("- **최종(중간): %s** (두 시드 각각 충족해야 seed32 진행)" % ("seed32 진행" if go else "seed32 보류"))
                continue
            crit = CRIT_LAZY.get(agent, CRIT_LAZY_MEAN)
            out("- 기준1 평가용 Lazy 평균 ≥ %.0f: %.2f → %s" % (crit, np.mean(lazy), "통과" if np.mean(lazy) >= crit else "미달"))
            ok2 = min(base) >= CRIT_KEEP
            out("- 기준2 base 각 시드 ≥ %.0f: %s → %s" % (CRIT_KEEP, ["%.2f" % v for v in base], "통과" if ok2 else "미달"))
            ok1 = np.mean(lazy) >= crit
            out("- **최종: %s** (기준1·기준2 모두 통과해야 성공)" % ("성공" if ok1 and ok2 else "미달"))
        except KeyError:
            out("- 판정: base/Lazy 결과가 모두 있어야 함")
    cond_tables(e3, "\n## E3 상세 — B vs C 적 조건별 · 분해 (20판)")

# ---------- E4: 적이 내 공을 받아내는 이유 (적 위치) ----------
# esegs = [시작 적y(내가 친 순간), 끝 공y, 끝 적y, 결과(1=적이 받아침)]
# '필요 이동' = |끝 공y − 시작 적y| : 적이 제자리에 있었다면 얼마나 움직여야 했나
BINS = [(0, 10), (10, 20), (20, 35), (35, 60), (60, 999)]


def bin_of(v):
    for i, (a, b) in enumerate(BINS):
        if a <= v < b:
            return i
    return len(BINS) - 1


e4 = load("e4")
if e4:
    out("\n## E4 — 적 쪽 분석: 적이 내 공을 받아내는 이유 (20판)")
    by = defaultdict(dict)
    for r in e4:
        by[(r["agent"], r["seed"])][r["cond"]] = r
    out("\n### 점수 재현 확인 (e3와 같아야 함) · 적 쪽 요약")
    out("| 에이전트 seed | 조건 | 점수 | 적 쪽 공 수 | 적 받기 | 필요 이동 평균 | 실제 적 이동 평균 | 끝 |공y−적y| (받음/놓침) |")
    out("|---|---|---|---|---|---|---|---|")
    for (agent, seed), d in sorted(by.items()):
        for c in ("base", "Lazy"):
            if c not in d:
                continue
            E = np.array(d[c].get("esegs", []), dtype=float)
            if len(E) == 0:
                continue
            need = np.abs(E[:, 1] - E[:, 0])
            move = np.abs(E[:, 2] - E[:, 0])
            gap = np.abs(E[:, 1] - E[:, 2])
            ok = E[:, -1] == 1
            out("| %s s%d | %s | %.2f | %d | %.3f | %.1f | %.1f | %.1f / %.1f |" % (
                agent, seed, c, np.mean(d[c]["scores"]), len(E), ok.mean(), need.mean(), move.mean(),
                gap[ok].mean() if ok.any() else np.nan, gap[~ok].mean() if (~ok).any() else np.nan))

    out("\n### 필요 이동 구간별: 비율 / 적 받기 — base vs Lazy")
    out("| 에이전트 seed | 조건 | " + " | ".join("%d–%s" % (a, "" if b == 999 else b) for a, b in BINS) + " |")
    out("|---|---|" + "---|" * len(BINS))
    for (agent, seed), d in sorted(by.items()):
        for c in ("base", "Lazy"):
            if c not in d or not d[c].get("esegs"):
                continue
            E = np.array(d[c]["esegs"], dtype=float)
            bi = np.array([bin_of(v) for v in np.abs(E[:, 1] - E[:, 0])])
            cells = []
            for i in range(len(BINS)):
                m = bi == i
                cells.append("%.2f / %s" % (m.mean(), "%.2f" % E[m, -1].mean() if m.sum() >= 5 else "-"))
            out("| %s s%d | %s | " % (agent, seed, c) + " | ".join(cells) + " |")

    out("\n### 분해: Lazy 에서 적 받기 상승 = 필요 이동 구성 변화 몫 + 같은 거리에서 더 잘 받는 몫")
    out("| 에이전트 seed | base 적 받기 | Lazy 적 받기 | Lazy 구성 + base 구간별 받기(예측) | 구성 몫 | 같은 거리 몫 |")
    out("|---|---|---|---|---|---|")
    for (agent, seed), d in sorted(by.items()):
        if "base" not in d or "Lazy" not in d or not d["base"].get("esegs"):
            continue
        Eb = np.array(d["base"]["esegs"], dtype=float)
        El = np.array(d["Lazy"]["esegs"], dtype=float)
        bb = np.array([bin_of(v) for v in np.abs(Eb[:, 1] - Eb[:, 0])])
        bl = np.array([bin_of(v) for v in np.abs(El[:, 1] - El[:, 0])])
        rb = Eb[:, -1].mean()
        rl = El[:, -1].mean()
        rate = {i: (Eb[bb == i, -1].mean() if (bb == i).sum() >= 5 else rb) for i in range(len(BINS))}
        pred = np.mean([rate[i] for i in bl])
        out("| %s s%d | %.3f | %.3f | %.3f | %.3f | %.3f |" % (agent, seed, rb, rl, pred, pred - rb, rl - pred))

# ---------- E5: 에이전트가 공으로 적 위치를 '간접적으로' 아는가 ----------
# esegs = [시작 적y, 끝 공y, 끝 적y, 시작 공y, 결과], track = [(공y, 적y)] (공이 내 쪽으로 오는 동안)
# 측정1: 내가 치는 순간 공y-적y 상관 (base 높고 Lazy 낮으면 '공으로 적 위치 추정' 가능성이 Lazy에서 깨짐)
# 측정2: 공이 내 쪽으로 오는 동안 공y-적y 상관
# 측정3: base에서 적y ≈ a + b·(친 순간 공y) 를 맞추고, 같은 식으로 '공만 보고 추정한 필요 이동'을 계산.
#        Lazy에서도 에이전트가 '추정 기준 20–35' 공을 비슷한 비율로 치는데 실제 적 기준 20–35 가 줄면
#        → 에이전트 행동은 그대로이고, 공으로 추정한 적 위치가 틀어진 것.
e5 = load("e5")
if e5:
    out("\n## E5 — 공으로 적 위치를 간접적으로 아는가 (20판)")
    by = defaultdict(dict)
    for r in e5:
        by[(r["agent"], r["seed"])][r["cond"]] = r
    out("| 에이전트 seed | 조건 | 점수 | 친 순간 상관 | 친 순간 |공y−적y| | 오는 동안 상관 | 추정식 R² | 추정 20–35 비율 | 실제 20–35 비율 | 추정 20–35 공의 적 받기 | 전체 적 받기 |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (agent, seed), d in sorted(by.items()):
        if "base" not in d or not d["base"].get("esegs"):
            continue
        Eb = np.array(d["base"]["esegs"], dtype=float)
        b, a = np.polyfit(Eb[:, 3], Eb[:, 0], 1)
        for c in ("base", "Lazy"):
            if c not in d:
                continue
            E = np.array(d[c]["esegs"], dtype=float)
            T = np.array(d[c]["track"], dtype=float)
            est = a + b * E[:, 3]
            r2 = 1 - np.sum((E[:, 0] - est) ** 2) / np.sum((E[:, 0] - E[:, 0].mean()) ** 2)
            need_est = np.abs(E[:, 1] - est)
            need_act = np.abs(E[:, 1] - E[:, 0])
            m_est = (need_est >= 20) & (need_est < 35)
            m_act = (need_act >= 20) & (need_act < 35)
            out("| %s s%d | %s | %.2f | %.2f | %.1f | %.2f | %.2f | %.2f | %.2f | %s | %.3f |" % (
                agent, seed, c, np.mean(d[c]["scores"]),
                np.corrcoef(E[:, 3], E[:, 0])[0, 1], np.mean(np.abs(E[:, 3] - E[:, 0])),
                np.corrcoef(T[:, 0], T[:, 1])[0, 1], r2, m_est.mean(), m_act.mean(),
                "%.2f" % E[m_est, -1].mean() if m_est.sum() >= 5 else "-", E[:, -1].mean()))

# ---------- E6: 개입 실험 — Lazy에서 '받아친 순간 적을 공 높이로' 옮기면 회복되나 ----------
# 판정 (결과 보기 전 고정, 2026-11-20): Lazy에서 무너진 seed16·32 (B, C 각각)에서
#   LazySnapBall ≥ 17 이고 LazySnapRand < 17 이면 → '적y≈공y 관계가 깨진 것이 붕괴 원인' 확정
SNAP_OK = 17.0
e6 = load("e6")
if e6:
    out("\n## E6 — 개입 실험: 받아친 순간 적 위치를 바꿔 놓기 (20판, 평가 시드 84)")
    ref = defaultdict(dict)
    for r in e3:
        ref[(r["agent"], r["seed"])][r["cond"]] = r
    for r in e6:
        ref[(r["agent"], r["seed"])][r["cond"]] = r
    out("| 에이전트 seed | base | Lazy | Lazy+공높이로 | Lazy+무작위 | 판당 옮긴 횟수 / 판당 내 받기 (공높이) | 적 받기 base / Lazy / 공높이 / 무작위 |")
    out("|---|---|---|---|---|---|---|")
    verdict = []
    for (agent, seed), d in sorted(ref.items()):
        if "LazySnapBall" not in d:
            continue
        sc = {c: np.mean(d[c]["scores"]) if c in d else np.nan for c in ("base", "Lazy", "LazySnapBall", "LazySnapRand")}
        en = {c: rates(d[c])[1] if c in d else np.nan for c in sc}
        sb = d["LazySnapBall"]
        out("| %s s%d | %.2f | %.2f | %.2f | %.2f | %.1f / %.1f | %.2f / %.2f / %.2f / %.2f |" % (
            agent, seed, sc["base"], sc["Lazy"], sc["LazySnapBall"], sc["LazySnapRand"],
            (sb.get("snaps") or 0) / sb["episodes"], sb["counts"].get("my_ret", 0) / sb["episodes"],
            en["base"], en["Lazy"], en["LazySnapBall"], en["LazySnapRand"]))
        if seed in (16, 32):
            verdict.append(sc["LazySnapBall"] >= SNAP_OK and sc["LazySnapRand"] < SNAP_OK)
    if verdict:
        out("- 판정 (seed16·32, B·C 모두 공높이 ≥ %.0f 이고 무작위 < %.0f): %d/%d 충족 → **%s**" % (
            SNAP_OK, SNAP_OK, sum(verdict), len(verdict), "원인 확정" if all(verdict) else "미확정"))

# ---------- E7: 다른 평가 시드에서도 C > B 인가 ----------
# 판정 (결과 보기 전 고정): 평가 시드 84·85·86 각각에서 C의 Lazy 3시드 평균 > B의 Lazy 3시드 평균 → 개선 확정
#  (기준1 '13 이상' 판정은 원래대로 평가 시드 84 결과만 사용. 여기 85·86의 13 비교는 참고용)
e7 = load("e7s8*")
if e7:
    out("\n## E7 — 평가 시드별 B vs C (20판씩)")
    allr = [dict(r, eval_seed=r.get("eval_seed", 84)) for r in e3] + e7
    tab = defaultdict(dict)
    for r in allr:
        tab[(r["eval_seed"], r["agent"], r["cond"])][r["seed"]] = np.mean(r["scores"])
    out("| 평가 시드 | 에이전트 | base (s0/s16/s32) | base 평균 | Lazy (s0/s16/s32) | Lazy 평균 |")
    out("|---|---|---|---|---|---|")
    ok = []
    for es in sorted({k[0] for k in tab}):
        means = {}
        for ag in ("B", "C"):
            b = tab.get((es, ag, "base"), {})
            z = tab.get((es, ag, "Lazy"), {})
            if len(z) < 3:
                continue
            means[ag] = np.mean(list(z.values()))
            out("| %d | %s | %s | %.2f | %s | %.2f |" % (
                es, ag, " / ".join("%.2f" % b[s] for s in sorted(b)), np.mean(list(b.values())) if b else np.nan,
                " / ".join("%.2f" % z[s] for s in sorted(z)), means[ag]))
        if "B" in means and "C" in means:
            ok.append(means["C"] > means["B"])
    if ok:
        out("- 판정 (평가 시드마다 C Lazy 평균 > B Lazy 평균): %d/%d → **%s**" % (sum(ok), len(ok), "개선 확정" if all(ok) else "미확정"))

with open(f"{D}/summary.md", "w") as f:
    f.write("# 확장 E1/E2 진단 요약\n\n" + "\n".join(lines) + "\n")
print(f"\n저장: {D}/summary.md")
