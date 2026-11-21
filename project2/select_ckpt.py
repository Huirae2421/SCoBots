"""
select_ckpt.py — 확장 E10: 모델 선택 방식을 바꾼 추가 실험 (팀장 결정 2026-11-21)

배경: 지금까지 best_model 은 학습 중 base 평가(8판)로만 골랐다 → base 전용 득점법을 많이 쓰는 모델이 뽑히기 쉬움.
방법 (결과 보기 전에 고정):
  1) D(seed 0·16), D2(seed 0·16·32)의 1M~20M 체크포인트 20개를
     학습에 쓴 상대 base, Freeze 에서 10판씩 평가 (평가 시드 42 — 판정용 84와 다름)
  2) 점수 = min(base 평균, Freeze 평균) 이 가장 높은 체크포인트를 고름
     (동점이면 두 점수 평균이 높은 쪽, 그래도 같으면 더 나중 체크포인트)
     판정용 상대(Lazy, LazyNoBallX)는 고르는 데 쓰지 않음
  3) 고른 모델을 평가 시드 84, 20판으로 base / Lazy / LazyNoBallX 평가
  4) 판정 (D와 같은 기준): 에이전트별 Lazy 시드 평균 >= 15 이고 base 각 시드 >= 17 → 성공
     보고서에는 "모델 선택 방식을 바꾼 추가 실험"으로 따로 싣고, 기존 C·D·D2 판정은 그대로 둔다.

  PYTHONPATH=. python project2/select_ckpt.py pick     # 2) 고르기 → sel_choices.json
  PYTHONPATH=. python project2/select_ckpt.py report   # 4) 판정 → sel_summary.md
"""
import sys
import json
import glob
import numpy as np

D = "project2/results/diag_ckpt"
RUNS = [("D", 0), ("D", 16), ("D", 32), ("D2", 0), ("D2", 16), ("D2", 32)]   # D 32는 2026-11-21 추가 (같은 방법)
CRIT_LAZY, CRIT_BASE = 15.0, 17.0


def load(path):
    rows = []
    for p in sorted(glob.glob(path)):
        rows += [json.loads(l) for l in open(p)]
    return rows


def pick():
    choices, lines = {}, []
    for ag, s in RUNS:
        tab = {}
        for r in load(f"{D}/sel_{ag}_seed{s}.jsonl"):
            tab.setdefault(r["ckpt"], {})[r["cond"]] = np.mean(r["scores"])
        cand = [(min(v["base"], v["Freeze"]), (v["base"] + v["Freeze"]) / 2, float(k), k, v)
                for k, v in tab.items() if "base" in v and "Freeze" in v]
        if not cand:
            lines.append(f"{ag} s{s}: 결과 없음")
            continue
        cand.sort(key=lambda c: (c[0], c[1], c[2]))
        best = cand[-1]
        choices[f"{ag}_{s}"] = best[3]
        lines.append(f"\n### {ag} seed{s} → 고른 체크포인트 {best[3]}M (min {best[0]:.2f})")
        lines.append("| ckpt | base | Freeze | min |")
        lines.append("|---|---|---|---|")
        for c in sorted(cand, key=lambda c: c[2]):
            mark = " ←" if c[3] == best[3] else ""
            lines.append(f"| {c[3]} | {c[4]['base']:.1f} | {c[4]['Freeze']:.1f} | {c[0]:.1f}{mark} |")
    json.dump(choices, open(f"{D}/sel_choices.json", "w"), indent=1)
    print("\n".join(lines))
    for k, v in choices.items():
        ag, s = k.split("_")
        print(f"CHOICE {ag} {s} {v}")


def report():
    choices = json.load(open(f"{D}/sel_choices.json"))
    fin, best = {}, {}
    for r in load(f"{D}/self_*_seed*.jsonl"):
        fin.setdefault((r["agent"], r["seed"]), {})[r["cond"]] = np.mean(r["scores"])
    for r in load(f"{D}/e3_D*_seed*.jsonl"):
        best.setdefault((r["agent"], r["seed"]), {})[r["cond"]] = np.mean(r["scores"])
    f2 = lambda d, c: "%.2f" % d[c] if c in d else "-"
    out = ["## E10 — 모델 선택 방식을 바꾼 추가 실험 (평가 시드 84, 20판)",
           "선택: 1M~20M 중 min(base, Freeze) 최대 (평가 시드 42, 10판). 판정: Lazy 시드 평균 >= 15 + base 각 >= 17",
           "",
           "| 에이전트 seed | 고른 ckpt | base | Lazy | LazyNoBallX | (기존 best) base / Lazy |",
           "|---|---|---|---|---|---|"]
    for ag, s in RUNS:
        f, b = fin.get((ag, s), {}), best.get((ag, s), {})
        out.append("| %s s%d | %sM | %s | %s | %s | %s / %s |" % (
            ag, s, choices.get(f"{ag}_{s}", "-"), f2(f, "base"), f2(f, "Lazy"), f2(f, "LazyNoBallX"),
            f2(b, "base"), f2(b, "Lazy")))
    for ag in ("D", "D2"):
        seeds = [s for a, s in RUNS if a == ag]
        try:
            lazy = [fin[(ag, s)]["Lazy"] for s in seeds]
            base = [fin[(ag, s)]["base"] for s in seeds]
        except KeyError:
            out.append(f"- {ag}: 결과 부족")
            continue
        ok1, ok2 = np.mean(lazy) >= CRIT_LAZY, min(base) >= CRIT_BASE
        out.append("- %s: Lazy 평균 %.2f (>=%.0f %s), base %s (각 >=%.0f %s) → **%s**" % (
            ag, np.mean(lazy), CRIT_LAZY, "통과" if ok1 else "미달", ["%.2f" % v for v in base], CRIT_BASE,
            "통과" if ok2 else "미달", "성공" if ok1 and ok2 else "미달"))
    txt = "\n".join(out)
    open(f"{D}/sel_summary.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    {"pick": pick, "report": report}[sys.argv[1]]()
