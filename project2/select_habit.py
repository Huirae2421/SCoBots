"""
select_habit.py — 확장 E11: '습관 진단'으로 체크포인트 고르기 (탐색적 추가 분석, 팀장 결정 2026-11-21)

E10까지 결과를 본 뒤 만든 세 번째 고르는 방식이므로 보고서에는 '탐색적 추가 분석'으로 싣는다. 기존 판정은 그대로.
규칙 (최종 평가 전에 고정):
  1) E10에서 이미 평가한 체크포인트(1M~20M, base·Freeze, 평가 시드 42, 10판)의 base 기록만으로
     '득점용 공 비율' = 적이 20~35 움직여야 하는 공의 비율 을 체크포인트마다 계산
  2) 비율 < 0.10 인 체크포인트만 후보 (12개 에이전트 종합표: 습관 없음 ≤ 0.06, 습관 있음 ≥ 0.22)
  3) 후보 중 min(base, Freeze) 최대 (동점: 평균 → 나중 체크포인트). 후보가 없으면 '해당 없음'
  4) 고른 모델을 평가 시드 84, 20판으로 base / Lazy / LazyNoBallX 평가
  5) 판정 (D와 같은 기준): 에이전트별 Lazy 시드 평균 >= 15 + base 각 시드 >= 17. 시드별 결과도 함께 보고
  판정용 Lazy·LazyNoBallX 는 고르는 데 쓰지 않는다.

  PYTHONPATH=. python project2/select_habit.py pick
  PYTHONPATH=. python project2/select_habit.py report
"""
import sys
import json
import glob
import numpy as np

D = "project2/results/diag_ckpt"
RUNS = [("D", 0), ("D", 16), ("D", 32), ("D2", 0), ("D2", 16), ("D2", 32)]
HABIT_MAX = 0.10
CRIT_LAZY, CRIT_BASE = 15.0, 17.0


def load(path):
    rows = []
    for p in sorted(glob.glob(path)):
        rows += [json.loads(l) for l in open(p)]
    return rows


def habit(r):
    E = np.array(r["esegs"], float)
    need = np.abs(E[:, 1] - E[:, 0])
    return float(np.mean((need >= 20) & (need < 35)))


def pick():
    choices, lines = {}, []
    for ag, s in RUNS:
        tab = {}
        for r in load(f"{D}/sel_{ag}_seed{s}.jsonl"):
            t = tab.setdefault(r["ckpt"], {})
            t[r["cond"]] = np.mean(r["scores"])
            if r["cond"] == "base":
                t["habit"] = habit(r)
        rows = [(k, v) for k, v in tab.items() if {"base", "Freeze", "habit"} <= set(v)]
        cand = [(min(v["base"], v["Freeze"]), (v["base"] + v["Freeze"]) / 2, float(k), k)
                for k, v in rows if v["habit"] < HABIT_MAX]
        cand.sort()
        ch = cand[-1][3] if cand else None
        choices[f"{ag}_{s}"] = ch
        lines.append(f"\n### {ag} seed{s} → " + (f"고른 체크포인트 {ch}M" if ch else "후보 없음 (모든 체크포인트가 습관 있음)"))
        lines.append("| ckpt | base | Freeze | 득점용 공 비율 | 후보 |")
        lines.append("|---|---|---|---|---|")
        for k, v in sorted(rows, key=lambda kv: float(kv[0])):
            mark = "←" if k == ch else ("O" if v["habit"] < HABIT_MAX else "")
            lines.append(f"| {k} | {v['base']:.1f} | {v['Freeze']:.1f} | {v['habit']:.2f} | {mark} |")
    json.dump(choices, open(f"{D}/selh_choices.json", "w"), indent=1)
    print("\n".join(lines))
    for k, v in choices.items():
        if v:
            ag, s = k.split("_")
            print(f"CHOICE {ag} {s} {v}")


def report():
    choices = json.load(open(f"{D}/selh_choices.json"))
    fin = {}
    for r in load(f"{D}/selh_*_seed*.jsonl"):
        d = fin.setdefault((r["agent"], r["seed"]), {})
        d[r["cond"]] = np.mean(r["scores"])
        if r["cond"] in ("base", "Lazy"):
            d["habit_" + r["cond"]] = habit(r)
    f2 = lambda d, c: "%.2f" % d[c] if c in d else "-"
    out = ["## E11 — 습관 진단으로 고르기 (탐색적, 평가 시드 84, 20판)",
           "선택: 득점용 공 비율 < %.2f (base 기록만) 중 min(base, Freeze) 최대. 판정: Lazy 시드 평균 >= 15 + base 각 >= 17" % HABIT_MAX,
           "",
           "| 에이전트 seed | 고른 ckpt | base | Lazy | LazyNoBallX | base 득점용 공 비율 (평가 시드 84) |",
           "|---|---|---|---|---|---|"]
    for ag, s in RUNS:
        f = fin.get((ag, s), {})
        ch = choices.get(f"{ag}_{s}")
        out.append("| %s s%d | %s | %s | %s | %s | %s |" % (
            ag, s, f"{ch}M" if ch else "후보 없음", f2(f, "base"), f2(f, "Lazy"), f2(f, "LazyNoBallX"),
            f2(f, "habit_base")))
    for ag in ("D", "D2"):
        seeds = [s for a, s in RUNS if a == ag]
        have = [s for s in seeds if "Lazy" in fin.get((ag, s), {})]
        per = ["s%d %s" % (s, "통과" if fin[(ag, s)]["base"] >= CRIT_BASE and fin[(ag, s)]["Lazy"] >= CRIT_LAZY else "미달")
               for s in have]
        if len(have) < len(seeds):
            out.append("- %s: 후보 없는 시드가 있어 3시드 판정 불가 (고른 시드: %s) → **미달**" % (ag, ", ".join(per) or "없음"))
            continue
        lazy = [fin[(ag, s)]["Lazy"] for s in seeds]
        base = [fin[(ag, s)]["base"] for s in seeds]
        ok = np.mean(lazy) >= CRIT_LAZY and min(base) >= CRIT_BASE
        out.append("- %s: Lazy 평균 %.2f, base %s → **%s** (시드별: %s)" % (
            ag, np.mean(lazy), ["%.2f" % v for v in base], "성공" if ok else "미달", ", ".join(per)))
    txt = "\n".join(out)
    open(f"{D}/selh_summary.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    {"pick": pick, "report": report}[sys.argv[1]]()
