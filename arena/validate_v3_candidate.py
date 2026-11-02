"""Independent fixed-seed validation of the v3 candidate against promoted v2."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from local_evaluate import run_game
from smart_agent_v2 import act as v2
from smart_agent_v3_candidate import act as v3

def stats(policy, seeds):
    places=[run_game(seed, policy) for seed in seeds]
    return {"mean_placement":mean(places),"wins":places.count(1),"top2":sum(p<=2 for p in places),
            "top4":sum(p<=4 for p in places),"eighth":places.count(8),"games":len(places)}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--games",type=int,default=512); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    base, cand=stats(v2,range(a.seed,a.seed+a.games)),stats(v3,range(a.seed,a.seed+a.games))
    approved=(cand["mean_placement"]<=base["mean_placement"]-0.15 and cand["top4"]>=base["top4"] and cand["eighth"]<=base["eighth"])
    report={"seed":a.seed,"baseline_v2":base,"candidate_v3":cand,
            "mean_placement_improvement":base["mean_placement"]-cand["mean_placement"],"approved":approved}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"out":str(a.out),"baseline_mean":round(base["mean_placement"],3),"candidate_mean":round(cand["mean_placement"],3),"approved":approved}))
if __name__=="__main__":main()
