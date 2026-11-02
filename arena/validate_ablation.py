"""Compare one ablation policy to v2 on identical local seeds."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
from statistics import mean
from local_evaluate import run_game
from smart_agent_v2 import act as v2

def load(spec):
    module,_,name=spec.partition(":")
    return getattr(importlib.import_module(module),name)
def stats(policy,seeds):
    p=[run_game(s,policy) for s in seeds]
    return {"mean_placement":mean(p),"wins":p.count(1),"top2":sum(x<=2 for x in p),"top4":sum(x<=4 for x in p),"eighth":p.count(8),"games":len(p)}
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--policy",required=True);a.add_argument("--seed",type=int,required=True);a.add_argument("--games",type=int,default=512);a.add_argument("--out",type=Path,required=True);x=a.parse_args()
    base,candidate=stats(v2,range(x.seed,x.seed+x.games)),stats(load(x.policy),range(x.seed,x.seed+x.games))
    approved=(candidate["mean_placement"]<=base["mean_placement"]-.15 and candidate["top4"]>=base["top4"] and candidate["eighth"]<=base["eighth"])
    r={"policy":x.policy,"seed":x.seed,"baseline_v2":base,"candidate":candidate,"mean_placement_improvement":base["mean_placement"]-candidate["mean_placement"],"approved":approved};x.out.parent.mkdir(parents=True,exist_ok=True);x.out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"out":str(x.out),"policy":x.policy,"baseline_mean":round(base["mean_placement"],3),"candidate_mean":round(candidate["mean_placement"],3),"approved":approved}))
if __name__=="__main__":main()
