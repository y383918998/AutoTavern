"""Search only visible-board positioning weights against smart_agent_v2."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from statistics import mean
from local_evaluate import run_game
from smart_agent import act_with_config
from smart_agent_v2 import CONFIG as V2

BASE={**V2,"positioning":True}
GRID=[BASE,
      {**BASE,"position_windfury_bonus":16.0,"position_cleave_bonus":12.0},
      {**BASE,"position_cleave_bonus":15.0,"position_deathrattle_bonus":7.0},
      {**BASE,"position_taunt_penalty":4.0},
      {**BASE,"position_taunt_penalty":12.0},
      {**BASE,"position_attack_weight":1.5,"position_windfury_bonus":8.0}]

def stats(config,seeds):
    policy=lambda o,c:act_with_config(o,c,config);p=[run_game(s,policy) for s in seeds]
    return {"mean_placement":mean(p),"top2":sum(x<=2 for x in p),"top4":sum(x<=4 for x in p),"eighth":p.count(8),"games":len(p)}
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--seed",type=int,required=True);a.add_argument("--rounds",type=int,default=3);a.add_argument("--search-games",type=int,default=32);a.add_argument("--valid-games",type=int,default=128);a.add_argument("--out",type=Path,required=True);x=a.parse_args();rows=[]
    for r in range(x.rounds):
        root=(x.seed+r*500_000_000)%1_900_000_000
        winner,search=min(((c,stats(c,range(root,root+x.search_games))) for c in GRID),key=lambda z:z[1]["mean_placement"])
        base,cand=stats(V2,range(root+100_000_000,root+100_000_000+x.valid_games)),stats(winner,range(root+100_000_000,root+100_000_000+x.valid_games))
        passed=cand["mean_placement"]<=base["mean_placement"]-.10 and cand["top4"]>=base["top4"] and cand["eighth"]<=base["eighth"]
        rows.append({"round":r+1,"winner":winner,"search":search,"baseline":base,"candidate":cand,"passed":passed});print(json.dumps({"round":r+1,"winner":winner,"baseline_mean":round(base["mean_placement"],3),"candidate_mean":round(cand["mean_placement"],3),"passed":passed}),flush=True)
    same=len({json.dumps(r["winner"],sort_keys=True) for r in rows})==1;approved=same and all(r["passed"] for r in rows)
    out={"baseline":V2,"rounds":rows,"approved":approved};x.out.parent.mkdir(parents=True,exist_ok=True);x.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"out":str(x.out),"approved":approved,"winner":rows[0]["winner"] if approved else None}))
if __name__=="__main__":main()
