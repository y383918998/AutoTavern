"""Deterministic, local examples for the opponent-aware positioning rule.

Run: python positioning_examples.py
"""
from __future__ import annotations
import numpy as np
from smart_agent_v4 import act

def card(card_id, atk=1, text="", keywords=()):
    return {"id":card_id,"type":"minion","atk":atk,"hp":5,"tier":4,"text":text,"keywords":list(keywords)}
def move(board, action):
    src, dst=divmod(action-196,7); value=board.pop(src); board.insert(dst,value)
def observation():
    mask=np.zeros(245,dtype=bool); mask[0]=True
    for src in range(4):
        for dst in range(4):
            mask[196+src*7+dst]=True
    return {"action_mask":mask}
def main():
    # Titus starts exposed on the left. The next opponent's known board has Cleave.
    board=[card("BG25_354",text="Your Deathrattles trigger an extra time."),
           card("TANK",atk=5,keywords=("TAUNT",)),card("WF",atk=8,keywords=("WINDFURY",)),card("FILL",atk=3)]
    view={"turn":8,"me":{"board":board,"hand":[],"shop":[],"gold":0,"tier":4,"upgrade_price":99},
          "opponents":[{"is_next_opponent":True,"seen_board":[card("CLEAVER",keywords=("CLEAVE",))]}]}
    state={}; actions=[]
    for _ in range(3):
        action=act(observation(),{"player_view":view,"state":state}); actions.append(action); move(board,action)
    assert actions==[203,211,219], actions
    assert board[0]["id"]=="TANK" and board[-1]["id"]=="BG25_354", board
    # Without a visible cleave threat, v4 makes no positional move.
    safe={**view,"opponents":[{"is_next_opponent":True,"seen_board":[]}]}
    assert act(observation(),{"player_view":safe,"state":{}})==0
    print("PASS: Cleave/Titus rule moves Taunt left and Titus right; no known Cleave means no move.")
if __name__=="__main__":main()
