"""Visible-state policy used for local TavernGym evaluation and TavernArena.

The policy is deliberately inspectable: every choice uses only ``player_view``
and the current legal-action mask. It never reads hidden game state or sends
model data, weights, or saved games to a remote service.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _minion_value(card: dict[str, Any]) -> float:
    value = float(card.get("atk", 0)) + float(card.get("hp", 0))
    value += 3.0 * float(card.get("tier", 1)) + 6.0 * bool(card.get("golden"))
    keywords = set(card.get("keywords", ()))
    for keyword in ("DIVINE_SHIELD", "TAUNT", "REBORN", "WINDFURY", "VENOMOUS"):
        value += 3.0 * bool(keyword in keywords)
    return value


def _item_value(card: dict[str, Any]) -> float:
    if card.get("type") == "minion":
        return _minion_value(card)
    return 2.0 + float(card.get("tier", 1))


def _owned(me: dict[str, Any]) -> Counter[str]:
    return Counter(
        str(card.get("id", "")) for card in [*me.get("board", ()), *me.get("hand", ())]
    )


DEFAULT_CONFIG = {"upgrade_turn_factor": 2.0, "pair_bonus": 15.0, "refresh_min_gold": 3,
                  "sell_margin": 4.0, "choice_scoring": False, "positioning": False,
                  "position_attack_weight": 1.0, "position_windfury_bonus": 12.0,
                  "position_cleave_bonus": 9.0, "position_deathrattle_bonus": 4.0,
                  "position_taunt_penalty": 8.0, "opponent_aware_positioning": False,
                  "position_max_moves": 1, "tribe_synergy_bonus": 0.0,
                  "tribe_text_bonus": 0.0, "engine_battlecry_bonus": 0.0,
                  "engine_deathrattle_bonus": 0.0, "eternal_knight_bonus": 0.0,
                  "public_prior_weight": 0.0, "replacement_scoring": False,
                  "engine_retention_bonus": 0.0, "low_health_threshold": -1,
                  "low_health_stat_weight": 0.0, "low_health_no_upgrade": False,
                  "low_health_refresh_block": False, "choice_pair_bonus": 0.0,
                  "choice_synergy_weight": 0.0, "choice_economy_bonus": 0.0,
                  "choice_cost_weight": 0.0, "choice_target_weight": 100.0,
                  "choice_scored_kinds": ()}


# These are deliberately a very small set of stable, visible-card interactions.
# They are not a scraped leaderboard model: every fact used below is already in
# the current player view (card id/text/board).  A narrow list makes rollback
# and ablation meaningful, rather than guessing at every card in the pool.
BRANN_ID = "BG_LOE_077"
TITUS_ID = "BG25_354"
ETERNAL_KNIGHT_ID = "BG25_008"

# The public /stats page is a survivorship-biased final-board aggregate.  Use it
# only as a tiny tie-breaker, never as a substitute for live board synergies.
# Positive values are deliberately capped and only apply to exact card ids.
PUBLIC_CARD_PRIOR = {
    "BG_LOE_077": 1.0,       # Brann Bronzebeard
    "BG25_008": 0.6,        # Eternal Knight
    "BG34_500": 0.4,        # Flaming Enforcer
    "BG36_620": -0.8,       # Boom-in-a-Box: poor public final-board average
}


def _dominant_races(me: dict[str, Any]) -> Counter[str]:
    cards = [*me.get("board", ()), *me.get("hand", ())]
    return Counter(race for card in cards for race in card.get("races", ()) if race)


def _synergy_value(card: dict[str, Any], races: Counter[str], config: dict[str, Any]) -> float:
    if not races or max(races.values()) < 2:
        return 0.0
    dominant = {race for race, count in races.items() if count == max(races.values())}
    bonus = config["tribe_synergy_bonus"] if dominant & set(card.get("races", ())) else 0.0
    # Text-only enablers (for example, "your Elementals") can support the visible board tribe.
    text = str(card.get("text", "")).upper()
    if any(race.title() in text.title() for race in dominant):
        bonus += config["tribe_text_bonus"]
    return bonus


def _engine_value(card: dict[str, Any], me: dict[str, Any], owned: Counter[str],
                  config: dict[str, Any]) -> float:
    """Score a few proven engine + payoff relationships from the visible state."""
    board_and_hand = [*me.get("board", ()), *me.get("hand", ())]
    ids = {str(other.get("id", "")) for other in board_and_hand}
    text = str(card.get("text", "")).upper()
    card_id = str(card.get("id", ""))
    value = 0.0
    if BRANN_ID in ids and "BATTLECRY" in text:
        value += float(config["engine_battlecry_bonus"])
    if TITUS_ID in ids and "DEATHRATTLE" in text:
        value += float(config["engine_deathrattle_bonus"])
    # A second/third Eternal Knight is a real scaling path; an isolated one is
    # still judged mainly by its printed stats, not force-bought.
    if card_id == ETERNAL_KNIGHT_ID and owned[ETERNAL_KNIGHT_ID]:
        value += float(config["eternal_knight_bonus"])
    value += float(config["public_prior_weight"]) * PUBLIC_CARD_PRIOR.get(card_id, 0.0)
    return value


def _buy_score(card: dict[str, Any], me: dict[str, Any], owned: Counter[str],
               races: Counter[str], pair_bonus: float, config: dict[str, Any]) -> float:
    """Score shared by normal buying and full-board replacement decisions."""
    pair = pair_bonus if card.get("type") == "minion" and owned[str(card.get("id", ""))] else 0.0
    immediate = _item_value(card)
    if int(me.get("health", 99)) <= int(config["low_health_threshold"]):
        immediate *= 1.0 + float(config["low_health_stat_weight"])
    return immediate + pair + _synergy_value(card, races, config) + _engine_value(card, me, owned, config)


def _best_buy(mask: np.ndarray, shop: list[dict[str, Any]], owned: Counter[str], pair_bonus: float,
              races: Counter[str], me: dict[str, Any], config: dict[str, Any]) -> int | None:
    candidates: list[tuple[float, int]] = []
    for action in np.flatnonzero(mask[8:16]) + 8:
        index = int(action - 8)
        if index >= len(shop):
            continue
        card = shop[index]
        score = _buy_score(card, me, owned, races, pair_bonus, config)
        candidates.append((score, int(action)))
    return max(candidates, default=(0.0, None))[1]


def _play_from_hand(mask: np.ndarray, hand: list[dict[str, Any]], board_size: int) -> int | None:
    for hand_index, card in enumerate(hand[:10]):
        if "MAGNETIC" in set(card.get("keywords", ())):
            for board_index in range(7):
                action = 126 + hand_index * 7 + board_index
                if mask[action]:
                    return action
        for position in (board_size, *range(8)):
            if position > 7:
                continue
            action = 46 + hand_index * 8 + position
            if mask[action]:
                return action
    return None


def _retention_value(card: dict[str, Any], board: list[dict[str, Any]], config: dict[str, Any]) -> float:
    """Avoid selling a small live engine before its visible payoff minions."""
    if not config["replacement_scoring"]:
        return 0.0
    card_id = str(card.get("id", ""))
    board_text = " ".join(str(other.get("text", "")).upper() for other in board)
    if card_id == BRANN_ID and "BATTLECRY" in board_text:
        return float(config["engine_retention_bonus"])
    if card_id == TITUS_ID and "DEATHRATTLE" in board_text:
        return float(config["engine_retention_bonus"])
    if card_id == ETERNAL_KNIGHT_ID and sum(str(other.get("id", "")) == ETERNAL_KNIGHT_ID for other in board) >= 2:
        return float(config["engine_retention_bonus"]) / 2.0
    return 0.0


def _weakest_sell(mask: np.ndarray, board: list[dict[str, Any]], config: dict[str, Any]) -> int | None:
    candidates = [(_minion_value(card) + _retention_value(card, board, config), index)
                  for index, card in enumerate(board[:7]) if mask[16 + index]]
    return min(candidates, default=(0.0, None))[1]


def _choice_score(item: dict[str, Any], me: dict[str, Any], owned: Counter[str],
                  races: Counter[str], config: dict[str, Any]) -> float:
    """Score a pending option from visible card data; never infer hidden offers."""
    if item.get("type") == "minion":
        score = _buy_score(item, me, owned, races, float(config["choice_pair_bonus"]), config)
        score += float(config["choice_synergy_weight"]) * _synergy_value(item, races, config)
    else:
        score = _item_value(item)
    text = str(item.get("text", "")).upper()
    if any(token in text for token in ("GOLD", "COIN", "REFRESH", "DISCOVER")):
        score += float(config["choice_economy_bonus"])
    score -= float(config["choice_cost_weight"]) * float(item.get("cost", 0))
    return score


def _best_choice(mask: np.ndarray, me: dict[str, Any], board: list[dict[str, Any]],
                 owned: Counter[str], races: Counter[str], config: dict[str, Any]) -> int:
    """Choose visible Discover/trinket options or targets with context-aware scores."""
    pending = me.get("pending") or {}
    options = pending.get("options") or []
    targets = pending.get("targets") or []
    best_score, best = float("-inf"), None
    for action in np.flatnonzero(mask[30:46]) + 30:
        index = int(action - 30)
        item = options[index] if index < len(options) else None
        score = _choice_score(item, me, owned, races, config) if isinstance(item, dict) else 0.0
        if index < len(targets) and isinstance(targets[index], dict):
            location = targets[index].get("location") or {}
            if location.get("zone") == "board" and location.get("slot", -1) < len(board):
                score = float(config["choice_target_weight"]) + _minion_value(board[location["slot"]])
        if score > best_score:
            best_score, best = score, int(action)
    return best if best is not None else int(np.flatnonzero(mask[30:46])[0] + 30)


def _should_score_choice(me: dict[str, Any], board: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    """Limit scoring to choice families for which this rule has visible semantics."""
    pending = me.get("pending") or {}
    kind = str(pending.get("kind", "")).lower()
    if kind in set(config["choice_scored_kinds"]):
        return True
    if any(isinstance(item, dict) and item.get("type") == "minion"
           for item in pending.get("options") or []):
        return True
    return any(isinstance(target, dict)
               and (target.get("location") or {}).get("zone") == "board"
               and (target.get("location") or {}).get("slot", -1) < len(board)
               for target in pending.get("targets") or [])


def _next_seen_board(view: dict[str, Any]) -> list[dict[str, Any]]:
    opponent = next((o for o in view.get("opponents", ()) if o.get("is_next_opponent")), {})
    board = opponent.get("seen_board") or opponent.get("last_warband") or []
    return board.get("board", []) if isinstance(board, dict) else board


def _positioning_move(mask: np.ndarray, board: list[dict[str, Any]], state: dict[str, Any], turn: int, config: dict[str, Any], view: dict[str, Any]) -> int | None:
    """Make at most one visible-board reorder per recruit turn.

    Early attackers (Windfury/Cleave/high Attack) move left; Taunts move right.
    """
    if state.get("position_turn") != turn:
        state["position_turn"], state["position_moves"] = turn, 0
    if state.get("position_moves", 0) >= config["position_max_moves"] or len(board) < 2:
        return None
    def priority(card: dict[str, Any]) -> float:
        keys = set(card.get("keywords", ()))
        return config["position_attack_weight"] * float(card.get("atk", 0)) \
            + config["position_windfury_bonus"] * ("WINDFURY" in keys) \
            + config["position_cleave_bonus"] * ("CLEAVE" in keys) \
            + config["position_deathrattle_bonus"] * ("DEATHRATTLE" in str(card.get("text", "")).upper()) \
            - config["position_taunt_penalty"] * ("TAUNT" in keys)
    enemy = _next_seen_board(view)
    enemy_cleave = any("CLEAVE" in set(card.get("keywords", ())) for card in enemy)
    protected = {"BG25_354"}  # Titus Rivendare: Deathrattles trigger an extra time.
    if config["opponent_aware_positioning"] and enemy_cleave and any(card.get("id") in protected for card in board):
        # A left Taunt funnels enemy cleave into slots 0/1.  Keep Titus on the far right,
        # never immediately left of a right-side Taunt where cleave would hit it.
        taunts = [i for i, card in enumerate(board) if "TAUNT" in set(card.get("keywords", ()))]
        supports = [i for i, card in enumerate(board) if card.get("id") in protected]
        middle = [i for i in sorted(range(len(board)), key=lambda i: priority(board[i]), reverse=True)
                  if i not in taunts and i not in supports]
        desired = taunts + middle + supports
    elif config["opponent_aware_positioning"]:
        # No verified cleave threat or no protected deathrattle multiplier: do not disturb
        # the v2 order based on guesses about hidden opponent state.
        state["positioned_turn"] = turn
        return None
    else:
        desired = sorted(range(len(board)), key=lambda i: priority(board[i]), reverse=True)
    for destination, source in enumerate(desired):
        if source != destination:
            action = 196 + source * 7 + destination
            if mask[action]:
                state["position_moves"] += 1
                return action
    return None


def act_with_config(observation: dict[str, np.ndarray], context: dict[str, Any], config: dict[str, Any]) -> int:
    """Return one integer action that is legal under the supplied action mask."""
    config = {**DEFAULT_CONFIG, **config}
    mask = np.asarray(observation["action_mask"], dtype=bool)
    if mask.shape != (245,) or not mask.any():
        raise ValueError("Arena observation has no legal action")

    view = context.get("player_view", {})
    me = view.get("me", {})
    board = list(me.get("board", ()))
    hand = list(me.get("hand", ()))
    shop = list(me.get("shop", ()))
    owned = _owned(me)
    races = _dominant_races(me)

    choices = np.flatnonzero(mask[30:46]) + 30
    if choices.size:
        if config["choice_scoring"] and _should_score_choice(me, board, config):
            return _best_choice(mask, me, board, owned, races, config)
        return int(choices[0])

    play = _play_from_hand(mask, hand, len(board))
    if play is not None:
        return play

    gold = int(me.get("gold", 0))
    health = int(me.get("health", 99))
    low_health = health <= int(config["low_health_threshold"])
    tier = int(me.get("tier", 1))
    turn = int(view.get("turn", context.get("turn", 1)))
    upgrade_price = int(me.get("upgrade_price", 99))
    can_upgrade_for_health = not (low_health and config["low_health_no_upgrade"])
    if mask[3] and can_upgrade_for_health and (upgrade_price <= 3 or (turn > config["upgrade_turn_factor"] * tier and gold >= upgrade_price)):
        return 3

    buy = _best_buy(mask, shop, owned, float(config["pair_bonus"]), races, me, config)
    board_full = len(board) >= 7
    minions_owned = len(board) + sum(card.get("type") == "minion" for card in hand)
    if board_full and buy is not None:
        shop_card = shop[buy - 8]
        weakest = _weakest_sell(mask, board, config)
        replacement_value = _buy_score(shop_card, me, owned, races, float(config["pair_bonus"]), config) \
            if config["replacement_scoring"] else _item_value(shop_card)
        retained_value = _minion_value(board[weakest]) + _retention_value(board[weakest], board, config) \
            if weakest is not None else float("inf")
        if weakest is not None and replacement_value > retained_value + config["sell_margin"]:
            return 16 + weakest
    if buy is not None:
        card = shop[buy - 8]
        pair = card.get("type") == "minion" and owned[str(card.get("id", ""))] > 0
        if card.get("type") != "minion" or minions_owned < 7 or pair:
            return buy

    powers = np.flatnonzero(mask[4:6]) + 4
    if gold >= 1 and powers.size:
        return int(powers[0])
    if mask[6] and minions_owned < 7:
        return 6

    activates = np.flatnonzero(mask[23:30]) + 23
    if activates.size:
        return int(activates[0])
    if config["positioning"]:
        move = _positioning_move(mask, board, context.setdefault("state", {}), turn, config, view)
        if move is not None:
            return move
    allow_refresh = not (low_health and config["low_health_refresh_block"] and int(me.get("refresh_price", 1)) > 0)
    if mask[1] and allow_refresh and gold >= config["refresh_min_gold"]:
        return 1
    if mask[0]:
        return 0
    return int(np.flatnonzero(mask)[0])


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    """Default stable policy; rule-search candidates never change this implicitly."""
    return act_with_config(observation, context, DEFAULT_CONFIG)
