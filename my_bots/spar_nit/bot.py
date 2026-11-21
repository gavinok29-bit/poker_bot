def decide(s):
    if s.get("type") == "warmup":
        return {"action": "check"}
    c = s["your_cards"]
    good = c[0][0] == c[1][0] or any(x[0] in "AKQJ" for x in c) and all(x[0] in "AKQJT" for x in c)
    if good:
        return {"action": "raise", "amount": s["current_bet"] + s["pot"]} if s["can_check"] else {"action": "call"}
    return {"action": "check"} if s["can_check"] else {"action": "fold"}
