import random
def decide(s):
    if s.get("type") == "warmup":
        return {"action": "check"}
    if random.random() < 0.6:
        return {"action": "raise", "amount": s["current_bet"] + max(s["pot"], s["blinds"]["bb"] * 2)}
    return {"action": "call"} if s["amount_owed"] else {"action": "check"}
