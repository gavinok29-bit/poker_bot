def decide(s):
    if s.get("type") == "warmup":
        return {"action": "check"}
    return {"action": "all_in"}
