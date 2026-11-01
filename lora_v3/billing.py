"""Read-only Fireworks billing snapshot. Never prints the key."""
import httpx, json, os, sys
H={"Authorization":"Bearer "+os.environ["FIREWORKS_API_KEY"]}
r=httpx.get("https://api.fireworks.ai/v1/accounts/alsopgroup/billing/summary",params={"startTime":"2026-10-01T00:00:00Z","endTime":"2026-11-30T00:00:00Z"},headers=H,timeout=60); r.raise_for_status()
out={}
for l in r.json().get("lineItems",[]):
    c,u=l["totalCost"],l["unitAmount"]
    out[f'{l["category"]}|{l.get("groupingValue")}']=dict(qty=l["quantity"],unit=int(u.get("units","0"))+u.get("nanos",0)/1e9,total=int(c.get("units","0"))+c.get("nanos",0)/1e9)
json.dump(out,open(sys.argv[1],"w"),indent=1)
for k,v in out.items():
    if "Training" in k or "glimmer" in k.lower() or "spark" in k.lower() or "GPU" in k: print(k,v)
