"""Generate sample metrics files: python examples/generate.py"""
import json, math, random, os
random.seed(7)
def gen(path, scenario, steps=200, ve=20):
    rows=[]
    for i in range(1,steps+1):
        tr=2.6*math.exp(-i/45)+0.32+random.uniform(-0.05,0.05)
        r={"iter":i,"train_loss":round(tr,4),"lr":round(1e-4*0.5*(1+math.cos(math.pi*i/steps)),8),
           "step_s":round(random.uniform(0.18,0.30),3),"peak_gb":round(7.2+random.uniform(-0.3,0.5),2)}
        if i%ve==0:
            v=2.5*math.exp(-i/50)+0.45 if scenario=="converge" else ((2.3*math.exp(-i/35)+0.5) if i<=90 else 0.95+0.004*(i-90))
            r["val_loss"]=round(v+random.uniform(-0.01,0.01),4)
        rows.append(r)
    open(path,"w").write("\n".join(json.dumps(r) for r in rows)+"\n")
here=os.path.dirname(__file__)
gen(os.path.join(here,"converging.jsonl"),"converge")
gen(os.path.join(here,"overfitting.jsonl"),"overfit")
print("wrote converging.jsonl + overfitting.jsonl")
