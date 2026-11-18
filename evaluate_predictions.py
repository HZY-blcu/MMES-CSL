"""Evaluate strict ID-aligned predictions against the fixed GPT test labels."""
from pathlib import Path
import argparse,json,hashlib
from run_baselines import TASK,TRAITS,read_jsonl,write_json,metrics
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('--five-traits',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    gold=read_jsonl(TASK/'data/test.jsonl');pred=read_jsonl(a.predictions)
    gt={r['essay_id']:r for r in gold};pt={r['essay_id']:r for r in pred}
    if len(pt)!=len(pred):raise ValueError('Duplicate prediction IDs')
    if set(pt)!=set(gt):raise ValueError(f'Expected all {len(gt)} test IDs exactly once; missing={len(set(gt)-set(pt))}, unexpected={len(set(pt)-set(gt))}')
    traits=TRAITS+(['书写'] if a.five_traits else [])
    for row in pred:
        if set(row['scores'])!=set(traits):raise ValueError('Prediction trait set does not match selected evaluation protocol')
        if any(type(v) is not int or not 1<=v<=5 for v in row['scores'].values()):raise ValueError('Only integers 1..5 are valid; missing/invalid outputs must be repaired before evaluation')
    ids=sorted(gt);result={}
    for trait in traits:
        y=np.array([gt[i]['scores'][trait] for i in ids]);q=np.array([pt[i]['scores'][trait] for i in ids]);result[trait]=metrics(y,q)
    qs=[r['qwk'] for r in result.values()]
    out={'reference':'GPT-generated scores','n':len(ids),'traits':result,'macro_qwk':float(np.mean(qs)) if all(x is not None for x in qs) else None,'prediction_sha256':hashlib.sha256(a.predictions.read_bytes()).hexdigest(),'split_sha256':json.loads((TASK/'data/splits.json').read_text(encoding='utf-8'))['split_sha256']}
    if a.output:write_json(a.output,out)
    print(json.dumps(out,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
