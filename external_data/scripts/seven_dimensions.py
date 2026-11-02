"""Freeze, evaluate, inspect or formally compare complete seven-dimensional profiles."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'runner'),str(ROOT/'runner/scorers')]
import seven_profile as S


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','evaluate','report','compare'])
    p.add_argument('--lock',type=Path,default=S.LOCK)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--agent-dir',type=Path)
    p.add_argument('--compare',type=Path)
    p.add_argument('--bench')
    p.add_argument('--out',type=Path)
    a = p.parse_args()
    if a.command == 'freeze':
        r = S.freeze(a.lock)
        print(json.dumps({'rule_sha256':r['rule_sha256'],'frozen_at':r['frozen_at'],'path':str(a.lock)}))
        return
    if a.command == 'evaluate':
        if not a.bundle: p.error('--bundle required')
        r = S.evaluate(a.bundle,a.lock)
        print(json.dumps({'task_id':r['task_id'],'scores':{k:v['score'] for k,v in r['dimensions'].items()},'admission':r['admission']},ensure_ascii=False))
        return
    if not a.agent_dir or not a.bench: p.error('--agent-dir and --bench required')
    r = S.aggregate(a.agent_dir,a.bench,lock_path=a.lock)
    if a.command == 'compare':
        if not a.compare: p.error('--compare required')
        other = S.aggregate(a.compare,a.bench,lock_path=a.lock)
        def admitted(report):
            return {v['task_id']:v for v in report['tasks'] if v['admission']['formal_analysis_eligible']}
        x,y = admitted(r),admitted(other)
        common = sorted(set(x)&set(y))
        compatible = [t for t in common if x[t]['rule_sha256']==y[t]['rule_sha256'] and x[t]['judge']==y[t]['judge']
                      and x[t]['input_sha256']['task.json']==y[t]['input_sha256']['task.json']]
        r = {'paired_task_ids':compatible,'n_paired':len(compatible),
             'excluded_or_unpaired_a':r['n_tasks']-len(compatible),'excluded_or_unpaired_b':other['n_tasks']-len(compatible),
             'dimensions':{d:{'a':sum(x[t]['dimensions'][d]['score'] for t in compatible)/len(compatible) if compatible else None,
                              'b':sum(y[t]['dimensions'][d]['score'] for t in compatible)/len(compatible) if compatible else None}
                           for d in S.DIMENSIONS}}
    if a.out:
        with a.out.open('x') as f:json.dump(r,f,indent=2,ensure_ascii=False)
    else:print(json.dumps(r,indent=2,ensure_ascii=False))


if __name__ == '__main__':main()
