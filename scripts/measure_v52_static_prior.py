"""Measure the unlearned DLS prior on saved simulation samples."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]


def main(args):
    import torch
    from homie_pushcart_v52.observation import frame_layout
    from homie_pushcart_v52.fixed_grasp import palm_offsets_from_handle
    from train_pushcart_v52_transfer import score
    torch.set_num_threads(1)
    field=next(f for f in frame_layout() if f['name']=='static_hand_wrench')
    offsets=torch.tensor(palm_offsets_from_handle(),dtype=torch.float32)
    prior=lambda x:x[:,-136+field['start']:][:,:12]
    rows=[]
    for path in sorted(args.captures.glob('*_samples.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)
        rows.append(dict(capture=str(path),engine=data['report']['engine'],drive=data['report']['upper_drive'],
                         scores=score(prior,data['x'],data['y'],offsets)))
    args.output.write_text(json.dumps(dict(results=rows,note='Raw DLS prior only; no fitting or gravity/inertia correction.'),indent=2),encoding='utf-8')
    for r in rows:
        print(r['engine'],r['drive'],r['scores']['handle']['rmse'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
