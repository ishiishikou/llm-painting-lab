#!/usr/bin/env python3
"""Refinement for observe-act v7.

Adds a palette-based mid-size mass pass before short observation cycles, forces an
initial scan across all major regions, and gives the iterative painter enough local
work to preserve likeness without returning to tens of thousands of residual marks.
"""
from collections import Counter

import reference_to_brush_process as base
import generate_observe_act_v7 as v7


_original_choose = v7.choose_problem
_original_run = v7.run_observe_act
_visit = {p: Counter() for p in ('light_shadow','face_structure','detail','finish')}
_prep_done = False


def build_palette(im, bg, sub, face, colors_per_region=12):
    p=im.load(); w,h=im.size
    counts={r:Counter() for r in v7.REGIONS}
    for y in range(4,h,6):
        for x in range(4,w,6):
            q=p[x,y]; r=base.region(x,y,q,bg,sub,face)
            if r in counts:
                c=tuple(int(max(0,min(255,round(ch/16)*16))) for ch in q)
                counts[r][c]+=1
    return {r:[c for c,_ in counts[r].most_common(12)] or [(128,128,128)] for r in v7.REGIONS}


v7.build_palette=build_palette


def choose_problem(obs, phase, recent_regions):
    # A painter normally scans every major part before obsessing over one area.
    # Force one visit to each visible region per phase, then return to score-based choice.
    missing=[r for r in v7.REGIONS if _visit[phase][r]==0 and obs[r]['count']>0]
    if missing:
        region=max(missing,key=lambda r:obs[r]['score']*v7.REGION_WEIGHT[r])
        z=obs[region]
        if phase=='light_shadow': intent='value' if z['value_mae']>=z['edge_mae']*.72 else 'edge'
        elif phase=='face_structure': intent='structure' if region=='face' else ('edge' if z['edge_mae']>z['chroma_mae'] else 'color')
        elif phase=='detail': intent='structure' if region=='face' and z['edge_mae']>5 else ('edge' if z['edge_mae']>z['chroma_mae'] else 'color')
        else: intent='edge' if z['edge_mae']>z['chroma_mae']*.75 else 'color'
    else:
        region,intent=_original_choose(obs,phase,recent_regions)
    _visit[phase][region]+=1
    return region,intent


v7.choose_problem=choose_problem


def palette_mass_pass(strokes,current,ref,bg,sub,face,palettes,rng,step,length,width,blur,opacity):
    src=ref.filter(base.ImageFilter.GaussianBlur(blur)) if blur else ref
    p=src.load(); raw=ref.load(); w,h=ref.size
    groups={r:[] for r in v7.REGIONS}
    off=step//2
    for y in range(off,h,step):
        for x in range(off,w,step):
            q=p[x,y]; r=base.region(x,y,q,bg,sub,face)
            if r not in groups: continue
            la,m=base.gradient(raw,x,y,w,h)
            angle=base.direction_field(x,y,r,sub,face,la,m,'light_shadow',rng)
            groups[r].append((x,y,q,angle,m))
    for region in ('face','headwrap','garment','subject'):
        vals=groups[region]
        vals.sort(key=lambda t:(int(t[1]//(step*3)),int(t[0]//(step*3))))
        load=None; load_left=0; load_id=0
        for x,y,q,angle,m in vals:
            if load_left<=0:
                load=v7.nearest_palette(q,palettes[region]); load_left=rng.randint(7,13); load_id+=1
            start=len(strokes)
            base.add_flat(strokes,'light_shadow',x,y,length*(.82 if m>24 else 1),width*(.84 if m>28 else 1),
                          base.hexcolor(load),opacity,angle,rng.randrange(1,2**31-1),region,None)
            s=strokes[-1]
            s['paintRun']='light-shadow-mass'; s['paintLoadId']=f'mass-{step}-{region}-{load_id}'
            s['intent']='value'; s['observationRegion']=region; s['loadedColor']=base.hexcolor(load)
            base.apply_stroke(current,s); load_left-=1


def run_observe_act(strokes,current,ref,bg,sub,face,palettes,phase,cycles,batch,stride,radius,rng,log,start_cycle,load_counter):
    global _prep_done
    if phase=='light_shadow' and not _prep_done:
        # Humans normally build coherent mid-sized value/color masses before resolving detail.
        palette_mass_pass(strokes,current,ref,bg,sub,face,palettes,rng,30,48,16,8,.56)
        palette_mass_pass(strokes,current,ref,bg,sub,face,palettes,rng,20,34,10,4,.62)
        _prep_done=True
    settings={
        'light_shadow':(16,48,9,110),
        'face_structure':(12,42,7,90),
        'detail':(42,64,5,76),
        'finish':(26,48,4,64),
    }
    c,b,s,r=settings.get(phase,(cycles,batch,stride,radius))
    return _original_run(strokes,current,ref,bg,sub,face,palettes,phase,c,b,s,r,rng,log,start_cycle,load_counter)


v7.run_observe_act=run_observe_act

if __name__=='__main__':
    v7.main()
