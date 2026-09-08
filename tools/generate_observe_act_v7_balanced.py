#!/usr/bin/env python3
"""Balanced observe-act v7.

No global corrective mass pass is used after block-in. Every later correction comes
from the same short loop: observe current painting -> choose one region/problem ->
load/mix one color -> make a small spatially connected run -> observe again.
"""
from collections import Counter

import reference_to_brush_process as base
import generate_observe_act_v7 as v7


_visits = {p: Counter() for p in ('light_shadow','face_structure','detail','finish')}


def build_palette(im, bg, sub, face, colors_per_region=24):
    p=im.load(); w,h=im.size
    counts={r:Counter() for r in v7.REGIONS}
    for y in range(4,h,5):
        for x in range(4,w,5):
            q=p[x,y]; r=base.region(x,y,q,bg,sub,face)
            if r in counts:
                c=tuple(int(max(0,min(255,round(ch/8)*8))) for ch in q)
                counts[r][c]+=1
    out={}
    for r in v7.REGIONS:
        common=[c for c,_ in counts[r].most_common(colors_per_region)]
        allc=list(counts[r])
        if allc:
            common += [min(allc,key=base.lum), max(allc,key=base.lum)]
        vals=[]
        for c in common:
            if c not in vals: vals.append(c)
        out[r]=vals[:colors_per_region+2] or [(128,128,128)]
    return out


def nearest_palette(q,palette):
    p=min(palette,key=lambda c:base.dist(q,c))
    # Mix once at the palette for this run. This is intentionally much closer to the
    # observed color than v7-refined, but it is still reused for several marks.
    t=.80
    return tuple(int(round(p[i]*(1-t)+q[i]*t)) for i in range(3))


v7.build_palette=build_palette
v7.nearest_palette=nearest_palette


def intent_for(z,phase,region):
    if phase=='light_shadow':
        if z['value_mae']>=max(5,z['edge_mae']*.72): return 'value'
        return 'color' if z['chroma_mae']>z['edge_mae']*.72 else 'edge'
    if phase=='face_structure':
        if region=='face' and z['edge_mae']>4: return 'structure'
        return 'color' if z['chroma_mae']>z['edge_mae']*.70 else 'edge'
    if phase=='detail':
        if region=='face' and z['edge_mae']>5: return 'structure'
        return 'color' if z['chroma_mae']>z['edge_mae']*.62 else 'edge'
    return 'color' if z['chroma_mae']>z['edge_mae']*.58 else 'edge'


def choose_problem(obs,phase,recent_regions):
    # Scan all major parts once per phase before repeatedly returning to the focal area.
    missing=[r for r in v7.REGIONS if _visits[phase][r]==0 and obs[r]['count']>0]
    if missing:
        r=max(missing,key=lambda x:obs[x]['score']*v7.REGION_WEIGHT[x])
    else:
        ranked=[]
        for r in v7.REGIONS:
            fatigue=.66 if len(recent_regions)>=2 and recent_regions[-1]==recent_regions[-2]==r else 1.0
            # Slightly reduce face dominance after it has received substantially more
            # visits than the rest. Humans periodically leave the focal area to judge whole.
            balance=1/(1+max(0,_visits[phase][r]-min(_visits[phase].values()))*.035)
            ranked.append((obs[r]['score']*v7.REGION_WEIGHT[r]*fatigue*balance,r))
        r=max(ranked)[1]
    _visits[phase][r]+=1
    return r,intent_for(obs[r],phase,r)


v7.choose_problem=choose_problem


def paint_run(strokes,current,ref,bg,sub,face,palettes,phase,region,intent,
              candidates,rng,cycle,run_id,load_counter):
    if not candidates: return 0,load_counter
    made=0; i=0
    while i<len(candidates):
        # Smaller connected load than earlier v7: enough to preserve painter state,
        # small enough that one mixed color still belongs to the same local passage.
        block=candidates[i:i+rng.randint(5,9)]
        load=nearest_palette(block[0][3],palettes[region]); color=base.hexcolor(load)
        load_counter+=1; start=len(strokes)
        for score,x,y,q,local_angle,m in block:
            angle=base.direction_field(x,y,region,sub,face,local_angle,m,phase,rng)
            if phase=='light_shadow':
                if made%6==5:
                    base.add_mixer(strokes,phase,x,y,18 if region=='face' else 24,7 if region=='face' else 10,
                                   color,.44,angle,f'mix-{region}',None,.12,.50,.32)
                else:
                    base.add_flat(strokes,phase,x,y,26 if region=='face' else 34,9 if region=='face' else 12,
                                  color,.48,angle,rng.randrange(1,2**31-1),region,None)
            elif phase=='face_structure':
                if region=='face' and made%7!=6:
                    base.add_variable(strokes,phase,x,y,8,4.0,1.3,color,.78,angle,'face-plane',None)
                elif made%4==3:
                    base.add_mixer(strokes,phase,x,y,13,6.5,color,.40,angle,f'mix-{region}',None,.10,.48,.28)
                else:
                    base.add_variable(strokes,phase,x,y,8.5,3.6,1.2,color,.72,angle,region,None)
            elif phase=='detail':
                if made%5:
                    base.add_variable(strokes,phase,x,y,5.8 if region=='face' else 6.8,2.4,.65,color,.88,angle,region,None)
                else:
                    base.add_line(strokes,phase,x,y,4.2 if region=='face' else 5.0,1.3 if region=='face' else 1.55,color,.90,angle,region)
            else:
                if intent=='color' and made%6==5:
                    base.add_mixer(strokes,phase,x,y,11,6,color,.30,angle,f'mix-{region}',None,.08,.44,.24)
                elif made%4:
                    base.add_variable(strokes,phase,x,y,4.0 if region=='face' else 4.8,1.8,.5,color,.84,angle,region,None)
                else:
                    base.add_line(strokes,phase,x,y,2.9 if region=='face' else 3.4,1.0 if region=='face' else 1.2,color,.88,angle,region)
            made+=1
        v7.mark_new(strokes,start,cycle,run_id,load_counter,intent,region,color)
        v7.apply_new(current,strokes,start)
        i+=len(block)
    return made,load_counter


v7.paint_run=paint_run

_original_run=v7.run_observe_act


def run_observe_act(strokes,current,ref,bg,sub,face,palettes,phase,cycles,batch,stride,radius,rng,log,start_cycle,load_counter):
    # Keep all corrections local and interleaved with observation. Increase the number
    # of sessions rather than reverting to a single thousands-of-strokes residual pass.
    settings={
        'light_shadow':(24,62,8,112),
        'face_structure':(18,54,7,90),
        'detail':(76,72,4,78),
        'finish':(48,58,3,66),
    }
    c,b,s,r=settings[phase]
    return _original_run(strokes,current,ref,bg,sub,face,palettes,phase,c,b,s,r,rng,log,start_cycle,load_counter)


v7.run_observe_act=run_observe_act

if __name__=='__main__':
    v7.main()
