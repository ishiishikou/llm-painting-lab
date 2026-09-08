#!/usr/bin/env python3
"""Production candidate for v7: proven broad block-in + true local observe/act loops.

A painter does not necessarily stop after every broad stroke during the first block-in.
They establish the large masses, step back, then repeatedly work one local passage and
look again. This generator follows that pattern while removing the previous global
thousands-of-residual-marks detail/finish passes.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

import reference_to_brush_process as base
import generate_artifact_cleanup_v6 as cleanup  # installs v5 directions + v6 organic face treatment

REGIONS=('face','headwrap','garment','subject')
WEIGHT={'face':1.45,'headwrap':1.12,'garment':.90,'subject':.74}
VISITS={p:Counter() for p in ('light_shadow','face_structure','detail','finish')}


def build_palette(im,bg,sub,face,n=32):
    p=im.load(); w,h=im.size; counts={r:Counter() for r in REGIONS}
    for y in range(3,h,4):
        for x in range(3,w,4):
            q=p[x,y]; r=base.region(x,y,q,bg,sub,face)
            if r in counts:
                c=tuple(int(max(0,min(255,round(ch/6)*6))) for ch in q)
                counts[r][c]+=1
    out={}
    for r in REGIONS:
        vals=[c for c,_ in counts[r].most_common(n)]
        allc=list(counts[r])
        if allc: vals += [min(allc,key=base.lum),max(allc,key=base.lum)]
        uniq=[]
        for c in vals:
            if c not in uniq: uniq.append(c)
        out[r]=uniq[:n+2] or [(128,128,128)]
    return out


def mix_load(target,palette):
    pile=min(palette,key=lambda c:base.dist(target,c))
    t=.92
    return tuple(int(round(pile[i]*(1-t)+target[i]*t)) for i in range(3))


def observe(ref,current,bg,sub,face,step=9):
    rp,cp=ref.load(),current.load(); w,h=ref.size
    z={r:{'n':0,'rgb':0.,'value':0.,'edge':0.} for r in REGIONS}
    for y in range(2,h-2,step):
        for x in range(2,w-2,step):
            q=rp[x,y]; r=base.region(x,y,q,bg,sub,face)
            if r not in z: continue
            c=cp[x,y]; _,re=base.gradient(rp,x,y,w,h); _,ce=base.gradient(cp,x,y,w,h)
            d=math.sqrt(sum((q[i]-c[i])**2 for i in range(3))/3)
            v=abs(base.lum(q)-base.lum(c)); e=abs(re-ce)
            a=z[r]; a['n']+=1; a['rgb']+=d; a['value']+=v; a['edge']+=e
    out={}
    for r,a in z.items():
        n=max(1,a['n']); rgb=a['rgb']/n; value=a['value']/n; edge=a['edge']/n
        chroma=max(0,rgb-value*.58)
        out[r]={'count':a['n'],'rgb_mae':round(rgb,3),'value_mae':round(value,3),
                'edge_mae':round(edge,3),'chroma_mae':round(chroma,3),
                'score':round(rgb+edge*.58,3)}
    return out


def intent(z,phase,region):
    if phase=='light_shadow':
        if z['value_mae']>z['edge_mae']*.70: return 'value'
        return 'color' if z['chroma_mae']>z['edge_mae']*.72 else 'edge'
    if phase=='face_structure':
        if region=='face': return 'structure'
        return 'color' if z['chroma_mae']>z['edge_mae']*.70 else 'edge'
    if phase=='detail':
        if region=='face' and z['edge_mae']>4.5: return 'structure'
        return 'color' if z['chroma_mae']>z['edge_mae']*.60 else 'edge'
    return 'color' if z['chroma_mae']>z['edge_mae']*.56 else 'edge'


def choose(obs,phase,recent):
    missing=[r for r in REGIONS if VISITS[phase][r]==0 and obs[r]['count']]
    if missing:
        r=max(missing,key=lambda x:obs[x]['score']*WEIGHT[x])
    else:
        low=min(VISITS[phase].values()) if VISITS[phase] else 0
        ranked=[]
        for r in REGIONS:
            repeat=.64 if len(recent)>=2 and recent[-1]==recent[-2]==r else 1
            balance=1/(1+max(0,VISITS[phase][r]-low)*.025)
            ranked.append((obs[r]['score']*WEIGHT[r]*repeat*balance,r))
        r=max(ranked)[1]
    VISITS[phase][r]+=1
    return r,intent(obs[r],phase,r)


def candidates(ref,current,bg,sub,face,region,intent_name,rng,count,stride,radius):
    rp,cp=ref.load(),current.load(); w,h=ref.size; arr=[]
    ox,oy=rng.randrange(stride),rng.randrange(stride)
    for y in range(2+oy,h-2,stride):
        for x in range(2+ox,w-2,stride):
            q=rp[x,y]
            if base.region(x,y,q,bg,sub,face)!=region: continue
            c=cp[x,y]; la,m=base.gradient(rp,x,y,w,h); _,cm=base.gradient(cp,x,y,w,h)
            rgb=math.sqrt(sum((q[i]-c[i])**2 for i in range(3))/3)
            value=abs(base.lum(q)-base.lum(c)); edge=abs(m-cm)
            if intent_name=='value': score=value*1.15+rgb*.40
            elif intent_name in {'edge','structure'}: score=edge*1.08+rgb*.50
            else: score=rgb*1.08+value*.18
            if score>2.5: arr.append((score,x,y,q,la,m))
    if not arr: return []
    arr.sort(reverse=True); seed=arr[0]; sx,sy,sq=seed[1],seed[2],seed[3]
    # A physical brush load should be used on a spatially and chromatically coherent passage.
    local=[v for v in arr if (v[1]-sx)**2+(v[2]-sy)**2<=radius*radius and base.dist(v[3],sq)<=52]
    local.sort(reverse=True)
    return local[:count]


def annotate(strokes,start,cycle,run,load_id,intent_name,region,color):
    for s in strokes[start:]:
        s['decisionCycle']=cycle; s['paintRun']=run; s['paintLoadId']=load_id
        s['intent']=intent_name; s['observationRegion']=region; s['loadedColor']=color


def apply_new(current,strokes,start):
    for s in strokes[start:]: base.apply_stroke(current,s)


def paint(strokes,current,ref,bg,sub,face,palettes,phase,region,intent_name,pts,rng,cycle,run,load_counter):
    made=0; i=0
    while i<len(pts):
        block=pts[i:i+rng.randint(4,7)]
        color=base.hexcolor(mix_load(block[0][3],palettes[region])); load_counter+=1; start=len(strokes)
        for score,x,y,q,la,m in block:
            a=base.direction_field(x,y,region,sub,face,la,m,phase,rng)
            if phase=='light_shadow':
                if made%7==6:
                    base.add_mixer(strokes,phase,x,y,16 if region=='face' else 22,7 if region=='face' else 9,
                                   color,.40,a,f'mix-{region}',None,.10,.48,.28)
                else:
                    base.add_flat(strokes,phase,x,y,24 if region=='face' else 32,8.5 if region=='face' else 11,
                                  color,.44,a,rng.randrange(1,2**31-1),region,None)
            elif phase=='face_structure':
                if region=='face' and made%8!=7:
                    base.add_variable(strokes,phase,x,y,7.5,3.8,1.15,color,.82,a,'face-plane',None)
                elif made%4==3:
                    base.add_mixer(strokes,phase,x,y,12,6,color,.36,a,f'mix-{region}',None,.08,.45,.25)
                else:
                    base.add_variable(strokes,phase,x,y,8,3.4,1.1,color,.78,a,region,None)
            elif phase=='detail':
                if made%5:
                    base.add_variable(strokes,phase,x,y,5.4 if region=='face' else 6.3,2.2,.55,color,.92,a,region,None)
                else:
                    base.add_line(strokes,phase,x,y,4.0 if region=='face' else 4.7,1.22 if region=='face' else 1.48,color,.94,a,region)
            else:
                if intent_name=='color' and made%7==6:
                    base.add_mixer(strokes,phase,x,y,10,5.5,color,.26,a,f'mix-{region}',None,.07,.42,.22)
                elif made%4:
                    base.add_variable(strokes,phase,x,y,3.7 if region=='face' else 4.4,1.65,.45,color,.91,a,region,None)
                else:
                    base.add_line(strokes,phase,x,y,2.7 if region=='face' else 3.2,.95 if region=='face' else 1.15,color,.94,a,region)
            made+=1
        annotate(strokes,start,cycle,run,load_counter,intent_name,region,color); apply_new(current,strokes,start)
        i+=len(block)
    return made,load_counter


def loop(strokes,current,ref,bg,sub,face,palettes,phase,cycles,batch,stride,radius,rng,log,cycle,load_counter):
    recent=[]
    for _ in range(cycles):
        before=observe(ref,current,bg,sub,face); region,intent_name=choose(before,phase,recent)
        pts=candidates(ref,current,bg,sub,face,region,intent_name,rng,batch,stride,radius)
        run=f'{phase}-{cycle:03d}-{region}'
        made,load_counter=paint(strokes,current,ref,bg,sub,face,palettes,phase,region,intent_name,pts,rng,cycle,run,load_counter)
        after=observe(ref,current,bg,sub,face)
        log.append({'cycle':cycle,'phase':phase,'region':region,'intent':intent_name,'stroke_count':made,
                    'before_score':before[region]['score'],'after_score':after[region]['score'],
                    'global_before':round(sum(before[r]['score'] for r in REGIONS),3),
                    'global_after':round(sum(after[r]['score'] for r in REGIONS),3)})
        recent.append(region); recent=recent[-3:]; cycle+=1
    return cycle,load_counter


def artifact_checks(strokes,face):
    fb=[round(v,2) for v in face]; fs=[s for s in strokes if cleanup.face_role(s.get('role',''))]
    return {'face_role_strokes':len(fs),'face_role_hard_clip_count':sum(1 for s in fs if s.get('clipBox')),
            'exact_face_bbox_clip_count':sum(1 for s in strokes if s.get('clipBox')==fb)}


def main():
    ap=argparse.ArgumentParser(description='大面積ブロックイン後に観察→局所作業を反復するv7')
    ap.add_argument('input'); ap.add_argument('--output',default='strokes.generated.json'); ap.add_argument('--preview',default='preview.png')
    ap.add_argument('--metrics',default='metrics.json'); ap.add_argument('--observation-log',default='observation-log.json'); ap.add_argument('--checkpoint-dir')
    ap.add_argument('--width',type=int,default=864); ap.add_argument('--height',type=int,default=1024); ap.add_argument('--seed',type=int,default=20260908)
    a=ap.parse_args(); rng=random.Random(a.seed)
    ref=base.crop_resize(Image.open(a.input),a.width,a.height); bg_rgb=base.border_bg(ref); bg=base.hexcolor(bg_rgb)
    sub=base.infer_subject(ref,bg_rgb); face=base.infer_face(ref,sub); palettes=build_palette(ref,bg_rgb,sub,face)
    strokes=[]; cps=[]; decisions=[]; cdir=Path(a.checkpoint_dir) if a.checkpoint_dir else None
    if cdir:cdir.mkdir(parents=True,exist_ok=True)
    def checkpoint(stage,name,current):
        m=base.metrics(ref,current); cps.append({'stage':stage,'stroke_count':len(strokes),**m})
        if cdir:current.save(cdir/name)

    base.composition(strokes,sub,face); current=base.render(strokes,a.width,a.height,bg); checkpoint('composition','01-composition.png',current)
    base.broad_pass(strokes,ref,'silhouette',bg_rgb,sub,face,rng,52,105,34,22,.58,True,('garment','headwrap','face','subject'))
    base.broad_pass(strokes,ref,'silhouette',bg_rgb,sub,face,rng,38,82,27,15,.72,False,('headwrap','face','garment','subject'))
    current=base.render(strokes,a.width,a.height,bg); checkpoint('silhouette','02-silhouette.png',current)

    # Planned mid-size block-in: establish broad value/color masses first, then step back.
    base.broad_pass(strokes,ref,'light_shadow',bg_rgb,sub,face,rng,24,52,17,9,.70,False,('face','headwrap','garment','subject'))
    base.broad_pass(strokes,ref,'light_shadow',bg_rgb,sub,face,rng,16,34,10,4,.76,False,('face','headwrap','garment','subject'))
    current=base.render(strokes,a.width,a.height,bg)
    cycle=1; loads=0
    cycle,loads=loop(strokes,current,ref,bg_rgb,sub,face,palettes,'light_shadow',8,46,7,68,rng,decisions,cycle,loads)
    checkpoint('light_shadow','03-light-shadow.png',current)

    # Construction pass, followed by local re-evaluation instead of immediately proceeding.
    start=len(strokes); base.face_structure(strokes,ref,sub,face,rng); apply_new(current,strokes,start)
    cycle,loads=loop(strokes,current,ref,bg_rgb,sub,face,palettes,'face_structure',10,44,6,58,rng,decisions,cycle,loads)
    checkpoint('face_structure','04-face-structure.png',current)

    cycle,loads=loop(strokes,current,ref,bg_rgb,sub,face,palettes,'detail',130,76,3,42,rng,decisions,cycle,loads)
    checkpoint('detail','05-detail.png',current)
    start=len(strokes); base.glaze_pass(strokes,ref,bg_rgb,sub,face,rng); apply_new(current,strokes,start)
    cycle,loads=loop(strokes,current,ref,bg_rgb,sub,face,palettes,'finish',90,64,2,34,rng,decisions,cycle,loads)
    checkpoint('finish','06-finish.png',current); current.save(a.preview)

    for i,s in enumerate(strokes,1):s['id']=i
    direction=base.direction_stats(strokes); art=artifact_checks(strokes,face)
    used=[s.get('paintLoadId') for s in strokes if s.get('paintLoadId')]
    meta={'slug':'observe-act-v7','title':'大面積ブロックイン後に観察して局所作業を反復する描画工程','seed':a.seed,
          'source_mode':'reference-guided-observe-act-v7','stroke_count':len(strokes),'subject_bbox':[round(v,2) for v in sub],
          'face_bbox':[round(v,2) for v in face],'phase_order':[p[0] for p in base.PHASES],'background_policy':'toned-ground-no-progress',
          'tool_policy':'docs/painting-tool-rules.md','evaluation_policy':'docs/painting-evaluation-rules.md','direction_policy':'region-form-following-v5',
          'artifact_policy':'no-hard-rectangular-face-mask-v6','observe_act_policy':'broad-block-in-then-observe-local-run-reobserve-v7',
          'palette_policy':'persistent-local-mixed-load-v7','palette':{r:[base.hexcolor(c) for c in palettes[r]] for r in REGIONS},
          'decision_cycles':len(decisions),'paint_load_count':len(set(used)),'direction_stats':direction,'artifact_checks':art,'quality_metrics':cps[-1]}
    data={'metadata':meta,'canvas':{'width':a.width,'height':a.height,'background':bg},'phases':[{'id':i,'label':l} for i,l in base.PHASES],'strokes':strokes}
    Path(a.output).write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
    Path(a.observation_log).write_text(json.dumps({'decisions':decisions},ensure_ascii=False,indent=2),encoding='utf-8')
    Path(a.metrics).write_text(json.dumps({'checkpoints':cps,'direction_stats':direction,'artifact_checks':art,'decision_summary':{
        'cycles':len(decisions),'regions':dict(Counter(d['region'] for d in decisions)),'intents':dict(Counter(d['intent'] for d in decisions)),
        'paint_loads':len(set(used))}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'strokes':len(strokes),'phase_counts':{p:sum(1 for s in strokes if s['phase']==p) for p,_ in base.PHASES},
        'brush_counts':dict(Counter(s['brush'] for s in strokes)),'cycles':len(decisions),'paint_loads':len(set(used)),
        'artifact_checks':art,'quality':cps[-1]},ensure_ascii=False))

if __name__=='__main__':main()
