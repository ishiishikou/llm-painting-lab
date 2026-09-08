#!/usr/bin/env python3
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WEIGHTS={'composition':.10,'silhouette':.18,'light_shadow':.24,'face_structure':.15,'detail':.22,'finish':.11}
ROLE_LABELS={'headwrap':'ターバン','face':'顔','garment':'衣服','subject':'人物','face-plane':'顔の面','underpaint-headwrap':'ターバン下塗り','underpaint-face':'顔の下塗り','underpaint-garment':'衣服下塗り','underpaint-subject':'人物下塗り'}

def parse_hex(s): s=s.lstrip('#'); return tuple(int(s[i:i+2],16) for i in (0,2,4))
def clamp(v,lo=0,hi=255): return max(lo,min(hi,int(round(v))))
def font_path(bold=False):
    cs=['/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    return next((p for p in cs if Path(p).exists()),None)

def draw_line(draw,s,fraction=1.0):
    rgb=parse_hex(s.get('color','#fff')); alpha=clamp(s.get('opacity',1)*255); x1,y1=s['x1'],s['y1']; x2=x1+(s['x2']-x1)*fraction; y2=y1+(s['y2']-y1)*fraction; w=max(1,int(round(s.get('width',1)))); draw.line((x1,y1,x2,y2),fill=(*rgb,alpha),width=w)
    if w>=3:
        r=w/2; draw.ellipse((x1-r,y1-r,x1+r,y1+r),fill=(*rgb,alpha)); draw.ellipse((x2-r,y2-r,x2+r,y2+r),fill=(*rgb,alpha))
    return x2,y2,w

def draw_dry(draw,s,fraction=1.0):
    rgb=parse_hex(s.get('color','#fff')); alpha=clamp(s.get('opacity',1)*255); x1,y1=s['x1'],s['y1']; x2=x1+(s['x2']-x1)*fraction; y2=y1+(s['y2']-y1)*fraction; dx=x2-x1;dy=y2-y1;ln=math.hypot(dx,dy) or 1;nx,ny=-dy/ln,dx/ln; rng=random.Random(int(s.get('seed',1))); strands=max(5,int(s.get('strands',18))); spread=s.get('spread',s.get('width',12)); sw=max(1,int(round(max(.7,s.get('width',8)/strands*.75))))
    for _ in range(strands):
        off=(rng.random()-.5)*spread;j1=(rng.random()-.5)*2;j2=(rng.random()-.5)*2;a=int(alpha*(.28+rng.random()*.65)); draw.line((x1+nx*off+j1,y1+ny*off+j1,x2+nx*off+j2,y2+ny*off+j2),fill=(*rgb,a),width=sw)
    return x2,y2,max(2,s.get('width',8))

def draw_stroke(draw,s,fraction=1.0): return draw_dry(draw,s,fraction) if s.get('brush')=='dryBrush' else draw_line(draw,s,fraction)

def slices(strokes,order):
    d=defaultdict(list)
    for i,s in enumerate(strokes): d[s.get('phase')].append(i)
    return [(p,v[0],v[-1]+1) for p in order if (v:=d.get(p))]

def main():
    ap=argparse.ArgumentParser(description='筆が実際に移動して見える工程別リプレイを生成する'); ap.add_argument('input');ap.add_argument('--output',default='replay.mp4');ap.add_argument('--seconds',type=float,default=90);ap.add_argument('--fps',type=int,default=12);ap.add_argument('--width',type=int,default=640);a=ap.parse_args()
    doc=json.loads(Path(a.input).read_text(encoding='utf-8')); strokes=doc['strokes']; order=[p['id'] for p in doc['phases']]; labels={p['id']:p.get('label',p['id']) for p in doc['phases']}; ss=slices(strokes,order); cw,ch=doc['canvas']['width'],doc['canvas']['height']; bg=parse_hex(doc['canvas'].get('background','#111318'))
    hh=132; ow=a.width; oh=math.ceil((ch+hh)*ow/cw); oh+=oh%2; reg=font_path(False);bold=font_path(True); ftitle=ImageFont.truetype(bold,24) if bold else ImageFont.load_default(); fmain=ImageFont.truetype(reg,21) if reg else ImageFont.load_default(); fsmall=ImageFont.truetype(reg,18) if reg else ImageFont.load_default()
    paint=Image.new('RGB',(cw,ch),bg); pd=ImageDraw.Draw(paint,'RGBA'); writer=imageio.get_writer(a.output,fps=a.fps,codec='libx264',macro_block_size=1,quality=7)
    total_frames=max(1,int(a.seconds*a.fps)); openf=a.fps*2; endf=a.fps*2; usable=max(1,total_frames-openf-endf); ws=[WEIGHTS.get(p,1/len(ss)) for p,_,_ in ss]; sm=sum(ws) or 1; pframes=[max(1,round(usable*w/sm)) for w in ws]; pframes[-1]+=usable-sum(pframes)
    def frame(phase=None,progress=0,note='',active=None):
        page=Image.new('RGB',(cw,ch+hh),'white'); page.paste(paint,(0,hh)); d=ImageDraw.Draw(page); d.text((18,8),'LLM Painting Lab　筆運びリプレイ',fill='black',font=ftitle)
        if phase: d.text((18,42),f'{labels.get(phase,phase)}　工程内 {progress:5.1f}%',fill='black',font=fmain)
        else: d.text((18,42),'地塗り済みキャンバスから開始',fill='black',font=fmain)
        d.text((18,76),note or '太筆 → 明暗 → 顔構造 → 細部の順で描きます',fill=(65,65,65),font=fsmall)
        if active:
            x,y,w=active; yy=y+hh; r=max(5,min(14,w*.38)); d.ellipse((x-r,yy-r,x+r,yy+r),outline=(95,95,95),width=2)
        page=page.resize((ow,oh),Image.Resampling.BILINEAR); writer.append_data(np.asarray(page))
    for _ in range(openf): frame(note='公開されている油彩制作過程を参考に、地塗りから人物を組み立てます')
    current=0
    for (phase,start,end),frames_for in zip(ss,pframes):
        while current<start: draw_stroke(pd,strokes[current],1);current+=1
        count=end-start
        for fi in range(frames_for):
            pos=count*(fi+1)/frames_for; full=min(count,int(pos)); frac=pos-full; target=start+full
            while current<target: draw_stroke(pd,strokes[current],1);current+=1
            active=None; note=''
            if current<end and frac>0:
                temp=paint.copy(); td=ImageDraw.Draw(temp,'RGBA'); active=draw_stroke(td,strokes[current],frac); saved=paint; paint=temp; role=strokes[current].get('role',''); note=f'描画中: {ROLE_LABELS.get(role,role or "筆を運ぶ")}'
                frame(phase,(pos/max(1,count))*100,note,active); paint=saved
            else: frame(phase,(pos/max(1,count))*100,'工程切替' if fi==0 else '')
        while current<end: draw_stroke(pd,strokes[current],1);current+=1
    while current<len(strokes): draw_stroke(pd,strokes[current],1);current+=1
    for _ in range(endf): frame(ss[-1][0],100,'完成')
    writer.close(); print(json.dumps({'output':a.output,'seconds':a.seconds,'fps':a.fps,'strokes':len(strokes),'phase_frames':{p:f for (p,_,_),f in zip(ss,pframes)}},ensure_ascii=False))
if __name__=='__main__': main()
