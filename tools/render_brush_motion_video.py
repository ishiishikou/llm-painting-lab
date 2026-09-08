#!/usr/bin/env python3
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WEIGHTS={'composition':.10,'silhouette':.18,'light_shadow':.24,'face_structure':.15,'detail':.22,'finish':.11}
ROLE_LABELS={'headwrap':'ターバン','face':'顔','garment':'衣服','subject':'人物','face-plane':'顔の面','underpaint-headwrap':'ターバン下塗り','underpaint-face':'顔の下塗り','underpaint-garment':'衣服下塗り','underpaint-subject':'人物下塗り','mix-face':'顔の混色','mix-headwrap':'ターバンの混色','smudge-face':'顔のぼかし','smudge-headwrap':'ターバンのぼかし','glaze-face':'顔のグレーズ','glaze-headwrap':'ターバンのグレーズ','glaze-garment':'衣服のグレーズ'}

def parse_hex(s): s=s.lstrip('#'); return tuple(int(s[i:i+2],16) for i in (0,2,4))
def clamp(v,lo=0,hi=255): return max(lo,min(hi,int(round(v))))
def blend(a,b,t): return tuple(a[i]*(1-t)+b[i]*t for i in range(3))
def rgb(v): return tuple(clamp(x) for x in v)
def font_path(bold=False):
    cs=['/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    return next((p for p in cs if Path(p).exists()),None)

def clipped(s,x,y):
    b=s.get('clipBox'); return True if not b else b[0]<=x<=b[2] and b[1]<=y<=b[3]

def draw_variable(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');c=parse_hex(s.get('color','#fff'));a=clamp(s.get('opacity',1)*255);x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;ln=math.hypot(x2-x1,y2-y1);n=max(2,int(ln/2.5));prev=None
    for i in range(n+1):
        t=i/n;x=x1+(x2-x1)*t;y=y1+(y2-y1)*t
        if not clipped(s,x,y):continue
        w=s.get('widthStart',s.get('width',1))*(1-t)+s.get('widthEnd',s.get('width',1))*t;w=max(.7,w);r=w/2
        if prev:
            px,py,pw=prev;d.line((px,py,x,y),fill=(*c,a),width=max(1,int(round((pw+w)/2))))
        d.ellipse((x-r,y-r,x+r,y+r),fill=(*c,a));prev=(x,y,w)
    return x2,y2,max(s.get('widthStart',1),s.get('widthEnd',1))

def draw_dry(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');c=parse_hex(s.get('color','#fff'));alpha=clamp(s.get('opacity',1)*255);x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;dx=x2-x1;dy=y2-y1;ln=math.hypot(dx,dy) or 1;nx,ny=-dy/ln,dx/ln;rng=random.Random(int(s.get('seed',1)));strands=max(5,int(s.get('strands',10)));spread=s.get('spread',s.get('width',12));sw=max(1,int(round(max(.7,s.get('width',8)/strands*.75))))
    for _ in range(strands):
        off=(rng.random()-.5)*spread;j1=(rng.random()-.5)*2;j2=(rng.random()-.5)*2;xa=x1+nx*off+j1;ya=y1+ny*off+j1;xb=x2+nx*off+j2;yb=y2+ny*off+j2
        if s.get('clipBox'):
            b=s['clipBox'];xa=max(b[0],min(b[2],xa));xb=max(b[0],min(b[2],xb));ya=max(b[1],min(b[3],ya));yb=max(b[1],min(b[3],yb))
        aa=clamp(alpha*(.28+rng.random()*.65));d.line((xa,ya,xb,yb),fill=(*c,aa),width=sw)
    return x2,y2,max(2,s.get('width',8))

def draw_mixer(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');paint=tuple(float(v) for v in parse_hex(s.get('color','#fff')));x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;ln=math.hypot(x2-x1,y2-y1);n=max(2,int(ln/3));pickup=float(s.get('pickup',.18));mix=float(s.get('mix',.58));deposit=float(s.get('deposit',.42));alpha=float(s.get('opacity',1))*deposit
    for i in range(n+1):
        t=i/n;x=x1+(x2-x1)*t;y=y1+(y2-y1)*t
        if not clipped(s,x,y):continue
        xi=max(0,min(im.width-1,int(round(x))));yi=max(0,min(im.height-1,int(round(y))));under=im.getpixel((xi,yi));paint=blend(paint,under,pickup);out=blend(under,paint,mix);r=max(1,s.get('width',8)/2);d.ellipse((x-r,y-r,x+r,y+r),fill=(*rgb(out),clamp(alpha*255)))
    return x2,y2,s.get('width',8)

def draw_smudge(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;xi=max(0,min(im.width-1,int(round(x1))));yi=max(0,min(im.height-1,int(round(y1))));carry=tuple(float(v) for v in im.getpixel((xi,yi)));ln=math.hypot(x2-x1,y2-y1);n=max(2,int(ln/3));strength=float(s.get('strength',.30))*float(s.get('opacity',1));pickup=float(s.get('pickup',.10))
    for i in range(1,n+1):
        t=i/n;x=x1+(x2-x1)*t;y=y1+(y2-y1)*t
        if not clipped(s,x,y):continue
        xi=max(0,min(im.width-1,int(round(x))));yi=max(0,min(im.height-1,int(round(y))));under=im.getpixel((xi,yi));r=max(1,s.get('width',8)/2);d.ellipse((x-r,y-r,x+r,y+r),fill=(*rgb(carry),clamp(strength*255)));carry=blend(carry,under,pickup)
    return x2,y2,s.get('width',8)

def draw_glaze(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');c=parse_hex(s.get('color','#fff'));x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;ln=math.hypot(x2-x1,y2-y1);n=max(2,int(ln/3));a=clamp(float(s.get('opacity',.1))*255);r=max(1,s.get('width',12)/2)
    for i in range(n+1):
        t=i/n;x=x1+(x2-x1)*t;y=y1+(y2-y1)*t
        if clipped(s,x,y):d.ellipse((x-r,y-r,x+r,y+r),fill=(*c,a))
    return x2,y2,s.get('width',12)

def draw_stroke(im,s,fraction=1.0):
    b=s.get('brush','line')
    if b=='dryBrush':return draw_dry(im,s,fraction)
    if b=='mixerBrush':return draw_mixer(im,s,fraction)
    if b=='smudgeBrush':return draw_smudge(im,s,fraction)
    if b=='glaze':return draw_glaze(im,s,fraction)
    t=dict(s);t.setdefault('widthStart',s.get('width',1));t.setdefault('widthEnd',s.get('width',1));return draw_variable(im,t,fraction)

def slices(strokes,order):
    d=defaultdict(list)
    for i,s in enumerate(strokes):d[s.get('phase')].append(i)
    return [(p,v[0],v[-1]+1) for p in order if (v:=d.get(p))]

def main():
    ap=argparse.ArgumentParser(description='筆・混色・ぼかしが実際に移動して見える工程別リプレイを生成する');ap.add_argument('input');ap.add_argument('--output',default='replay.mp4');ap.add_argument('--seconds',type=float,default=90);ap.add_argument('--fps',type=int,default=12);ap.add_argument('--width',type=int,default=640);a=ap.parse_args()
    doc=json.loads(Path(a.input).read_text(encoding='utf-8'));strokes=doc['strokes'];order=[p['id'] for p in doc['phases']];labels={p['id']:p.get('label',p['id']) for p in doc['phases']};ss=slices(strokes,order);cw,ch=doc['canvas']['width'],doc['canvas']['height'];bg=parse_hex(doc['canvas'].get('background','#111318'))
    hh=132;ow=a.width;oh=math.ceil((ch+hh)*ow/cw);oh+=oh%2;reg=font_path(False);bold=font_path(True);ftitle=ImageFont.truetype(bold,24) if bold else ImageFont.load_default();fmain=ImageFont.truetype(reg,21) if reg else ImageFont.load_default();fsmall=ImageFont.truetype(reg,18) if reg else ImageFont.load_default();paint=Image.new('RGB',(cw,ch),bg);writer=imageio.get_writer(a.output,fps=a.fps,codec='libx264',macro_block_size=1,quality=7)
    total_frames=max(1,int(a.seconds*a.fps));openf=a.fps*2;endf=a.fps*2;usable=max(1,total_frames-openf-endf);ws=[WEIGHTS.get(p,1/len(ss)) for p,_,_ in ss];sm=sum(ws) or 1;pframes=[max(1,round(usable*w/sm)) for w in ws];pframes[-1]+=usable-sum(pframes)
    def frame(phase=None,progress=0,note='',active=None):
        page=Image.new('RGB',(cw,ch+hh),'white');page.paste(paint,(0,hh));d=ImageDraw.Draw(page);d.text((18,8),'LLM Painting Lab　描画道具リプレイ',fill='black',font=ftitle)
        if phase:d.text((18,42),f'{labels.get(phase,phase)}　工程内 {progress:5.1f}%',fill='black',font=fmain)
        else:d.text((18,42),'地塗り済みキャンバスから開始',fill='black',font=fmain)
        d.text((18,76),note or '太筆 → 混色/ぼかし → 顔構造 → 細部 → グレーズ',fill=(65,65,65),font=fsmall)
        if active:
            x,y,w=active;yy=y+hh;r=max(5,min(14,w*.38));d.ellipse((x-r,yy-r,x+r,yy+r),outline=(95,95,95),width=2)
        page=page.resize((ow,oh),Image.Resampling.BILINEAR);writer.append_data(np.asarray(page))
    for _ in range(openf):frame(note='地塗りから始め、必要な場所だけ混色・ぼかし・グレーズを使います')
    current=0
    for (phase,start,end),frames_for in zip(ss,pframes):
        while current<start:draw_stroke(paint,strokes[current],1);current+=1
        count=end-start
        for fi in range(frames_for):
            pos=count*(fi+1)/frames_for;full=min(count,int(pos));frac=pos-full;target=start+full
            while current<target:draw_stroke(paint,strokes[current],1);current+=1
            if current<end and frac>0:
                temp=paint.copy();active=draw_stroke(temp,strokes[current],frac);saved=paint;paint=temp;role=strokes[current].get('role','');brush=strokes[current].get('brush','line');note=f'{brush}: {ROLE_LABELS.get(role,role or "描画")}' ;frame(phase,(pos/max(1,count))*100,note,active);paint=saved
            else:frame(phase,(pos/max(1,count))*100,'工程切替' if fi==0 else '')
        while current<end:draw_stroke(paint,strokes[current],1);current+=1
    while current<len(strokes):draw_stroke(paint,strokes[current],1);current+=1
    for _ in range(endf):frame(ss[-1][0],100,'完成')
    writer.close();print(json.dumps({'output':a.output,'seconds':a.seconds,'fps':a.fps,'strokes':len(strokes),'phase_frames':{p:f for (p,_,_),f in zip(ss,pframes)}},ensure_ascii=False))
if __name__=='__main__':main()
