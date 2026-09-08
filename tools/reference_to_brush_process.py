#!/usr/bin/env python3
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

PHASES = [
    ('composition','1. 構図'),
    ('silhouette','2. 大きな形'),
    ('light_shadow','3. 明暗の面'),
    ('face_structure','4. 顔構造'),
    ('detail','5. 細部'),
    ('finish','6. 仕上げ'),
]

def clamp(v, lo=0, hi=255): return max(lo, min(hi, v))
def hexcolor(rgb): return '#%02x%02x%02x' % tuple(int(clamp(v)) for v in rgb)
def parse_hex(s):
    s=s.lstrip('#'); return tuple(int(s[i:i+2],16) for i in (0,2,4))
def lum(p): return .2126*p[0]+.7152*p[1]+.0722*p[2]
def dist(a,b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def crop_resize(im,w,h):
    im=im.convert('RGB'); sr=im.width/im.height; tr=w/h
    if sr>tr:
        nw=int(im.height*tr); left=(im.width-nw)//2; im=im.crop((left,0,left+nw,im.height))
    else:
        nh=int(im.width/tr); top=(im.height-nh)//2; im=im.crop((0,top,im.width,top+nh))
    return im.resize((w,h),Image.Resampling.LANCZOS)

def gradient(pix,x,y,w,h):
    x0,x1=max(0,x-1),min(w-1,x+1); y0,y1=max(0,y-1),min(h-1,y+1)
    gx=lum(pix[x1,y])-lum(pix[x0,y]); gy=lum(pix[x,y1])-lum(pix[x,y0])
    m=math.hypot(gx,gy); return (math.atan2(gy,gx)+math.pi/2 if m>1.5 else 0.0),m

def border_bg(im):
    w,h=im.size; p=im.load(); pts=[]; s=max(4,min(w,h)//100)
    for x in range(0,w,s): pts.extend([p[x,0],p[x,h-1]])
    for y in range(0,h,s): pts.extend([p[0,y],p[w-1,y]])
    ch=[sorted(q[i] for q in pts) for i in range(3)]
    return tuple(c[len(c)//2] for c in ch)

def infer_subject(im,bg):
    w,h=im.size; p=im.load(); xs=[]; ys=[]; bl=lum(bg)
    for y in range(2,h,4):
        for x in range(2,w,4):
            q=p[x,y]
            if dist(q,bg)>45 or lum(q)>bl+24: xs.append(x); ys.append(y)
    if not xs: return (w*.18,h*.08,w*.88,h*.96)
    xs.sort(); ys.sort(); a=int(len(xs)*.015); b=int(len(xs)*.985)-1
    return xs[a],ys[a],xs[b],ys[b]

def skin(p):
    r,g,b=p; return r>88 and g>58 and b>38 and r>g*1.01 and g>b*.90 and max(p)-min(p)>10

def infer_face(im,sub):
    w,h=im.size; p=im.load(); sx0,sy0,sx1,sy1=sub; xs=[];ys=[]
    y1=min(h,int(sy0+(sy1-sy0)*.67))
    for y in range(max(0,int(sy0)),y1,3):
        for x in range(max(0,int(sx0)),min(w,int(sx1)),3):
            if skin(p[x,y]): xs.append(x);ys.append(y)
    if len(xs)<100: return sx0+(sx1-sx0)*.18,sy0+(sy1-sy0)*.22,sx0+(sx1-sx0)*.67,sy0+(sy1-sy0)*.62
    xs.sort();ys.sort()
    def q(a,f): return a[min(len(a)-1,max(0,int(len(a)*f)))]
    return q(xs,.08),q(ys,.08),q(xs,.92),q(ys,.92)

def inside(box,x,y,pad=0):
    x0,y0,x1,y1=box; return x0-pad<=x<=x1+pad and y0-pad<=y<=y1+pad

def subject_pixel(p,bg): return dist(p,bg)>42 or lum(p)>lum(bg)+22

def region(x,y,p,bg,sub,face):
    sx0,sy0,sx1,sy1=sub; fx0,fy0,fx1,fy1=face; sh=sy1-sy0
    if not subject_pixel(p,bg): return 'background'
    if inside(face,x,y,max(10,(fx1-fx0)*.08)): return 'face'
    if y<sy0+sh*.48: return 'headwrap'
    if y>fy1-(fy1-fy0)*.05: return 'garment'
    return 'subject'

def add_line(strokes,phase,x,y,length,width,color,opacity,angle=0,role=''):
    dx=math.cos(angle)*length/2; dy=math.sin(angle)*length/2
    strokes.append({'phase':phase,'brush':'line','x1':round(x-dx,2),'y1':round(y-dy,2),'x2':round(x+dx,2),'y2':round(y+dy,2),'width':round(width,2),'color':color,'opacity':round(opacity,3),'lineCap':'round','role':role})

def add_dry(strokes,phase,x,y,length,width,color,opacity,angle,seed,role='',strands=18):
    dx=math.cos(angle)*length/2; dy=math.sin(angle)*length/2
    strokes.append({'phase':phase,'brush':'dryBrush','x1':round(x-dx,2),'y1':round(y-dy,2),'x2':round(x+dx,2),'y2':round(y+dy,2),'width':round(width,2),'spread':round(width*.92,2),'strands':strands,'seed':int(seed),'color':color,'opacity':round(opacity,3),'role':role})

def underpaint_color(p):
    l=lum(p)/255; return hexcolor((35+95*l,25+65*l,18+42*l))

def apply_stroke(draw,s):
    rgb=parse_hex(s.get('color','#fff')); alpha=int(clamp(round(s.get('opacity',1)*255))); fill=(*rgb,alpha)
    if s.get('brush')=='line':
        w=max(1,int(round(s.get('width',1)))); xy=(s['x1'],s['y1'],s['x2'],s['y2']); draw.line(xy,fill=fill,width=w)
        if w>=3:
            r=w/2
            for x,y in ((s['x1'],s['y1']),(s['x2'],s['y2'])): draw.ellipse((x-r,y-r,x+r,y+r),fill=fill)
    elif s.get('brush')=='dryBrush':
        rng=random.Random(int(s.get('seed',1))); x1,y1,x2,y2=s['x1'],s['y1'],s['x2'],s['y2']; dx=x2-x1;dy=y2-y1; ln=math.hypot(dx,dy) or 1; nx,ny=-dy/ln,dx/ln
        strands=max(5,int(s.get('strands',18))); spread=s.get('spread',s.get('width',12)); sw=max(1,int(round(max(.7,s.get('width',8)/strands*.75))))
        for _ in range(strands):
            off=(rng.random()-.5)*spread; j1=(rng.random()-.5)*2; j2=(rng.random()-.5)*2; a=int(alpha*(.28+rng.random()*.65))
            draw.line((x1+nx*off+j1,y1+ny*off+j1,x2+nx*off+j2,y2+ny*off+j2),fill=(*rgb,a),width=sw)

def render(strokes,w,h,bg):
    im=Image.new('RGB',(w,h),bg); d=ImageDraw.Draw(im,'RGBA')
    for s in strokes: apply_stroke(d,s)
    return im

def metrics(ref,out):
    diff=ImageChops.difference(ref,out); st=ImageStat.Stat(diff); mae=sum(st.mean)/3; rms=math.sqrt(sum(v*v for v in st.rms)/3); psnr=99 if rms==0 else 20*math.log10(255/rms)
    re=ref.convert('L').filter(ImageFilter.FIND_EDGES); oe=out.convert('L').filter(ImageFilter.FIND_EDGES); edge=ImageStat.Stat(ImageChops.difference(re,oe)).mean[0]
    return {'mae':round(mae,4),'rmse':round(rms,4),'psnr':round(psnr,4),'edge_mae':round(edge,4)}

def composition(strokes,sub,face):
    sx0,sy0,sx1,sy1=sub; fx0,fy0,fx1,fy1=face; sw,sh=sx1-sx0,sy1-sy0; fw,fh=fx1-fx0,fy1-fy0; cx=(fx0+fx1)/2; cy=(fy0+fy1)/2; c='#9a8169'
    add_line(strokes,'composition',cx,cy,fh*.78,1.5,c,.28,math.pi/2,'face-center')
    add_line(strokes,'composition',cx,fy0+fh*.42,fw*.84,1.4,c,.26,0,'eye-line')
    add_line(strokes,'composition',cx,fy0+fh*.70,fw*.52,1.2,c,.22,0,'mouth-line')
    add_line(strokes,'composition',sx0+sw*.52,sy0+sh*.78,sw*.70,1.8,c,.30,-.06,'shoulder-gesture')
    for x,y,l,a in [(sx0+sw*.34,sy0+sh*.10,sw*.26,.55),(sx0+sw*.57,sy0+sh*.11,sw*.28,-.15),(sx0+sw*.72,sy0+sh*.28,sh*.22,1.25),(sx0+sw*.23,sy0+sh*.31,sh*.21,1.88),(sx0+sw*.67,sy0+sh*.60,sh*.22,1.43)]: add_line(strokes,'composition',x,y,l,1.5,c,.20,a,'contour-note')

def broad_pass(strokes,im,phase,bg,sub,face,rng,step,length,width,blur,opacity,under=False,order=('headwrap','face','garment','subject')):
    src=im.filter(ImageFilter.GaussianBlur(blur)) if blur else im; p=src.load(); raw=im.load(); w,h=im.size; groups=defaultdict(list)
    for y in range(step//2,h,step):
        for x in range(step//2,w,step):
            q=p[x,y]; r=region(x,y,q,bg,sub,face)
            if r=='background': continue
            a,m=gradient(raw,x,y,w,h)
            if m<4: a={'headwrap':-.10,'face':1.00,'garment':1.35,'subject':1.10}.get(r,0)
            groups[r].append((x,y,q,a,m))
    for r in order:
        vals=groups.get(r,[]); rng.shuffle(vals)
        vals.sort(key=lambda t:(int(t[1]//(step*3)),int(t[0]//(step*3))))
        for x,y,q,a,m in vals:
            ln=length*(.78 if m>24 else 1); wd=width*(.82 if m>28 else 1); col=underpaint_color(q) if under else hexcolor(q)
            add_dry(strokes,phase,x,y,ln,wd,col,opacity,a,rng.randrange(1,2**31-1),('underpaint-' if under else '')+r,24 if under else 18)

def face_structure(strokes,im,face,rng):
    p=im.load(); w,h=im.size; fx0,fy0,fx1,fy1=map(int,face); fx0=max(2,fx0);fy0=max(2,fy0);fx1=min(w-2,fx1);fy1=min(h-2,fy1); fw,fh=fx1-fx0,fy1-fy0; cx=(fx0+fx1)/2; tone=hexcolor(p[int(cx),int((fy0+fy1)/2)])
    add_line(strokes,'face_structure',cx,(fy0+fy1)/2,fh*.70,1.4,tone,.30,math.pi/2,'face-center')
    add_line(strokes,'face_structure',cx,fy0+fh*.42,fw*.74,1.3,tone,.28,0,'eye-line')
    add_line(strokes,'face_structure',cx,fy0+fh*.70,fw*.48,1.2,tone,.25,0,'mouth-line')
    pts=[]
    for y in range(fy0,fy1,8):
        for x in range(fx0,fx1,8):
            a,m=gradient(p,x,y,w,h); pts.append((m,x,y,p[x,y],a))
    pts.sort(reverse=True); pts=pts[:min(len(pts),2400)]
    for m,x,y,q,a in pts:
        add_dry(strokes,'face_structure',x,y,11 if m<18 else 7,4.2 if m<18 else 2.8,hexcolor(q),.88,a,rng.randrange(1,2**31-1),'face-plane',10)

def detail_points(im,bg,sub,face,count,rng):
    p=im.load();w,h=im.size;fx0,fy0,fx1,fy1=face; arr=[]
    for y in range(2,h-2,3):
        for x in range(2,w-2,3):
            q=p[x,y];r=region(x,y,q,bg,sub,face)
            if r=='background': continue
            a,m=gradient(p,x,y,w,h)
            if m<4: continue
            bonus=34 if fx0<=x<=fx1 and fy0<=y<=fy1 else 0; arr.append((m+bonus+rng.random()*4,x,y,q,a,r))
    arr.sort(reverse=True); return arr[:count]

def finish_points(ref,out,bg,sub,face,count):
    rp=ref.load();op=out.load();w,h=ref.size;fx0,fy0,fx1,fy1=face; arr=[]
    for y in range(2,h-2,3):
        for x in range(2,w-2,3):
            q=rp[x,y];r=region(x,y,q,bg,sub,face)
            if r=='background': continue
            err=math.sqrt(sum((q[i]-op[x,y][i])**2 for i in range(3))/3); a,m=gradient(rp,x,y,w,h)
            if fx0<=x<=fx1 and fy0<=y<=fy1: err*=1.20
            sc=err*(1+min(m,80)/240)
            if sc>3: arr.append((sc,x,y,q,a,r))
    arr.sort(reverse=True); return arr[:count]

def main():
    ap=argparse.ArgumentParser(description='公開されている油彩制作過程を参考に、筆運び主体の6工程で描画する')
    ap.add_argument('input'); ap.add_argument('--output',default='strokes.generated.json'); ap.add_argument('--preview',default='preview.png'); ap.add_argument('--metrics',default='metrics.json'); ap.add_argument('--checkpoint-dir'); ap.add_argument('--width',type=int,default=864); ap.add_argument('--height',type=int,default=1024); ap.add_argument('--seed',type=int,default=20260908); ap.add_argument('--detail',type=int,default=18000); ap.add_argument('--finish',type=int,default=10000)
    a=ap.parse_args(); rng=random.Random(a.seed); im=crop_resize(Image.open(a.input),a.width,a.height); bg_rgb=border_bg(im); bg=hexcolor(bg_rgb); sub=infer_subject(im,bg_rgb); face=infer_face(im,sub); strokes=[]; cps=[]; cdir=Path(a.checkpoint_dir) if a.checkpoint_dir else None
    if cdir: cdir.mkdir(parents=True,exist_ok=True)
    def cp(stage,name):
        out=render(strokes,a.width,a.height,bg); cps.append({'stage':stage,'stroke_count':len(strokes),**metrics(im,out)}); out.save(cdir/name) if cdir else None; return out
    composition(strokes,sub,face); cp('composition','01-composition.png')
    broad_pass(strokes,im,'silhouette',bg_rgb,sub,face,rng,52,105,34,22,.58,True,('garment','headwrap','face','subject'))
    broad_pass(strokes,im,'silhouette',bg_rgb,sub,face,rng,38,82,27,15,.78,False,('headwrap','face','garment','subject')); cp('silhouette','02-silhouette.png')
    broad_pass(strokes,im,'light_shadow',bg_rgb,sub,face,rng,24,52,17,9,.80,False,('face','headwrap','garment','subject'))
    broad_pass(strokes,im,'light_shadow',bg_rgb,sub,face,rng,16,34,10,4,.86,False,('face','headwrap','garment','subject')); cp('light_shadow','03-light-shadow.png')
    face_structure(strokes,im,face,rng); cp('face_structure','04-face-structure.png')
    for sc,x,y,q,ang,r in detail_points(im,bg_rgb,sub,face,a.detail,rng): add_line(strokes,'detail',x,y,4.0 if r=='face' else 4.8,1.7 if r=='face' else 2.0,hexcolor(q),.94,ang,r)
    cp('detail','05-detail.png')
    out=render(strokes,a.width,a.height,bg)
    for sc,x,y,q,ang,r in finish_points(im,out,bg_rgb,sub,face,a.finish): add_line(strokes,'finish',x,y,2.8 if r=='face' else 3.3,1.15 if r=='face' else 1.4,hexcolor(q),.98,ang,r)
    out=cp('finish','06-finish.png'); out.save(a.preview)
    for i,s in enumerate(strokes,1): s['id']=i
    data={'metadata':{'slug':'brush-motion-v3','title':'公開描画動画を参考にした筆運び主体の描画','seed':a.seed,'source_mode':'reference-guided-brush-process-v3','stroke_count':len(strokes),'subject_bbox':[round(v,2) for v in sub],'face_bbox':[round(v,2) for v in face],'phase_order':[p[0] for p in PHASES],'background_policy':'toned-ground-no-progress','process_basis':['drawing','underpainting','broad-brush block-in','value/color masses','facial structure','details','final accents'],'quality_metrics':cps[-1]},'canvas':{'width':a.width,'height':a.height,'background':bg},'phases':[{'id':i,'label':l} for i,l in PHASES],'strokes':strokes}
    Path(a.output).write_text(json.dumps(data,separators=(',',':')),encoding='utf-8'); Path(a.metrics).write_text(json.dumps({'checkpoints':cps},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'strokes':len(strokes),'phase_counts':{p:sum(1 for s in strokes if s['phase']==p) for p,_ in PHASES},'brush_counts':{b:sum(1 for s in strokes if s['brush']==b) for b in ['line','dryBrush']},'quality':cps[-1]},ensure_ascii=False))
if __name__=='__main__': main()
