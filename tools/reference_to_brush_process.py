#!/usr/bin/env python3
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

PHASES=[('composition','1. 構図'),('silhouette','2. 大きな形'),('light_shadow','3. 明暗の面'),('face_structure','4. 顔構造'),('detail','5. 細部'),('finish','6. 仕上げ')]

def clamp(v,lo=0,hi=255): return max(lo,min(hi,v))
def rgb(v): return tuple(int(clamp(x)) for x in v)
def hexcolor(v): return '#%02x%02x%02x'%rgb(v)
def parse_hex(s): s=s.lstrip('#'); return tuple(int(s[i:i+2],16) for i in (0,2,4))
def lum(p): return .2126*p[0]+.7152*p[1]+.0722*p[2]
def dist(a,b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))
def blend(a,b,t): return tuple(a[i]*(1-t)+b[i]*t for i in range(3))

def crop_resize(im,w,h):
    im=im.convert('RGB'); sr=im.width/im.height; tr=w/h
    if sr>tr:
        nw=int(im.height*tr); left=(im.width-nw)//2; im=im.crop((left,0,left+nw,im.height))
    else:
        nh=int(im.width/tr); top=(im.height-nh)//2; im=im.crop((0,top,im.width,top+nh))
    return im.resize((w,h),Image.Resampling.LANCZOS)

def gradient(pix,x,y,w,h):
    x0,x1=max(0,x-1),min(w-1,x+1); y0,y1=max(0,y-1),min(h-1,y+1)
    gx=lum(pix[x1,y])-lum(pix[x0,y]); gy=lum(pix[x,y1])-lum(pix[x,y0]); m=math.hypot(gx,gy)
    return (math.atan2(gy,gx)+math.pi/2 if m>1.5 else 0.0),m

def axis(angle): return angle % math.pi

def axis_delta(a,b):
    return (axis(a)-axis(b)+math.pi/2)%math.pi-math.pi/2

def blend_axis(a,b,t): return axis(a-axis_delta(a,b)*t)

def ellipse_tangent(x,y,box):
    x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2;rx=max(1,(x1-x0)/2);ry=max(1,(y1-y0)/2)
    nx=(x-cx)/(rx*rx);ny=(y-cy)/(ry*ry)
    return axis(math.atan2(ny,nx)+math.pi/2)

def border_bg(im):
    w,h=im.size; p=im.load(); pts=[]; s=max(4,min(w,h)//100)
    for x in range(0,w,s): pts.extend([p[x,0],p[x,h-1]])
    for y in range(0,h,s): pts.extend([p[0,y],p[w-1,y]])
    ch=[sorted(q[i] for q in pts) for i in range(3)]
    return tuple(c[len(c)//2] for c in ch)

def infer_subject(im,bg):
    w,h=im.size;p=im.load();xs=[];ys=[];bl=lum(bg)
    for y in range(2,h,4):
        for x in range(2,w,4):
            q=p[x,y]
            if dist(q,bg)>45 or lum(q)>bl+24:xs.append(x);ys.append(y)
    if not xs:return (w*.18,h*.08,w*.88,h*.96)
    xs.sort();ys.sort();a=int(len(xs)*.015);b=int(len(xs)*.985)-1
    return xs[a],ys[a],xs[b],ys[b]

def skin(p):
    r,g,b=p;return r>88 and g>58 and b>38 and r>g*1.01 and g>b*.90 and max(p)-min(p)>10

def infer_face(im,sub):
    w,h=im.size;p=im.load();sx0,sy0,sx1,sy1=sub;xs=[];ys=[];y1=min(h,int(sy0+(sy1-sy0)*.67))
    for y in range(max(0,int(sy0)),y1,3):
        for x in range(max(0,int(sx0)),min(w,int(sx1)),3):
            if skin(p[x,y]):xs.append(x);ys.append(y)
    if len(xs)<100:return sx0+(sx1-sx0)*.18,sy0+(sy1-sy0)*.22,sx0+(sx1-sx0)*.67,sy0+(sy1-sy0)*.62
    xs.sort();ys.sort()
    def q(a,f):return a[min(len(a)-1,max(0,int(len(a)*f)))]
    return q(xs,.08),q(ys,.08),q(xs,.92),q(ys,.92)

def inside(box,x,y,pad=0):
    x0,y0,x1,y1=box;return x0-pad<=x<=x1+pad and y0-pad<=y<=y1+pad

def subject_pixel(p,bg):return dist(p,bg)>42 or lum(p)>lum(bg)+22
def strong_subject_pixel(p,bg):return dist(p,bg)>58 or lum(p)>lum(bg)+30

def region(x,y,p,bg,sub,face):
    sx0,sy0,sx1,sy1=sub;fx0,fy0,fx1,fy1=face;sh=sy1-sy0
    if not subject_pixel(p,bg):return 'background'
    if inside(face,x,y,max(10,(fx1-fx0)*.08)):return 'face'
    if y<sy0+sh*.48:return 'headwrap'
    if y>fy1-(fy1-fy0)*.05:return 'garment'
    return 'subject'

def direction_field(x,y,r,sub,face,local_angle,local_strength,phase,rng):
    sx0,sy0,sx1,sy1=sub;fx0,fy0,fx1,fy1=face;sw=max(1,sx1-sx0);sh=max(1,sy1-sy0);fw=max(1,fx1-fx0);fh=max(1,fy1-fy0);local=axis(local_angle)
    if r=='face':
        form=ellipse_tangent(x,y,face);nx=(x-(fx0+fx1)/2)/fw
        if phase in {'face_structure','detail','finish'}:
            yy=(y-fy0)/fh
            if .30<yy<.53:form=blend_axis(form,0.0,.52)
            if abs(nx)<.12 and .32<yy<.72:form=blend_axis(form,math.pi/2,.62)
            if .63<yy<.78:form=blend_axis(form,0.0,.48)
        local_w=.18 if phase=='silhouette' else (.32 if phase=='light_shadow' else .58)
    elif r=='headwrap':
        hb=(sx0,sy0,sx1,max(sy0+1,sy0+sh*.52));form=ellipse_tangent(x,y,hb);local_w=.20 if phase=='silhouette' else (.38 if phase=='light_shadow' else .62)
    elif r=='garment':
        cx=(sx0+sx1)/2;outward=(x-cx)/(sw/2);form=axis(math.pi/2+.28*outward);local_w=.16 if phase=='silhouette' else (.30 if phase=='light_shadow' else .52)
    else:
        form=ellipse_tangent(x,y,sub);local_w=.22 if phase=='silhouette' else .42
    if local_strength<4:local_w*=.25
    elif local_strength<10:local_w*=.55
    a=blend_axis(form,local,local_w);jitter={'silhouette':.18,'light_shadow':.12,'face_structure':.10,'detail':.075,'finish':.06,'blend':.09}.get(phase,.10)
    return axis(a+rng.uniform(-jitter,jitter))

def segment(x,y,length,angle):
    dx=math.cos(angle)*length/2;dy=math.sin(angle)*length/2
    return round(x-dx,2),round(y-dy,2),round(x+dx,2),round(y+dy,2)

def add_line(strokes,phase,x,y,length,width,color,opacity,angle=0,role=''):
    x1,y1,x2,y2=segment(x,y,length,angle);strokes.append({'phase':phase,'brush':'line','x1':x1,'y1':y1,'x2':x2,'y2':y2,'width':round(width,2),'color':color,'opacity':round(opacity,3),'lineCap':'round','role':role})

def add_variable(strokes,phase,x,y,length,w0,w1,color,opacity,angle=0,role='',clip=None):
    x1,y1,x2,y2=segment(x,y,length,angle);s={'phase':phase,'brush':'variableBrush','x1':x1,'y1':y1,'x2':x2,'y2':y2,'widthStart':round(w0,2),'widthEnd':round(w1,2),'width':round(max(w0,w1),2),'color':color,'opacity':round(opacity,3),'role':role}
    if clip:s['clipBox']=[round(v,2) for v in clip]
    strokes.append(s)

def add_flat(strokes,phase,x,y,length,width,color,opacity,angle,seed,role='',clip=None):
    x1,y1,x2,y2=segment(x,y,length,angle);s={'phase':phase,'brush':'flatBrush','x1':x1,'y1':y1,'x2':x2,'y2':y2,'widthStart':round(width,2),'widthEnd':round(width*.82,2),'width':round(width,2),'color':color,'opacity':round(opacity,3),'seed':int(seed),'role':role}
    if clip:s['clipBox']=[round(v,2) for v in clip]
    strokes.append(s)

def add_dry(strokes,phase,x,y,length,width,color,opacity,angle,seed,role='',strands=7,clip=None):
    x1,y1,x2,y2=segment(x,y,length,angle);s={'phase':phase,'brush':'dryBrush','x1':x1,'y1':y1,'x2':x2,'y2':y2,'width':round(width,2),'spread':round(width*.92,2),'strands':strands,'seed':int(seed),'color':color,'opacity':round(opacity,3),'role':role}
    if clip:s['clipBox']=[round(v,2) for v in clip]
    strokes.append(s)

def add_mixer(strokes,phase,x,y,length,width,color,opacity,angle,role='',clip=None,pickup=.18,mix=.58,deposit=.42):
    x1,y1,x2,y2=segment(x,y,length,angle);s={'phase':phase,'brush':'mixerBrush','x1':x1,'y1':y1,'x2':x2,'y2':y2,'width':round(width,2),'color':color,'opacity':round(opacity,3),'pickup':pickup,'mix':mix,'deposit':deposit,'role':role}
    if clip:s['clipBox']=[round(v,2) for v in clip]
    strokes.append(s)

def add_smudge(strokes,phase,x,y,length,width,opacity,angle,role='',clip=None,strength=.30,pickup=.10):
    x1,y1,x2,y2=segment(x,y,length,angle);s={'phase':phase,'brush':'smudgeBrush','x1':x1,'y1':y1,'x2':x2,'y2':y2,'width':round(width,2),'color':'#808080','opacity':round(opacity,3),'strength':strength,'pickup':pickup,'role':role}
    if clip:s['clipBox']=[round(v,2) for v in clip]
    strokes.append(s)

def add_glaze(strokes,phase,x,y,length,width,color,opacity,angle,role='',clip=None):
    x1,y1,x2,y2=segment(x,y,length,angle);s={'phase':phase,'brush':'glaze','x1':x1,'y1':y1,'x2':x2,'y2':y2,'width':round(width,2),'color':color,'opacity':round(opacity,3),'role':role}
    if clip:s['clipBox']=[round(v,2) for v in clip]
    strokes.append(s)

def underpaint_color(p):
    l=lum(p)/255;return hexcolor((35+95*l,25+65*l,18+42*l))

def clipped(s,x,y):
    b=s.get('clipBox');return True if not b else b[0]<=x<=b[2] and b[1]<=y<=b[3]

def draw_variable(im,s,fraction=1.0,texture=False):
    d=ImageDraw.Draw(im,'RGBA');c=parse_hex(s.get('color','#fff'));a=int(clamp(round(s.get('opacity',1)*255)));x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;ln=math.hypot(x2-x1,y2-y1);n=max(2,int(ln/2.5));prev=None;rng=random.Random(int(s.get('seed',1)))
    for i in range(n+1):
        t=i/n;x=x1+(x2-x1)*t;y=y1+(y2-y1)*t
        if not clipped(s,x,y):continue
        w=s.get('widthStart',s.get('width',1))*(1-t)+s.get('widthEnd',s.get('width',1))*t;w=max(.7,w);r=w/2
        if prev:
            px,py,pw=prev;d.line((px,py,x,y),fill=(*c,a),width=max(1,int(round((pw+w)/2))))
        d.ellipse((x-r,y-r,x+r,y+r),fill=(*c,a));prev=(x,y,w)
    if texture and ln>5:
        dx=x2-x1;dy=y2-y1;L=math.hypot(dx,dy) or 1;nx,ny=-dy/L,dx/L
        for _ in range(3):
            off=(rng.random()-.5)*s.get('width',10)*.65;aa=int(a*(.16+rng.random()*.10));d.line((x1+nx*off,y1+ny*off,x2+nx*off,y2+ny*off),fill=(*c,aa),width=max(1,int(s.get('width',10)*.08)))

def draw_dry(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');c=parse_hex(s.get('color','#fff'));alpha=int(clamp(round(s.get('opacity',1)*255)));x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;dx=x2-x1;dy=y2-y1;ln=math.hypot(dx,dy) or 1;nx,ny=-dy/ln,dx/ln;rng=random.Random(int(s.get('seed',1)));strands=max(4,int(s.get('strands',7)));spread=s.get('spread',s.get('width',12));sw=max(1,int(round(max(.9,s.get('width',8)/strands*1.15))))
    for _ in range(strands):
        off=(rng.random()-.5)*spread;j1=(rng.random()-.5)*2;j2=(rng.random()-.5)*2;xa=x1+nx*off+j1;ya=y1+ny*off+j1;xb=x2+nx*off+j2;yb=y2+ny*off+j2
        if s.get('clipBox'):
            b=s['clipBox'];xa=max(b[0],min(b[2],xa));xb=max(b[0],min(b[2],xb));ya=max(b[1],min(b[3],ya));yb=max(b[1],min(b[3],yb))
        aa=int(alpha*(.34+rng.random()*.58));d.line((xa,ya,xb,yb),fill=(*c,aa),width=sw)

def draw_mixer(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');paint=tuple(float(v) for v in parse_hex(s.get('color','#fff')));x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;ln=math.hypot(x2-x1,y2-y1);n=max(2,int(ln/3));pickup=float(s.get('pickup',.18));mix=float(s.get('mix',.58));deposit=float(s.get('deposit',.42));base_alpha=float(s.get('opacity',1))*deposit
    for i in range(n+1):
        t=i/n;x=x1+(x2-x1)*t;y=y1+(y2-y1)*t
        if not clipped(s,x,y):continue
        xi=max(0,min(im.width-1,int(round(x))));yi=max(0,min(im.height-1,int(round(y))));under=im.getpixel((xi,yi));paint=blend(paint,under,pickup);out=blend(under,paint,mix);r=max(1,s.get('width',8)/2);d.ellipse((x-r,y-r,x+r,y+r),fill=(*rgb(out),int(clamp(base_alpha*255))))

def draw_smudge(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;xi=max(0,min(im.width-1,int(round(x1))));yi=max(0,min(im.height-1,int(round(y1))));carry=tuple(float(v) for v in im.getpixel((xi,yi)));ln=math.hypot(x2-x1,y2-y1);n=max(2,int(ln/3));strength=float(s.get('strength',.30))*float(s.get('opacity',1));pickup=float(s.get('pickup',.10))
    for i in range(1,n+1):
        t=i/n;x=x1+(x2-x1)*t;y=y1+(y2-y1)*t
        if not clipped(s,x,y):continue
        xi=max(0,min(im.width-1,int(round(x))));yi=max(0,min(im.height-1,int(round(y))));under=im.getpixel((xi,yi));r=max(1,s.get('width',8)/2);d.ellipse((x-r,y-r,x+r,y+r),fill=(*rgb(carry),int(clamp(strength*255))));carry=blend(carry,under,pickup)

def draw_glaze(im,s,fraction=1.0):
    d=ImageDraw.Draw(im,'RGBA');c=parse_hex(s.get('color','#fff'));x1,y1=s['x1'],s['y1'];x2=x1+(s['x2']-x1)*fraction;y2=y1+(s['y2']-y1)*fraction;ln=math.hypot(x2-x1,y2-y1);n=max(2,int(ln/3));a=int(clamp(float(s.get('opacity',.1))*255));r=max(1,s.get('width',12)/2)
    for i in range(n+1):
        t=i/n;x=x1+(x2-x1)*t;y=y1+(y2-y1)*t
        if clipped(s,x,y):d.ellipse((x-r,y-r,x+r,y+r),fill=(*c,a))

def apply_stroke(im,s,fraction=1.0):
    b=s.get('brush','line')
    if b=='dryBrush':draw_dry(im,s,fraction)
    elif b=='flatBrush':draw_variable(im,s,fraction,True)
    elif b=='variableBrush':draw_variable(im,s,fraction)
    elif b=='mixerBrush':draw_mixer(im,s,fraction)
    elif b=='smudgeBrush':draw_smudge(im,s,fraction)
    elif b=='glaze':draw_glaze(im,s,fraction)
    else:
        t=dict(s);t.setdefault('widthStart',s.get('width',1));t.setdefault('widthEnd',s.get('width',1));draw_variable(im,t,fraction)

def render(strokes,w,h,bg):
    im=Image.new('RGB',(w,h),bg)
    for s in strokes:apply_stroke(im,s)
    return im

def metrics(ref,out):
    diff=ImageChops.difference(ref,out);st=ImageStat.Stat(diff);mae=sum(st.mean)/3;rms=math.sqrt(sum(v*v for v in st.rms)/3);psnr=99 if rms==0 else 20*math.log10(255/rms);re=ref.convert('L').filter(ImageFilter.FIND_EDGES);oe=out.convert('L').filter(ImageFilter.FIND_EDGES);edge=ImageStat.Stat(ImageChops.difference(re,oe)).mean[0]
    return {'mae':round(mae,4),'rmse':round(rms,4),'psnr':round(psnr,4),'edge_mae':round(edge,4)}

def composition(strokes,sub,face):
    sx0,sy0,sx1,sy1=sub;fx0,fy0,fx1,fy1=face;sw,sh=sx1-sx0,sy1-sy0;fw,fh=fx1-fx0,fy1-fy0;cx=(fx0+fx1)/2;cy=(fy0+fy1)/2;c='#9a8169'
    add_variable(strokes,'composition',cx,cy,fh*.78,1.8,.7,c,.28,math.pi/2,'face-center');add_variable(strokes,'composition',cx,fy0+fh*.42,fw*.84,1.7,.6,c,.26,0,'eye-line');add_variable(strokes,'composition',cx,fy0+fh*.70,fw*.52,1.5,.55,c,.22,0,'mouth-line');add_variable(strokes,'composition',sx0+sw*.52,sy0+sh*.78,sw*.70,2.3,.8,c,.30,-.06,'shoulder-gesture')
    for x,y,l,a in [(sx0+sw*.34,sy0+sh*.10,sw*.26,.55),(sx0+sw*.57,sy0+sh*.11,sw*.28,-.15),(sx0+sw*.72,sy0+sh*.28,sh*.22,1.25),(sx0+sw*.23,sy0+sh*.31,sh*.21,1.88),(sx0+sw*.67,sy0+sh*.60,sh*.22,1.43)]:add_variable(strokes,'composition',x,y,l,2.0,.7,c,.20,a,'contour-note')

def broad_pass(strokes,im,phase,bg,sub,face,rng,step,length,width,blur,opacity,under=False,order=('headwrap','face','garment','subject')):
    src=im.filter(ImageFilter.GaussianBlur(blur)) if blur else im;p=src.load();raw=im.load();w,h=im.size;groups=defaultdict(list)
    for y in range(step//2,h,step):
        for x in range(step//2,w,step):
            q=p[x,y];r=region(x,y,q,bg,sub,face)
            if r=='background':continue
            la,m=gradient(raw,x,y,w,h);a=direction_field(x,y,r,sub,face,la,m,phase,rng);groups[r].append((x,y,q,a,m))
    for r in order:
        vals=groups.get(r,[]);rng.shuffle(vals);vals.sort(key=lambda t:(int(t[1]//(step*3)),int(t[0]//(step*3))))
        for x,y,q,a,m in vals:
            ln=length*(.78 if m>24 else 1);wd=width*(.82 if m>28 else 1);col=underpaint_color(q) if under else hexcolor(q);seed=rng.randrange(1,2**31-1);clip=face if r=='face' else sub
            if under:add_dry(strokes,phase,x,y,ln,wd,col,opacity,a,seed,'underpaint-'+r,6,clip)
            else:add_flat(strokes,phase,x,y,ln,wd,col,opacity,a,seed,r,clip)

def transition_candidates(im,bg,sub,face,count,rng,regions=('face','headwrap')):
    p=im.load();w,h=im.size;arr=[]
    for y in range(6,h-6,7):
        for x in range(6,w-6,7):
            q=p[x,y];r=region(x,y,q,bg,sub,face)
            if r not in regions:continue
            a,m=gradient(p,x,y,w,h)
            if 3.5<=m<=28:arr.append((28-abs(m-12)+rng.random()*3,x,y,q,a,m,r))
    arr.sort(reverse=True);return arr[:count]

def add_blending_pass(strokes,im,phase,bg,sub,face,rng,mix_count,smudge_count):
    vals=transition_candidates(im,bg,sub,face,mix_count+smudge_count,rng)
    for i,(_,x,y,q,la,m,r) in enumerate(vals):
        a=direction_field(x,y,r,sub,face,la,m,'blend',rng);clip=face if r=='face' else sub
        if i<mix_count:add_mixer(strokes,phase,x,y,18 if r=='face' else 24,8 if r=='face' else 11,hexcolor(q),.55,a,f'mix-{r}',clip,.16,.56,.38)
        else:add_smudge(strokes,phase,x,y,12 if r=='face' else 16,7 if r=='face' else 9,.55,a,f'smudge-{r}',clip,.24,.08)

def face_structure(strokes,im,sub,face,rng):
    p=im.load();w,h=im.size;fx0,fy0,fx1,fy1=map(int,face);fx0=max(2,fx0);fy0=max(2,fy0);fx1=min(w-2,fx1);fy1=min(h-2,fy1);fw,fh=fx1-fx0,fy1-fy0;cx=(fx0+fx1)/2;tone=hexcolor(p[int(cx),int((fy0+fy1)/2)])
    add_variable(strokes,'face_structure',cx,(fy0+fy1)/2,fh*.70,1.8,.6,tone,.30,math.pi/2,'face-center',face);add_variable(strokes,'face_structure',cx,fy0+fh*.42,fw*.74,1.6,.55,tone,.28,0,'eye-line',face);add_variable(strokes,'face_structure',cx,fy0+fh*.70,fw*.48,1.5,.5,tone,.25,0,'mouth-line',face)
    pts=[]
    for y in range(fy0,fy1,8):
        for x in range(fx0,fx1,8):
            la,m=gradient(p,x,y,w,h);pts.append((m,x,y,p[x,y],la))
    pts.sort(reverse=True);pts=pts[:min(len(pts),2400)]
    for j,(m,x,y,q,la) in enumerate(pts):
        a=direction_field(x,y,'face',sub,face,la,m,'face_structure',rng)
        if j<650:add_variable(strokes,'face_structure',x,y,10 if m<18 else 6.5,4.8,1.6,hexcolor(q),.88,a,'face-plane',face)
        else:add_dry(strokes,'face_structure',x,y,11 if m<18 else 7,4.2 if m<18 else 2.8,hexcolor(q),.84,a,rng.randrange(1,2**31-1),'face-plane',6,face)

def detail_points(im,bg,sub,face,count,rng):
    p=im.load();w,h=im.size;fx0,fy0,fx1,fy1=face;groups=defaultdict(list)
    for y in range(2,h-2,3):
        for x in range(2,w-2,3):
            q=p[x,y]
            if not inside(sub,x,y,18) or not strong_subject_pixel(q,bg):continue
            r=region(x,y,q,bg,sub,face)
            if r=='background':continue
            la,m=gradient(p,x,y,w,h)
            if m<4:continue
            bonus=38 if fx0<=x<=fx1 and fy0<=y<=fy1 else 0;groups[r].append((m+bonus+rng.random()*4,x,y,q,la,m,r))
    quotas={'face':.52,'headwrap':.25,'garment':.16,'subject':.07};result=[]
    for r,f in quotas.items():
        vals=groups.get(r,[]);vals.sort(reverse=True);result.extend(vals[:int(count*f)])
    result.sort(reverse=True);return result[:count]

def finish_points(ref,out,bg,sub,face,count,rng):
    rp=ref.load();op=out.load();w,h=ref.size;fx0,fy0,fx1,fy1=face;groups=defaultdict(list)
    for y in range(2,h-2,3):
        for x in range(2,w-2,3):
            q=rp[x,y]
            if not inside(sub,x,y,18) or not strong_subject_pixel(q,bg):continue
            r=region(x,y,q,bg,sub,face)
            if r=='background':continue
            err=math.sqrt(sum((q[i]-op[x,y][i])**2 for i in range(3))/3);la,m=gradient(rp,x,y,w,h)
            if fx0<=x<=fx1 and fy0<=y<=fy1:err*=1.25
            sc=err*(1+min(m,80)/240)
            if sc>3:groups[r].append((sc,x,y,q,la,m,r))
    quotas={'face':.58,'headwrap':.23,'garment':.13,'subject':.06};result=[]
    for r,f in quotas.items():
        vals=groups.get(r,[]);vals.sort(reverse=True);result.extend(vals[:int(count*f)])
    result.sort(reverse=True);return result[:count]

def region_average(im,bg,sub,face,wanted):
    p=im.load();w,h=im.size;vals=[]
    for y in range(4,h,8):
        for x in range(4,w,8):
            q=p[x,y]
            if region(x,y,q,bg,sub,face)==wanted:vals.append(q)
    if not vals:return (128,128,128)
    return tuple(sum(v[i] for v in vals)/len(vals) for i in range(3))

def glaze_pass(strokes,im,bg,sub,face,rng):
    sx0,sy0,sx1,sy1=sub;fx0,fy0,fx1,fy1=face;specs=[('face',face,region_average(im,bg,sub,face,'face'),12,34,16,1.0),('headwrap',(sx0,sy0,sx1,sy0+(sy1-sy0)*.48),region_average(im,bg,sub,face,'headwrap'),16,52,22,-.10),('garment',(sx0,fy1,sx1,sy1),region_average(im,bg,sub,face,'garment'),16,58,26,1.35)]
    for name,box,col,n,length,width,angle in specs:
        x0,y0,x1,y1=box
        for _ in range(n):add_glaze(strokes,'finish',rng.uniform(x0,x1),rng.uniform(y0,y1),length,width,hexcolor(col),.07 if name=='face' else .055,axis(angle+rng.uniform(-.12,.12)),f'glaze-{name}',box)

def direction_stats(strokes,bins=12):
    broad=[s for s in strokes if s.get('phase') in {'silhouette','light_shadow'} and s.get('brush') in {'flatBrush','dryBrush'}];out={}
    for role in ['all','face','headwrap','garment','subject']:
        vals=broad if role=='all' else [s for s in broad if s.get('role','').removeprefix('underpaint-')==role];hist=[0]*bins
        for s in vals:
            a=axis(math.atan2(s['y2']-s['y1'],s['x2']-s['x1']));hist[min(bins-1,int(a/math.pi*bins))]+=1
        n=sum(hist);out[role]={'count':n,'histogram':hist,'dominant_share':round(max(hist)/n,4) if n else 0,'active_bins':sum(1 for v in hist if n and v/n>=.03)}
    return out

def main():
    ap=argparse.ArgumentParser(description='部位別方向場と描画道具ルールを使う6工程の油彩風描画');ap.add_argument('input');ap.add_argument('--output',default='strokes.generated.json');ap.add_argument('--preview',default='preview.png');ap.add_argument('--metrics',default='metrics.json');ap.add_argument('--checkpoint-dir');ap.add_argument('--width',type=int,default=864);ap.add_argument('--height',type=int,default=1024);ap.add_argument('--seed',type=int,default=20260908);ap.add_argument('--detail',type=int,default=14000);ap.add_argument('--finish',type=int,default=7000);a=ap.parse_args()
    rng=random.Random(a.seed);im=crop_resize(Image.open(a.input),a.width,a.height);bg_rgb=border_bg(im);bg=hexcolor(bg_rgb);sub=infer_subject(im,bg_rgb);face=infer_face(im,sub);strokes=[];cps=[];cdir=Path(a.checkpoint_dir) if a.checkpoint_dir else None
    if cdir:cdir.mkdir(parents=True,exist_ok=True)
    def cp(stage,name):
        out=render(strokes,a.width,a.height,bg);cps.append({'stage':stage,'stroke_count':len(strokes),**metrics(im,out)});out.save(cdir/name) if cdir else None;return out
    composition(strokes,sub,face);cp('composition','01-composition.png')
    broad_pass(strokes,im,'silhouette',bg_rgb,sub,face,rng,52,105,34,22,.58,True,('garment','headwrap','face','subject'));broad_pass(strokes,im,'silhouette',bg_rgb,sub,face,rng,38,82,27,15,.72,False,('headwrap','face','garment','subject'));cp('silhouette','02-silhouette.png')
    broad_pass(strokes,im,'light_shadow',bg_rgb,sub,face,rng,24,52,17,9,.70,False,('face','headwrap','garment','subject'));broad_pass(strokes,im,'light_shadow',bg_rgb,sub,face,rng,16,34,10,4,.76,False,('face','headwrap','garment','subject'));add_blending_pass(strokes,im,'light_shadow',bg_rgb,sub,face,rng,260,120);cp('light_shadow','03-light-shadow.png')
    face_structure(strokes,im,sub,face,rng);add_blending_pass(strokes,im,'face_structure',bg_rgb,sub,face,rng,90,24);cp('face_structure','04-face-structure.png')
    for j,(sc,x,y,q,la,m,r) in enumerate(detail_points(im,bg_rgb,sub,face,a.detail,rng)):
        a_dir=direction_field(x,y,r,sub,face,la,m,'detail',rng);clip=face if r=='face' else sub
        if j<int(a.detail*.62):add_variable(strokes,'detail',x,y,5.3 if r=='face' else 6.2,2.5 if r=='face' else 3.0,.65,hexcolor(q),.94,a_dir,r,clip)
        else:add_line(strokes,'detail',x,y,4.2 if r=='face' else 5.0,1.55 if r=='face' else 1.85,hexcolor(q),.92,a_dir,r)
    cp('detail','05-detail.png')
    glaze_pass(strokes,im,bg_rgb,sub,face,rng);add_blending_pass(strokes,im,'finish',bg_rgb,sub,face,rng,70,45);out=render(strokes,a.width,a.height,bg)
    for j,(sc,x,y,q,la,m,r) in enumerate(finish_points(im,out,bg_rgb,sub,face,a.finish,rng)):
        a_dir=direction_field(x,y,r,sub,face,la,m,'finish',rng);clip=face if r=='face' else sub
        if j<int(a.finish*.68):add_variable(strokes,'finish',x,y,3.7 if r=='face' else 4.2,1.9 if r=='face' else 2.1,.40,hexcolor(q),.97,a_dir,r,clip)
        else:add_line(strokes,'finish',x,y,3.0 if r=='face' else 3.5,1.1 if r=='face' else 1.3,hexcolor(q),.96,a_dir,r)
    out=cp('finish','06-finish.png');out.save(a.preview)
    for i,s in enumerate(strokes,1):s['id']=i
    brushes=sorted({s['brush'] for s in strokes});dstats=direction_stats(strokes);data={'metadata':{'slug':'direction-field-v5','title':'部位別方向場と描画道具ルールを使う描画工程','seed':a.seed,'source_mode':'reference-guided-direction-field-v5','stroke_count':len(strokes),'subject_bbox':[round(v,2) for v in sub],'face_bbox':[round(v,2) for v in face],'phase_order':[p[0] for p in PHASES],'background_policy':'toned-ground-no-progress','tool_policy':'docs/painting-tool-rules.md','evaluation_policy':'docs/painting-evaluation-rules.md','direction_policy':'region-form-following-v5','available_generated_brushes':brushes,'direction_stats':dstats,'process_basis':['drawing','underpainting','form-following flat-brush block-in','value/color masses','mixer blending','smudge edge control','facial structure','importance-weighted details','glaze','final accents'],'quality_metrics':cps[-1]},'canvas':{'width':a.width,'height':a.height,'background':bg},'phases':[{'id':i,'label':l} for i,l in PHASES],'strokes':strokes}
    Path(a.output).write_text(json.dumps(data,separators=(',',':')),encoding='utf-8');Path(a.metrics).write_text(json.dumps({'checkpoints':cps,'direction_stats':dstats},ensure_ascii=False,indent=2),encoding='utf-8');counts={b:sum(1 for s in strokes if s['brush']==b) for b in brushes};print(json.dumps({'strokes':len(strokes),'phase_counts':{p:sum(1 for s in strokes if s['phase']==p) for p,_ in PHASES},'brush_counts':counts,'masked_strokes':sum(1 for s in strokes if s.get('clipBox')),'direction_stats':dstats,'quality':cps[-1]},ensure_ascii=False))
if __name__=='__main__':main()
