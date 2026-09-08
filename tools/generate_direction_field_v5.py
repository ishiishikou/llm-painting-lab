#!/usr/bin/env python3
"""v5 standard entrypoint: use semantic region direction fields with the v4/v5 tool renderer."""
import math
import random
import reference_to_brush_process as base


def direction_field(x,y,r,sub,face,local_angle,local_strength,phase,rng):
    sx0,sy0,sx1,sy1=sub
    fx0,fy0,fx1,fy1=face
    sw=max(1,sx1-sx0); fw=max(1,fx1-fx0); fh=max(1,fy1-fy0)
    local=base.axis(local_angle)

    if r=='face':
        xx=(x-fx0)/fw; yy=(y-fy0)/fh
        # 顔全体を楕円の接線で回すと渦巻きに見えるため、面ごとに主方向を分ける。
        if yy<.30:
            form=base.axis(.20+.18*(xx-.5))          # 額: やや横
        elif yy<.62:
            form=base.axis(.92+.34*(xx-.5))          # 頬: 斜め〜縦
        else:
            form=base.axis(.28+.28*(xx-.5))          # 顎: やや横
        if phase in {'face_structure','detail','finish'}:
            if .30<yy<.53:
                form=base.blend_axis(form,0.0,.58)    # 目の帯
            if abs(xx-.5)<.13 and .32<yy<.72:
                form=base.blend_axis(form,math.pi/2,.70)  # 鼻筋
            if .63<yy<.78:
                form=base.blend_axis(form,0.0,.58)    # 口の帯
        local_w=.24 if phase=='silhouette' else (.42 if phase=='light_shadow' else .66)

    elif r=='headwrap':
        xx=(x-fx0)/fw; yy=(y-fy0)/fh
        # 冠部は横へ巻き、右側の垂れ布は縦へ落とす。
        if x>fx1+fw*.08 and y>fy0+fh*.12:
            form=base.axis(1.48+.10*(xx-1))
        elif y<fy0+fh*.15:
            form=base.axis(.05+.12*(xx-.5))
        else:
            form=base.axis(.30+.32*max(-.5,min(1.2,yy)))
        local_w=.24 if phase=='silhouette' else (.44 if phase=='light_shadow' else .66)

    elif r=='garment':
        cx=(sx0+sx1)/2
        outward=(x-cx)/(sw/2)
        form=base.axis(math.pi/2+.28*outward)          # 重力方向 + 外側への開き
        local_w=.16 if phase=='silhouette' else (.30 if phase=='light_shadow' else .52)

    else:
        form=base.ellipse_tangent(x,y,sub)
        local_w=.22 if phase=='silhouette' else .42

    if local_strength<4:
        local_w*=.25
    elif local_strength<10:
        local_w*=.55

    angle=base.blend_axis(form,local,local_w)
    jitter={'silhouette':.18,'light_shadow':.12,'face_structure':.10,'detail':.075,'finish':.06,'blend':.09}.get(phase,.10)
    return base.axis(angle+rng.uniform(-jitter,jitter))


base.direction_field=direction_field

if __name__=='__main__':
    base.main()
