"""EMBERWING - third film. 3D world + the dragon rig + cinematic post."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

from lib import clamp, smooth, sstep, lerp, ease_io, ease_out, ease_in, coords, arr, rgb_img, rgba_from, fbm_img, W, H
import world as Wm
import dragon as DR
from film import dragon_layer, twos, kf, flap_val, FLY, spirit_light, MEM, cache

RW, RH = 854, 480
SUN = np.array(Wm.SUN)


# ------------------------------------------------------------------ cinematic post

def sun_screen(cam):
    v = cam.v
    f, r, u, tf = v[3:6], v[6:9], v[9:12], v[12]
    d = SUN
    z = d @ f
    if z <= 0.05:
        return None
    x = (d @ r) / z / (tf * W / H)
    y = (d @ u) / z / tf
    return (x * 0.5 + 0.5) * W, (0.5 - y * 0.5) * H


def godrays(hdr, sp, strength=0.6, n=28, decay=0.965):
    if sp is None or strength <= 0:
        return 0
    h, w = hdr.shape[:2]
    lum = hdr.mean(2)
    m = np.clip(lum - 1.3, 0, 4)
    small = Image.fromarray(np.uint8(np.clip(m / 4, 0, 1) * 255)).resize((w // 4, h // 4), Image.BILINEAR)
    sx, sy = sp[0] / W * (w // 4), sp[1] / H * (h // 4)
    acc = np.zeros((h // 4, w // 4), np.float32)
    wgt = 1.0
    a = np.asarray(small, np.float32) / 255
    for k in range(n):
        s = 1 - k * 0.022
        im = small.transform(small.size, Image.AFFINE, (s, 0, sx * (1 - s), 0, s, sy * (1 - s)), Image.BILINEAR)
        acc += np.asarray(im, np.float32) / 255 * wgt
        wgt *= decay
    acc /= n * 0.5
    big = np.asarray(Image.fromarray(np.uint8(np.clip(acc, 0, 1) * 255)).resize((w, h), Image.BILINEAR), np.float32) / 255
    return big[..., None] * np.array([1.0, 0.75, 0.45], np.float32) * strength


def bloom(hdr, th=1.0, strength=0.5):
    h, w = hdr.shape[:2]
    br = np.clip(hdr - th, 0, 6) / 6
    im = Image.fromarray(np.uint8(np.clip(br, 0, 1) * 255))
    acc = 0
    for r, wgt in ((2, 0.5), (6, 0.35), (16, 0.25)):
        s = im.resize((w // 4, h // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(r))
        acc = acc + np.asarray(s.resize((w, h), Image.BILINEAR), np.float32) / 255 * wgt
    return acc * 6 * strength


_grain = None


def grade(hdr, exposure=1.0, sat=1.18, contrast=1.08, warm=0.0, vig=0.35):
    global _grain
    x = Wm.tonemap(hdr * exposure)
    x = np.clip(x, 0, 1) ** (1 / 2.2)
    l = x.mean(2, keepdims=True)
    x = l + (x - l) * sat
    x = 0.5 + (x - 0.5) * contrast
    # split tone: teal shadows, warm highlights
    sh = np.clip(1 - l * 1.6, 0, 1)
    hi = np.clip((l - 0.55) * 2, 0, 1)
    x = x + sh * np.array([-0.025, 0.005, 0.03]) + hi * np.array([0.03, 0.01, -0.03]) * (1 + warm)
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 * 0.8 + ((yy - h / 2) / (h / 2)) ** 2)
    x = x * (1 - vig * np.clip(d - 0.45, 0, 1) ** 1.6)[..., None]
    if _grain is None or _grain.shape[:2] != (h, w):
        _grain = (np.random.default_rng(5).standard_normal((h, w)).astype(np.float32) * 0.012)
    x = x + _grain[..., None]
    return np.clip(x, 0, 1)


def to_img(x):
    return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255 + 0.5))


def render_world(cam, P, blur=0.0):
    hdr = Wm.shoot(cam, P, RW, RH)
    im = Image.fromarray(np.uint8(np.clip(hdr / 8, 0, 1) ** (1 / 2.0) * 255)) if False else None
    # upscale HDR (float) to output size
    up = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        ch = Image.fromarray(hdr[..., c].astype(np.float32), 'F').resize((W, H), Image.BICUBIC)
        a_ = np.asarray(ch, np.float32)
        if blur:
            a_ = ndimage.gaussian_filter(a_, blur)
        up[..., c] = a_
    return np.maximum(up, 0)


def add_layer_hdr(hdr, layer, gain=1.0):
    a = arr(layer)
    al = a[..., 3:4]
    lin = a[..., :3] ** 2.2 * gain
    return hdr * (1 - al) + lin * al


def finish3(hdr, cam, rays=0.55, bloom_s=0.45, exposure=1.0, sat=1.18, warm=0.0, extra=None, vig=0.35):
    sp = sun_screen(cam) if cam is not None else None
    hdr = hdr + bloom(hdr, 1.0, bloom_s)
    hdr = hdr + godrays(hdr, sp, rays)
    if extra is not None:
        hdr = hdr + extra
    x = grade(hdr, exposure, sat, 1.08, warm, vig)
    img = to_img(x)
    return img.filter(ImageFilter.UnsharpMask(1.2, 60, 2))


def lin_light(rgb_arr, gain=1.0):
    return rgb_arr * gain


def embers(t, pts_fn, n, seed, life=1.6, size=1.0):
    """additive ember particles. pts_fn(age_t) -> emission point (x,y) at time t - age"""
    rg = np.random.default_rng(seed)
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    for k in range(n):
        ph = rg.random() * life
        age = (t + ph) % life
        t0 = t - age
        ex, ey = pts_fn(t0, k)
        vx, vy = rg.normal(-40, 30), rg.normal(-30, 25)
        x = ex + vx * age + 10 * math.sin(age * 4 + k)
        y = ey + vy * age - 12 * age * age
        a = (1 - age / life) ** 1.5
        r = (1.2 + rg.random() * 1.8) * size
        d.ellipse([x - r, y - r, x + r, y + r], fill=int(255 * a))
    a = arr(im)
    g = arr(im.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR))
    return (a[..., None] * np.array([3.0, 2.0, 0.9]) + g[..., None] * np.array([3.0, 1.4, 0.4]) * 2.0)


# ------------------------------------------------------------------ camera helpers

def rx(z):
    return float(Wm.river_x(np.array(z)))


def river_cam(z, y, side=0.0, look_ahead=80.0, look_y=None, yaw=0.0, fov=68, roll=0.0, pitch=0.0):
    x = rx(z) + side
    tz = z + look_ahead
    tx = rx(tz) + side
    d = np.array([tx - x, 0, tz - z])
    d /= np.linalg.norm(d)
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.array([d[0] * c + d[2] * s, 0, -d[0] * s + d[2] * c])
    ly = y + math.tan(pitch) * 100 if look_y is None else look_y
    return Wm.Cam((x, y, z), (x + d[0] * 100, ly, z + d[2] * 100), fov=fov, roll=roll)


def dragon_hdr(hdr, pose, tint=(1, 1, 1), rim_amt=1.0, glow=0.0, gain=1.15, sun_side=1):
    lay, A = dragon_layer(pose, tint=tint, rim=(255, 226, 170), rim_amt=rim_amt, grad=0.2, glow=glow,
                          glow_col=(255, 214, 140))
    return add_layer_hdr(hdr, lay, gain), A, lay


# ------------------------------------------------------------------ SHOTS

def S1(lt, T):
    u = lt / 6
    z = lerp(60, 300, ease_io(u))
    x = rx(z) - 140
    cam = Wm.Cam((x, lerp(300, 280, u), z), (x + 0.25 * 600, 160, z + 600), fov=60, roll=0.04 * math.sin(lt * 0.4))
    k = ease_io(clamp((lt - 0.5) / 5.0))
    P = Wm.params(Wm.SUN, exposure=1.0, stars=lerp(2.0, 0.0, k), zen=(lerp(0.01, 0.05, k), lerp(0.02, 0.13, k), lerp(0.08, 0.38, k)),
                  hor=(lerp(0.4, 0.95, k), lerp(0.25, 0.62, k), lerp(0.3, 0.4, k)), sunc=(1.9 * k + 0.1, 1.15 * k + 0.05, 0.6 * k + 0.05),
                  mist=0.004, mist_c=(0.3, 0.34, 0.5), time=T)
    hdr = render_world(cam, P)
    img = finish3(hdr, cam, rays=0.6 * k, exposure=lerp(0.55, 1.0, k))
    return img, ease_io(clamp(lt / 1.8))


def rock_fg(lt):
    """painted rock perch, foreground (frame coords), rim-lit by the sunrise"""
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    pts = [(0, H), (0, 560), (120, 545), (260, 515), (420, 518), (560, 540), (640, 575), (700, 640), (740, H)]
    d.polygon(pts, fill=(70, 56, 60, 255))
    d.line(pts[1:-1], fill=(255, 200, 130, 255), width=4)
    a = arr(lay)
    n = fbm_img(W, H, 14, 4, 77)
    n2 = fbm_img(W, H, 3, 2, 78)
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    a[..., :3] *= ((0.55 + 0.7 * n) * (0.85 + 0.3 * n2) * np.clip(1.2 - (yy - 520) / 260, 0.35, 1.2))[..., None]
    return rgba_from(a[..., :3], a[..., 3])


def S2(lt, T):
    cam = Wm.Cam((rx(420) - 90, 250, 420), (rx(420) - 90 + 0.3 * 600, 230, 420 + 600), fov=55)
    P = Wm.params(Wm.SUN, mist=0.003, time=T)
    hdr = bg_cache('s2', cam, P, 5.0)
    t = twos(lt)
    crouch = sstep(3.0, 3.6, t)
    base = dict(x=0, y=0, scale=105, legs='sit' if crouch < 0.5 else 'stand', body_angle=-0.25 + 0.3 * crouch,
                neck_angle=-1.2 + 0.5 * crouch, head_angle=-0.15 + 0.15 * crouch, look=(0.7, -0.25),
                eye_open=kf(t, [(0, 1.0), (1.3, 1.0), (1.38, 0.0), (1.46, 1.0)]), pupil=1.1,
                smile=0.5 * sstep(1.8, 2.4, t), brow=0.5 * sstep(2.2, 2.8, t), squint=0.2 * sstep(2.2, 2.8, t),
                ear=0.6 + 0.25 * math.sin(lt * 9), tail_angle=2.9, tail_wave=0.25, tail_phase=lt * 3,
                wing_spread=0.3 * crouch + 0.06 * (1 + math.sin(lt * 7)), wing_tremble=math.sin(lt * 23) * 0.6,
                light=(1, -0.3), ss=2, lw=1.6, pal='gold', blush=0.4)
    from film import place
    p = place(base, 'paw', 330, 520)
    fg = rock_fg(lt)
    hdr = add_layer_hdr(hdr, fg, 1.0)
    hdr, A, lay = dragon_hdr(hdr, p, tint=(0.9, 0.86, 0.9), rim_amt=1.0)
    ex, ey = A['eye']
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glint = np.exp(-((xx - ex - 8) ** 2 + (yy - ey + 6) ** 2) / 60) * 3.0
    hdr = hdr + glint[..., None] * np.array([1.0, 0.8, 0.5])
    # wind-blown snow
    rg = np.random.default_rng(3)
    im = Image.new('L', (W, H), 0)
    dd = ImageDraw.Draw(im)
    for k in range(70):
        x0, y0 = rg.uniform(0, W), rg.uniform(0, H)
        x = (x0 - lt * rg.uniform(300, 600)) % W
        y = (y0 + lt * rg.uniform(10, 40)) % H
        dd.line([(x, y), (x + 10, y - 1)], fill=int(rg.uniform(60, 200)), width=1)
    extra = arr(im)[..., None] * np.array([1.2, 1.1, 1.0])
    return finish3(hdr, cam, rays=0.7, extra=extra), 1.0


_bg = {}


def bg_cache(key, cam, P, blur):
    if key not in _bg:
        _bg[key] = render_world(cam, P, blur)
    return _bg[key].copy()


def S3(lt, T):
    u = lt / 5
    z0 = 700
    z = z0 + 260 * ease_in(clamp(u * 1.1))
    y = lerp(270, 70, ease_io(u))
    pitch = lerp(-0.62, -0.12, ease_io(u))
    cam = river_cam(z, y, side=-20, pitch=pitch, fov=70, roll=0.12 * math.sin(lt * 1.2))
    P = Wm.params(Wm.SUN, mist=0.003, time=T)
    hdr = render_world(cam, P)
    t = twos(lt)
    open_ = ease_out(clamp((t - 3.4) / 0.5))
    fl = flap_val(t, 0.45) if t > 3.9 else 1.0
    p = dict(FLY, x=lerp(520, 600, u), y=lerp(250, 330, u) + 12 * math.sin(lt * 3), scale=62,
             body_angle=lerp(1.0, 0.05, open_), wing_spread=lerp(0.12, 1.0, open_), wing_flap=fl,
             neck_angle=lerp(-0.1, -0.45, open_), head_angle=0.1, squint=0.5 * (1 - open_), mouth=0.15,
             ear=0.9, tail_wave=0.8, tail_phase=t * 8, light=(0.6, -0.8), ss=2, lw=1.3, pal='gold', smile=0.4 * open_)
    hdr, A, lay = dragon_hdr(hdr, p, tint=(0.95, 0.9, 0.92))
    extra = 0
    g = lt - 3.45
    if g > 0:
        r = 80 + 900 * ease_out(g / 0.7)
        im = Image.new('L', (W, H), 0)
        hx, hy = A['head']
        ImageDraw.Draw(im).ellipse([hx - 120 - r, hy - r * 0.5, hx - 120 + r, hy + r * 0.5], outline=255, width=int(18 * (1 - clamp(g / 0.7))) + 2)
        extra = arr(im.filter(ImageFilter.GaussianBlur(8)))[..., None] * np.array([1.4, 1.35, 1.3]) * (1 - clamp(g / 0.7))
    # speed lines while diving
    sp = 1 - open_
    im = Image.new('L', (W, H), 0)
    dd = ImageDraw.Draw(im)
    rg = np.random.default_rng(7)
    for k in range(50):
        x0 = rg.uniform(0, W)
        yy = (rg.uniform(0, H) - lt * 1800 * rg.uniform(0.6, 1.2)) % H
        dd.line([(x0, yy), (x0, yy + 90)], fill=int(35 * sp), width=1)
    extra = extra + arr(im)[..., None]
    return finish3(hdr, cam, rays=0.45, extra=extra), 1.0


def spray(t, xs, y0, n, seed, strength=1.0):
    rg = np.random.default_rng(seed)
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    for k in range(n):
        ph = rg.random()
        age = (t * 1.5 + ph) % 1.0
        x = xs + rg.normal(0, 60) + age * rg.uniform(-200, 200)
        y = y0 - age * rg.uniform(30, 90) + age * age * 80
        r = rg.uniform(1, 3)
        d.ellipse([x - r, y - r, x + r, y + r], fill=int(220 * (1 - age) * strength))
    return arr(im.filter(ImageFilter.GaussianBlur(0.8)))[..., None] * np.array([1.3, 1.35, 1.4])


def S4a(lt, T):
    z = 1250 + lt * 95
    y = Wm.WL + 26 + 4 * math.sin(lt * 1.3)
    cam = river_cam(z, y, side=-30, yaw=1.0, look_ahead=60, fov=64, roll=0.06 * math.sin(lt * 0.9))
    P = Wm.params(Wm.SUN, mist=0.0025, time=T)
    hdr = render_world(cam, P)
    t = twos(lt)
    fl = flap_val(t, 0.42)
    p = dict(FLY, x=640, y=330 + 22 * math.sin(lt * 2.1) + 8 * fl, scale=78, facing=-1, wing_flap=fl,
             body_angle=0.05 * math.sin(lt * 2), neck_angle=-0.45, head_angle=0.05, look=(0.6, 0.1), smile=0.6,
             mouth=0.25, ear=0.9, tail_wave=0.6, tail_phase=t * 7, light=(-0.3, -0.9), ss=2, lw=1.4, pal='gold',
             blush=0.45)
    hdr, A, lay = dragon_hdr(hdr, p, tint=(0.92, 0.9, 0.95), rim_amt=0.8)
    extra = 0
    return finish3(hdr, cam, rays=0.3, extra=extra), 1.0


def S4b(lt, T):
    z = 1800 + lt * 120
    y = Wm.WL + 16 + 30 * sstep(3.8, 5.0, lt)
    pitch = 0.05 + 0.25 * sstep(3.6, 5.0, lt)
    cam = river_cam(z, y, side=0, look_ahead=120, pitch=pitch, fov=72, roll=0.1 * math.sin(lt * 1.6))
    P = Wm.params(Wm.SUN, mist=0.0025, time=T)
    hdr = render_world(cam, P)
    t = twos(lt)
    fl = flap_val(t, 0.4)
    climb = sstep(3.6, 5.0, lt)
    p = dict(FLY, x=640 + 120 * math.sin(lt * 1.4), y=lerp(330, 180, climb) + 6 * fl, scale=lerp(30, 34, climb),
             wing_flap=fl, body_angle=lerp(0.0, -0.9, climb), tail_phase=t * 8, light=(0.0, -1), ss=2, lw=1.0,
             pal='gold', facing=1 if math.cos(lt * 1.4) > 0 else -1)
    hdr, A, lay = dragon_hdr(hdr, p, tint=(0.85, 0.82, 0.9), rim_amt=1.0)
    extra = spray(lt, A['tail_tip'][0], 620, 70, 5, 0.6 * (1 - climb))
    return finish3(hdr, cam, rays=0.5, extra=extra), 1.0


def S5(lt, T):
    # 0-2.4 climbing under the cloud deck; 2.4-3.4 whiteout; 3.4-6 burst above the sea of clouds
    below = lt < 2.9
    if below:
        u = lt / 2.9
        z = 2500 + lt * 60
        y = lerp(80, 140, u)
        cam = river_cam(z, y, side=20, look_ahead=100, pitch=lerp(0.35, 0.6, u), fov=70)
        P = Wm.params(Wm.SUN, mist=0.002, time=T, clouds=1.0, cloud_y=170, cloud_s=0.6, cloud_th=0.3)
    else:
        u = (lt - 2.9) / 3.1
        z = 2700 + u * 120
        y = lerp(250, 300, ease_out(u))
        cam = Wm.Cam((rx(z) - 100, y, z), (rx(z) - 100 + 0.3 * 600, y + lerp(120, 0, ease_out(u)), z + 600), fov=66)
        P = Wm.params(Wm.SUN, mist=0.0, time=T, slab=1.0, slab_y=140, slab_th=110, slab_s=0.3)
    hdr = render_world(cam, P)
    t = twos(lt)
    wo = sstep(2.1, 2.9, lt) * (1 - sstep(2.9, 3.9, lt))
    fl = flap_val(t, 0.4)
    if lt > 3.0:
        b = ease_out(clamp((lt - 3.0) / 1.2))
        p = dict(FLY, x=lerp(560, 700, b), y=lerp(700, 330, b), scale=60, wing_flap=fl, body_angle=lerp(-1.25, -0.3, b),
                 neck_angle=-0.7, smile=0.7, mouth=0.3, ear=0.9, tail_phase=t * 6, light=(0.4, -0.9), ss=2, lw=1.4,
                 pal='gold')
        hdr, A, lay = dragon_hdr(hdr, p, tint=(0.95, 0.92, 0.95))
    # whiteout + cloud wisps
    white = np.array([2.4, 2.3, 2.2])
    hdr = hdr * (1 - wo) + white * wo
    rg = np.random.default_rng(11)
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    wk = sstep(1.6, 2.6, lt) * (1 - sstep(3.3, 3.9, lt))
    for k in range(18):
        x = rg.uniform(-100, W + 100)
        y0 = rg.uniform(0, H)
        y = (y0 + lt * rg.uniform(600, 1200)) % (H + 400) - 200
        r = rg.uniform(60, 180)
        d.ellipse([x - r * 1.4, y - r, x + r * 1.4, y + r], fill=int(200 * wk))
    wisp = arr(im.filter(ImageFilter.GaussianBlur(30)))[..., None]
    hdr = hdr * (1 - wisp * 0.8) + wisp * 0.8 * white
    return finish3(hdr, cam, rays=0.6, exposure=0.85 if lt > 2.9 else 1.0, sat=1.22), 1.0


def S6(lt, T):
    u = lt / 6
    z = 3000 + lt * 40
    x = rx(3000) - 300 + lt * 25
    cam = Wm.Cam((x, 330, z), (x + 0.3 * 600 + lt * 10, 250, z + 600), fov=64, roll=-0.03)
    P = Wm.params(Wm.SUN, mist=0.0, time=T, slab=1.0, slab_y=140, slab_th=110, slab_s=0.3)
    hdr = bg_cache('s6', cam, P, 0) if False else render_world(cam, P)
    t = twos(lt)
    ign = ease_io(clamp((lt - 1.5) / 2.5))
    # spirit dragons rising from the cloud sea
    light = 0
    for k, (sx_, sy_, sc, fc, ph) in enumerate([(220, 470, 40, 1, 0.0), (1040, 420, 52, -1, 1.1), (420, 560, 30, 1, 2.0),
                                                (900, 600, 34, -1, 2.7)]):
        e = clamp((lt - 0.4 - k * 0.5) / 1.6)
        if e <= 0:
            continue
        pp = dict(MEM, x=sx_ + fc * lt * 18, y=sy_ - ease_out(e) * 160 + 6 * math.sin(lt + ph), scale=sc, facing=fc,
                  wing_flap=math.sin(lt * 1.6 + ph) * 0.9, tail_phase=lt * 1.5 + ph, body_angle=-0.35)
        light = light + spirit_light(pp, 0.3 * e)
    fl = flap_val(t, 0.7)
    dx_, dy_ = lerp(280, 640, ease_io(u)), lerp(470, 420, ease_io(u))
    p = dict(FLY, x=dx_, y=dy_ + 8 * fl, scale=58, wing_flap=fl, body_angle=-0.12, mark=ign, smile=0.6, mouth=0.2,
             ear=0.7, tail_phase=t * 4, light=(0.6, -0.6), ss=2, lw=1.4, pal='gold')
    hdr, A, lay = dragon_hdr(hdr, p, tint=(0.95, 0.9, 0.92), glow=0.15 * ign)
    if ign > 0:
        al = lay.split()[3]
        wg = arr(al.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(5)).resize((W, H)))
        light = light + wg[..., None] * np.array([2.2, 1.3, 0.5]) * ign * 1.2
        tip = A['wing_tip']
        light = light + embers(lt, lambda t0, k: (tip[0] - 40 + (k % 5) * 18 - (lt - t0) * 0, tip[1] + (k % 7) * 10), 90, 3,
                               1.4) * ign
    return finish3(hdr, cam, rays=0.6, extra=light, warm=0.3, exposure=0.78, sat=1.25), 1.0


@cache
def title_layer():
    w, h = W, H
    im = Image.new('L', (w * 2, h * 2), 0)
    d = ImageDraw.Draw(im)
    fb = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Black.ttc'
    F1 = ImageFont.truetype(fb, 210, index=0)
    F2 = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc', 46, index=0)
    t1 = "EMBERWING"
    tw = d.textlength(t1, font=F1)
    d.text(((w * 2 - tw) / 2, 120), t1, font=F1, fill=255)
    t2 = "fly toward the light"
    tw2 = d.textlength(t2, font=F2)
    d.text(((w * 2 - tw2) / 2, 400), t2, font=F2, fill=200)
    a = np.asarray(im, np.float32) / 255
    hh, ww = a.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    dx = (np.asarray(Image.fromarray(np.uint8(fbm_img(ww, hh, 18, 3, 5) * 255)), np.float32) / 255 - 0.5) * 6
    a = ndimage.map_coordinates(a, [yy + dx, xx + dx * 0.6], order=1)
    dry = fbm_img(ww, hh, 3, 2, 9)
    a = a * (0.7 + 0.3 * np.clip((dry - 0.3) * 3, 0, 1))
    a = np.asarray(Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).resize((w, h), Image.LANCZOS), np.float32) / 255
    return a


def S7(lt, T):
    u = lt / 6
    z = 3300
    x = rx(3000) - 120
    pull = ease_io(u)
    cam = Wm.Cam((x - 40 * pull, 330 + 20 * pull, z - 120 * pull), (x + 0.3 * 700, 330 + 0.3 * 700 * 0.62 + 20 * pull, z + 700),
                 fov=58 + 8 * pull)
    P = Wm.params(Wm.SUN, mist=0.0, time=T, slab=1.0, slab_y=140, slab_th=110, slab_s=0.3)
    hdr = render_world(cam, P)
    sp = sun_screen(cam)
    t = twos(lt)
    fl = flap_val(t, 1.1)
    sx_, sy_ = sp if sp else (640, 260)
    p = dict(FLY, x=sx_ - 40, y=sy_ + 40 + 6 * fl, scale=lerp(70, 54, pull), wing_flap=0.65 + 0.35 * fl, body_angle=-0.2, mark=1.0,
             ear=0.6, tail_phase=t * 3, light=(0.0, -1), ss=2, lw=1.5, pal='gold', smile=0.5)
    lay, A = dragon_layer(p, tint=(0.35, 0.28, 0.32), rim=(255, 220, 150), rim_amt=1.0, grad=0.1)
    hdr = add_layer_hdr(hdr, lay, 1.0)
    al = lay.split()[3]
    wg = arr(al.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(6)).resize((W, H)))
    light = wg[..., None] * np.array([2.0, 1.2, 0.45]) * 0.35
    tip = A['wing_tip']
    light = light + embers(lt, lambda t0, k: (tip[0] - 60 + (k % 6) * 20, tip[1] + (k % 5) * 14), 110, 9, 1.6)
    ta = clamp((lt - 1.6) / 1.4)
    if ta > 0:
        a = title_layer()
        rev = clamp((ta * 1.3 - np.abs(np.arange(W) - W / 2) / (W / 2)) / 0.25)[None, :] * ta
        tl = a * rev
        tl = np.roll(tl, 300, axis=0)
        g = arr(Image.fromarray(np.uint8(np.clip(tl, 0, 1) * 255)).resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(4)).resize((W, H)))
        hdr = hdr * (1 - 0.55 * np.clip(g * 1.5, 0, 1))[..., None]
        hdr = hdr * (1 - tl[..., None]) + tl[..., None] * np.array([3.2, 2.0, 0.75])
        light = light + g[..., None] * np.array([1.2, 0.6, 0.15]) * 0.8
    img = finish3(hdr, cam, rays=0.8, extra=light, warm=0.4, exposure=0.8, sat=1.25)
    fade = 1 - sstep(4.9, 6.0, lt)
    return img, fade


SHOTS = [('S1', 0.0, 6.0), ('S2', 6.0, 10.0), ('S3', 10.0, 15.0), ('S4a', 15.0, 20.0), ('S4b', 20.0, 25.0),
         ('S5', 25.0, 31.0), ('S6', 31.0, 37.0), ('S7', 37.0, 43.0)]
