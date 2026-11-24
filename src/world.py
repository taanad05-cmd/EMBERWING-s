"""EMBERWING - a procedural 3D world: heightfield raymarcher (numba) with baked sun lighting,
valley mist, a river canyon with reflections, a cloud sea and a sunrise sky."""
import math
import os
import numpy as np
from numba import njit, prange
from scipy import ndimage

N = 2048            # heightmap cells (world units), tiles seamlessly
M = 1024            # cloud / wave textures
WL = 34.0           # water level
CACHE = 'world_cache.npz'


# ------------------------------------------------------------------ periodic noise

def spectral(n, seed, fmin, fmax, beta=2.0):
    r = np.random.default_rng(seed)
    w = r.standard_normal((n, n))
    F = np.fft.fft2(w)
    fx = np.fft.fftfreq(n) * n
    f = np.sqrt(fx[None, :] ** 2 + fx[:, None] ** 2)
    amp = np.where((f >= fmin) & (f < fmax), (np.maximum(f, 1)) ** (-beta / 2), 0)
    x = np.real(np.fft.ifft2(F * amp))
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def river_x(z):
    a = 2 * np.pi * z / N
    return N * 0.5 + 230 * np.sin(2 * a) + 80 * np.sin(5 * a + 1.0) + 30 * np.sin(11 * a + 2.0)


def build_height():
    n = N
    warpx = spectral(n, 11, 1, 6) * 40
    warpz = spectral(n, 12, 1, 6) * 40
    low = spectral(n, 1, 1, 3, 2.4)
    mid = spectral(n, 2, 2, 8, 2.2)
    hi = spectral(n, 3, 8, 40, 2.0)
    vhi = spectral(n, 4, 70, 300, 2.0)
    ridge = lambda a: (1 - np.abs(np.tanh(a * 0.8))) ** 2
    h = 0.55 * (low * 0.5 + 0.5) + 0.55 * ridge(mid) * (0.6 + 0.4 * (low * 0.5 + 0.5))
    h = h + 0.10 * ridge(hi) * h + 0.012 * vhi * h
    # domain warp (periodic)
    zz, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    h = ndimage.map_coordinates(h, [zz + warpz, xx + warpx], order=1, mode='grid-wrap')
    h = np.clip(h, 0, None)
    h = h / np.percentile(h, 99.7)
    h = np.clip(h, 0, 1.15) ** 1.7
    h = ndimage.gaussian_filter(h, 2.0, mode='wrap')
    H = 38 + h * 250
    # carve the river canyon (periodic in z)
    xr = river_x(zz)
    d = np.abs(((xx - xr + n / 2) % n) - n / 2)
    floor = WL - 7 + d * 0.06
    s = np.clip((d - 18) / 150, 0, 1)
    s = s * s * (3 - 2 * s)
    H = floor + (H - floor) * s
    # some terracing / strata on canyon walls
    H = H + np.sin(H * 0.25) * 1.5 * (1 - s) * s * 4
    return H.astype(np.float32)


@njit(cache=True, fastmath=True)
def samp(A, x, z):
    n = A.shape[0]
    fx = math.floor(x)
    fz = math.floor(z)
    tx = x - fx
    tz = z - fz
    ix = int(fx) % n
    iz = int(fz) % n
    ix1 = (ix + 1) % n
    iz1 = (iz + 1) % n
    a = A[iz, ix] * (1 - tx) + A[iz, ix1] * tx
    b = A[iz1, ix] * (1 - tx) + A[iz1, ix1] * tx
    return a * (1 - tz) + b * tz


@njit(cache=True, fastmath=True)
def samp3(A, x, z, out):
    n = A.shape[0]
    fx = math.floor(x)
    fz = math.floor(z)
    tx = x - fx
    tz = z - fz
    ix = int(fx) % n
    iz = int(fz) % n
    ix1 = (ix + 1) % n
    iz1 = (iz + 1) % n
    for c in range(3):
        a = A[iz, ix, c] * (1 - tx) + A[iz, ix1, c] * tx
        b = A[iz1, ix, c] * (1 - tx) + A[iz1, ix1, c] * tx
        out[c] = a * (1 - tz) + b * tz


@njit(parallel=True, cache=True, fastmath=True)
def bake_shadow(H, sx, sy, sz):
    n = H.shape[0]
    out = np.ones((n, n), np.float32)
    hl = math.sqrt(sx * sx + sz * sz)
    dx, dz = sx / hl, sz / hl
    slope = sy / hl
    for i in prange(n):
        for j in range(n):
            h0 = H[i, j] + 0.5
            res = 1.0
            t = 1.5
            while t < 900.0:
                x = j + dx * t
                z = i + dz * t
                hh = samp(H, x, z)
                ry = h0 + slope * t
                if hh > ry:
                    res = 0.0
                    break
                k = (ry - hh) / (t * 0.035)
                if k < res:
                    res = k
                if ry > 520:
                    break
                t += max(1.0, t * 0.03)
            out[i, j] = res
    return out


def bake(H, sun):
    n = N
    gx = (np.roll(H, -1, 1) - np.roll(H, 1, 1)) / 2
    gz = (np.roll(H, -1, 0) - np.roll(H, 1, 0)) / 2
    nrm = np.stack([-gx, np.ones_like(H), -gz], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    ny = nrm[..., 1]
    sh = bake_shadow(H, *sun)
    sh = np.clip(sh, 0, 1)
    sh = sh * sh * (3 - 2 * sh)
    blur = ndimage.gaussian_filter(H, 10, mode='wrap')
    ao = np.clip(0.55 + (H - blur) / 45, 0.15, 1.0)
    zz, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    nz = spectral(n, 21, 30, 300) * 0.5 + 0.5
    nz2 = spectral(n, 22, 4, 40) * 0.5 + 0.5
    strata = np.sin(H * 0.18 + nz2 * 9) * 0.5 + 0.5
    rock = np.stack([0.30 + 0.06 * strata, 0.26 + 0.05 * strata, 0.25 + 0.03 * strata], -1) * (0.8 + 0.4 * nz)[..., None]
    rock2 = np.array([0.42, 0.33, 0.27], np.float32)
    rock = rock * (1 - 0.35 * (nz2 > 0.6)[..., None]) + rock2 * 0.35 * (nz2 > 0.6)[..., None]
    grass = np.array([0.20, 0.32, 0.11], np.float32) * (0.8 + 0.4 * nz)[..., None]
    forest = np.array([0.07, 0.15, 0.08], np.float32) * (0.8 + 0.4 * nz2)[..., None]
    snow = np.array([0.93, 0.95, 1.0], np.float32)
    sand = np.array([0.55, 0.48, 0.36], np.float32)

    def sm(a, b, x):
        t = np.clip((x - a) / (b - a), 0, 1)
        return t * t * (3 - 2 * t)
    flat = sm(0.72, 0.9, ny)
    alt = H
    veg = flat * sm(150, 90, alt + nz * 40)
    forest_m = veg * sm(0.45, 0.6, nz2 + 0.1 * nz) * sm(0.8, 0.88, ny)
    snow_m = sm(170, 215, alt + nz * 50) * sm(0.55, 0.75, ny + 0.1 * nz2)
    sand_m = sm(WL + 6, WL + 1, alt)
    alb = rock
    alb = alb * (1 - veg[..., None]) + grass * veg[..., None]
    alb = alb * (1 - forest_m[..., None]) + forest * forest_m[..., None]
    alb = alb * (1 - snow_m[..., None]) + snow * snow_m[..., None]
    alb = alb * (1 - sand_m[..., None]) + sand * sand_m[..., None]
    sun_c = np.array([2.9, 1.9, 1.15], np.float32)
    sky_c = np.array([0.20, 0.28, 0.50], np.float32)
    ndl = np.clip((nrm * np.array(sun)[None, None]).sum(-1), 0, 1)
    light = sun_c[None, None] * (ndl * sh)[..., None] + sky_c[None, None] * (ao * (0.55 + 0.45 * ny))[..., None]
    # warm bounce in canyons
    light += np.array([0.10, 0.06, 0.03]) * (1 - ao)[..., None]
    cav = H - ndimage.gaussian_filter(H, 2.5, mode='wrap')
    cav2 = H - ndimage.gaussian_filter(H, 6, mode='wrap')
    det = np.clip(1 + cav * 0.09 + cav2 * 0.02, 0.45, 1.5)
    col = (alb * light * det[..., None]).astype(np.float32)
    # forest canopy texture: darken in clumps
    return col, sh.astype(np.float32)


def cloud_textures():
    m = M
    c = spectral(m, 31, 2, 10, 2.0) * 0.6 + spectral(m, 32, 10, 60, 2.0) * 0.3 + spectral(m, 33, 60, 250) * 0.1
    c = (c - c.min()) / (c.max() - c.min())
    s = spectral(m, 41, 2, 9, 2.2) * 0.6 + spectral(m, 42, 9, 32, 2.0) * 0.3 + spectral(m, 43, 32, 120) * 0.1
    s = (s - s.min()) / (s.max() - s.min())
    # puffy tops: rounded billows
    s = np.clip((s - 0.22) / 0.6, 0, 1) ** 0.5
    w1 = spectral(m, 51, 20, 120)
    w2 = spectral(m, 52, 20, 120)
    return c.astype(np.float32), s.astype(np.float32), w1.astype(np.float32), w2.astype(np.float32)


def load(sun):
    key = 'v8_' + '_'.join(f'{v:.3f}' for v in sun)
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if str(d['key']) == key:
            return {k: d[k] for k in d.files}
    H = build_height()
    col, sh = bake(H, sun)
    C, S, W1, W2 = cloud_textures()
    out = dict(H=H, col=col, sh=sh, C=C, S=S, W1=W1, W2=W2, key=np.array(key))
    np.savez(CACHE, **out)
    return out


# ------------------------------------------------------------------ renderer

@njit(cache=True, fastmath=True)
def sky(dx, dy, dz, sx, sy, sz, P, out):
    # P: 0-2 zenith, 3-5 horizon, 6-8 sun col, 9 stars, 10 exposure
    mu = dx * sx + dy * sy + dz * sz
    t = 1.0 - max(dy, 0.0)
    t = t * t * t
    for c in range(3):
        out[c] = P[c] * (1 - t) + P[3 + c] * t
    if dy < 0:
        k = min(1.0, -dy * 6)
        for c in range(3):
            out[c] = out[c] * (1 - 0.5 * k)
    m = max(mu, 0.0)
    g1 = m ** 6 * 0.55
    g2 = m ** 48 * 1.6
    for c in range(3):
        out[c] += P[6 + c] * (g1 + g2)
    if mu > 0.99965:
        for c in range(3):
            out[c] += P[6 + c] * 30.0
    if P[9] > 0 and dy > 0.02:
        # stars: hash on direction grid
        q = 900.0
        ix = math.floor(dx * q)
        iy = math.floor(dy * q)
        iz = math.floor(dz * q)
        hsh = math.sin(ix * 12.9898 + iy * 78.233 + iz * 37.719) * 43758.5453
        hsh = hsh - math.floor(hsh)
        if hsh > 0.9985:
            b = (hsh - 0.9985) / 0.0015 * P[9] * min(1.0, dy * 5)
            out[0] += b * 0.9
            out[1] += b * 0.95
            out[2] += b * 1.1


@njit(cache=True, fastmath=True)
def surf(H, S, x, z, P):
    """effective surface: terrain (with water plane) vs cloud-sea tops. returns (height, kind)"""
    h = samp(H, x, z)
    if h < P[20]:
        h2 = P[20]
        kind = 1
    else:
        h2 = h
        kind = 0
    if P[30] > 0:
        s = samp(S, x * P[33], z * P[33])
        if s > 0.02:
            ct = P[31] + P[32] * s
            if ct > h2:
                return ct, 2
    return h2, kind


@njit(cache=True, fastmath=True)
def march(H, S, ox, oy, oz, dx, dy, dz, tmax, P, steps):
    t = 0.5
    kind = -1
    for i in range(steps):
        x = ox + dx * t
        y = oy + dy * t
        z = oz + dz * t
        h, k = surf(H, S, x, z, P)
        dh = y - h
        if dh < 0.0015 * t:
            # refine
            a = t - max(0.35 * abs(dh), 0.002 * t + 0.25) * 1.2
            b = t
            for j in range(5):
                mtt = 0.5 * (a + b)
                hh, kk = surf(H, S, ox + dx * mtt, oz + dz * mtt, P)
                if oy + dy * mtt - hh < 0:
                    b = mtt
                else:
                    a = mtt
            t = b
            h, kind = surf(H, S, ox + dx * t, oz + dz * t, P)
            return t, kind
        if y > 560 and dy > 0:
            return 1e9, -1
        t += max(0.22 * dh, 0.0016 * t + 0.15)
        if t > tmax:
            break
    return 1e9, -1


@njit(cache=True, fastmath=True)
def fog_apply(col, t, oy, dy, mu, P, sk):
    # height fog 1 (atmosphere) + height fog 2 (valley mist)
    dens, hf = P[11], P[12]
    if abs(dy) < 1e-4:
        dy = 1e-4
    f1 = dens * math.exp(-oy * hf) * (1 - math.exp(-t * dy * hf)) / (dy * hf)
    d2, h2, b2 = P[13], P[14], P[15]
    yo = oy - b2
    f2 = d2 * math.exp(-max(yo, -40.0) * h2) * (1 - math.exp(-t * dy * h2)) / (dy * h2)
    a1 = 1 - math.exp(-max(f1, 0.0))
    a2 = 1 - math.exp(-max(f2, 0.0))
    m = max(mu, 0.0)
    g = m ** 8
    for c in range(3):
        fc = sk[c] * 0.85 + P[6 + c] * g * 0.5
        col[c] = col[c] * (1 - a1) + fc * a1
        mc = P[16 + c] + P[6 + c] * g * 0.35
        col[c] = col[c] * (1 - a2) + mc * a2


@njit(cache=True, fastmath=True)
def cloud_plane(C, ox, oy, oz, dx, dy, dz, sx, sy, sz, P, col):
    if dy <= 0.002 or P[21] <= 0:
        return
    tc = (P[22] - oy) / dy
    if tc < 0:
        return
    x = (ox + dx * tc) * P[23] + P[40] * 0.02
    z = (oz + dz * tc) * P[23]
    c = samp(C, x, z)
    a = (c - P[24]) / 0.3
    if a <= 0:
        return
    a = min(a, 1.0) * P[21]
    a *= max(0.0, 1 - tc / 30000.0)
    c2 = samp(C, x + sx * 6, z + sz * 6)
    lit = min(1.0, max(0.0, 0.6 - (c2 - c) * 4))
    mu = dx * sx + dy * sy + dz * sz
    fw = max(mu, 0.0) ** 6
    for k in range(3):
        base = P[3 + k] * 0.7
        cc = base * (0.6 + 0.4 * lit) + P[6 + k] * (0.35 * lit + fw * 0.9 * (1 - a))
        col[k] = col[k] * (1 - a) + cc * a


@njit(parallel=True, cache=True, fastmath=True)
def render(H, COL, S, C, W1, W2, Wd, Ht, cam, P, img):
    ox, oy, oz = cam[0], cam[1], cam[2]
    fx, fy, fz = cam[3], cam[4], cam[5]
    rx, ry, rz = cam[6], cam[7], cam[8]
    ux, uy, uz = cam[9], cam[10], cam[11]
    tf = cam[12]
    sx, sy, sz = P[25], P[26], P[27]
    aspect = Wd / Ht
    for j in prange(Ht):
        sk = np.zeros(3)
        col = np.zeros(3)
        tmp = np.zeros(3)
        for i in range(Wd):
            u = (2 * (i + 0.5) / Wd - 1) * tf * aspect
            v = (1 - 2 * (j + 0.5) / Ht) * tf
            dx = fx + rx * u + ux * v
            dy = fy + ry * u + uy * v
            dz = fz + rz * u + uz * v
            L = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= L
            dy /= L
            dz /= L
            mu = dx * sx + dy * sy + dz * sz
            hdx, hdz = dx, dz
            sky(dx, max(dy, 0.0) * 0 + 0.0, dz, sx, sy, sz, P, sk)  # horizon colour for fog
            t, kind = march(H, S, ox, oy, oz, dx, dy, dz, P[28], P, int(P[29]))
            if kind < 0:
                sky(dx, dy, dz, sx, sy, sz, P, col)
                cloud_plane(C, ox, oy, oz, dx, dy, dz, sx, sy, sz, P, col)
                t = P[28]
                if dy < 0.05:
                    fog_apply(col, t * 0.6, oy, dy, mu, P, sk)
            else:
                px = ox + dx * t
                py = oy + dy * t
                pz = oz + dz * t
                if kind == 0:
                    samp3(COL, px, pz, col)
                    # detail: fine variation
                    dn = samp(C, px * 0.9, pz * 0.9) - 0.5
                    for c in range(3):
                        col[c] *= (1 + dn * 0.35)
                elif kind == 1:
                    # water: rippled reflection
                    wx = samp(W1, px * 0.35 + P[40] * 0.6, pz * 0.35) * 0.06
                    wz = samp(W2, px * 0.35, pz * 0.35 + P[40] * 0.9) * 0.06
                    nx, ny_, nz = -wx, 1.0, -wz
                    nl = math.sqrt(nx * nx + ny_ * ny_ + nz * nz)
                    nx /= nl
                    ny_ /= nl
                    nz /= nl
                    dd = dx * nx + dy * ny_ + dz * nz
                    rdx = dx - 2 * dd * nx
                    rdy = abs(dy - 2 * dd * ny_)
                    rdz = dz - 2 * dd * nz
                    t2, k2 = march(H, S, px, py + 0.6, pz, rdx, rdy, rdz, 1500.0, P, 90)
                    if k2 == 0 or k2 == 2:
                        qx, qz = px + rdx * t2, pz + rdz * t2
                        if k2 == 0:
                            samp3(COL, qx, qz, tmp)
                        else:
                            for c in range(3):
                                tmp[c] = P[3 + c] * 0.9
                        fog_apply(tmp, t2, py, rdy, rdx * sx + rdy * sy + rdz * sz, P, sk)
                    else:
                        sky(rdx, rdy, rdz, sx, sy, sz, P, tmp)
                    fr = 0.03 + 0.97 * (1 - abs(dd)) ** 5
                    fr = min(1.0, fr * 1.4 + 0.25)
                    for c in range(3):
                        col[c] = tmp[c] * fr + (0.015, 0.05, 0.06)[c] * (1 - fr)
                else:
                    # cloud-sea surface
                    e = 1.2 / P[33]
                    s0 = samp(S, px * P[33], pz * P[33])
                    sxp = samp(S, px * P[33] + 1.2, pz * P[33])
                    szp = samp(S, px * P[33], pz * P[33] + 1.2)
                    gx = (sxp - s0) * P[32] / e
                    gz = (szp - s0) * P[32] / e
                    nx, ny_, nz = -gx, 1.0, -gz
                    nl = math.sqrt(nx * nx + ny_ * ny_ + nz * nz)
                    nx /= nl
                    ny_ /= nl
                    nz /= nl
                    ndl = nx * sx + ny_ * sy + nz * sz
                    wrap = max(0.0, (ndl + 0.25) / 1.25) ** 1.8
                    sl = samp(S, px * P[33] + sx * 5.0, pz * P[33] + sz * 5.0)
                    occ = min(1.0, max(0.0, 1.0 - (sl - s0) * 5.0))
                    sss = max(mu, 0.0) ** 5 * (1 - s0) * 1.5
                    amb = 0.5 + 0.5 * ny_
                    for c in range(3):
                        col[c] = (P[6 + c] * 0.75 * wrap * occ + (0.26, 0.28, 0.46)[c] * amb * (0.7 + 0.3 * s0) +
                                  P[6 + c] * sss * 0.6)
                fog_apply(col, t, oy, dy, mu, P, sk)
            ex = P[10]
            for c in range(3):
                img[j, i, c] = col[c] * ex


class Cam:
    def __init__(self, pos, look, up=(0, 1, 0), fov=60.0, roll=0.0):
        p = np.array(pos, float)
        f = np.array(look, float) - p
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array(up, float))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        if roll:
            c, s = math.cos(roll), math.sin(roll)
            r, u = r * c + u * s, -r * s + u * c
        self.v = np.concatenate([p, f, r, u, [math.tan(math.radians(fov) / 2)]]).astype(np.float64)


def params(sun, **kw):
    P = np.zeros(48)
    d = dict(zen=(0.05, 0.13, 0.38), hor=(0.95, 0.62, 0.40), sunc=(1.9, 1.15, 0.6), stars=0.0, exposure=1.0,
             fog=0.00022, fog_h=0.004, mist=0.0022, mist_h=0.07, mist_b=WL + 6, mist_c=(0.42, 0.48, 0.62),
             wl=WL, clouds=1.0, cloud_y=1400.0, cloud_s=0.25, cloud_th=0.55, tmax=4200.0, steps=380,
             slab=0.0, slab_y=250.0, slab_th=90.0, slab_s=0.6, time=0.0)
    d.update(kw)
    P[0:3] = d['zen']
    P[3:6] = d['hor']
    P[6:9] = d['sunc']
    P[9] = d['stars']
    P[10] = d['exposure']
    P[11] = d['fog']
    P[12] = d['fog_h']
    P[13] = d['mist']
    P[14] = d['mist_h']
    P[15] = d['mist_b']
    P[16:19] = d['mist_c']
    P[20] = d['wl']
    P[21] = d['clouds']
    P[22] = d['cloud_y']
    P[23] = d['cloud_s']
    P[24] = d['cloud_th']
    P[25:28] = sun
    P[28] = d['tmax']
    P[29] = d['steps']
    P[30] = d['slab']
    P[31] = d['slab_y']
    P[32] = d['slab_th']
    P[33] = d['slab_s']
    P[40] = d['time']
    return P


SUN = tuple(np.array([0.3, 0.3, 1.0]) / np.linalg.norm([0.3, 0.3, 1.0]))
_W = None


def world():
    global _W
    if _W is None:
        _W = load(SUN)
    return _W


def shoot(cam, P, w=1280, h=720):
    Wd = world()
    img = np.zeros((h, w, 3), np.float32)
    render(Wd['H'], Wd['col'], Wd['S'], Wd['C'], Wd['W1'], Wd['W2'], w, h, cam.v, P, img)
    return img


def tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
