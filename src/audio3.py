"""Original epic score + sound for EMBERWING (synthesized)."""
import math
import numpy as np
import wave
from scipy import signal
import audio as A
from audio import (SR, midi, nm, tt, lp, hp, bp, noise, piano, bell, strings, choir, brass, taiko, timp_roll,
                   cymbal_swell, wind, breath, wing_flap, gust, shimmer, magic_swell, reverb, fall_whoosh,
                   mega_wingbeat, flame_sfx)

A.DUR = 43.0
A.N = N = int(SR * A.DUR)
add = A.add
rng = np.random.default_rng(21)
music = np.zeros((2, N), np.float32)
sfx = np.zeros((2, N), np.float32)
amb = np.zeros((2, N), np.float32)

HERO = [('D5', 1), ('A4', .5), ('D5', .5), ('E5', 1), ('F5', 1), ('E5', 1.5), ('D5', .5), ('C5', 1), ('A4', 1), ('D5', 3)]
HERO_MAJ = [('D5', 1), ('A4', .5), ('D5', .5), ('E5', 1), ('F#5', 1), ('A5', 2), ('G5', 1), ('F#5', 1), ('D5', 3)]


def mel(buf, t0, notes, inst, vel, pan=0.0, beat=0.45, oct=0, bright=3600):
    t = t0
    for n, b in notes:
        f = midi(nm(n) + 12 * oct)
        d = b * beat
        x = {'piano': lambda: piano(f, d + 0.5, vel), 'strings': lambda: strings(f, d * 1.02, vel, attack=0.08, release=0.4, bright=bright),
             'brass': lambda: brass(f, d * 0.98, vel), 'bell': lambda: bell(f, d + 2, vel)}[inst]()
        add(buf, t, x, pan)
        t += d
    return t


def chord(buf, t0, names, d, inst, vel, **kw):
    for i, n in enumerate(names):
        f = midi(nm(n))
        p = (i / max(1, len(names) - 1) - 0.5) * 0.9
        x = {'strings': lambda: strings(f, d, vel, **kw), 'choir': lambda: choir(f, d, vel, **kw),
             'brass': lambda: brass(f, d, vel, **kw), 'piano': lambda: piano(f, d, vel)}[inst]()
        add(buf, t0, x, p)


def hit(t0, vel=1.0):
    add(music, t0, taiko(2.2, 1.3 * vel, 42), 0.0)
    add(music, t0, taiko(1.5, 0.8 * vel, 70), 0.0)
    x = lp(noise(2.5), 120, 4) * np.exp(-tt(2.5) / 0.6) * 3 * vel
    add(sfx, t0, x.astype(np.float32), 0.0)
    add(music, t0, hp(noise(2.0), 4000) * np.exp(-tt(2.0) / 0.8) * 0.12 * vel, 0.0)


def riser(t0, d, vel=0.6):
    t = tt(d)
    f = 200 * 2 ** (t / d * 3)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / d) ** 2 * 0.12
    x = x + hp(noise(d), 2000) * (t / d) ** 3 * 0.25
    add(music, t0, (x * vel).astype(np.float32), 0.0)


def ostinato(t0, t1, notes, step, vel, bright=1600, pan=-0.2):
    t, i = t0, 0
    while t < t1:
        n = notes[i % len(notes)]
        add(music, t, strings(midi(nm(n)), step * 0.7, vel, attack=0.005, release=0.08, bright=bright, voices=2), pan)
        add(music, t, strings(midi(nm(n)) * 2, step * 0.7, vel * 0.5, attack=0.005, release=0.08, bright=bright, voices=2), -pan)
        t += step
        i += 1


def build():
    # ---------------- S1 0-6 dawn
    add(amb, 0.0, wind(10.0, 0.4, 31, 150, 700) * np.minimum(1, tt(10.0) / 2.0), 0.0)
    t = tt(7.0)
    drone = (np.sin(2 * np.pi * midi(nm('D2')) * t) + 0.5 * np.sin(2 * np.pi * midi(nm('A2')) * t)) * np.minimum(1, t / 3) * 0.05
    add(music, 0.0, drone.astype(np.float32), 0.0)
    chord(music, 0.5, ('D3', 'A3', 'F4'), 5.6, 'choir', 0.28, attack=3.0, release=1.0)
    chord(music, 0.0, ('D2', 'A2', 'D3'), 6.2, 'strings', 0.12, attack=3.0, release=0.6, bright=1200)
    mel(music, 3.0, [('A4', 2), ('D5', 1), ('E5', 1), ('F5', 3)], 'brass', 0.28, 0.2, beat=0.45)
    add(sfx, 4.3, shimmer(2.0, 0.2, 1760), 0.3)
    # ---------------- S2 6-10 the peak
    mel(music, 6.1, HERO[:6], 'piano', 0.5, 0.1, beat=0.45)
    ostinato(6.0, 10.0, ['D3', 'D3', 'A2', 'D3'], 0.225, 0.22, 1400)
    for k in range(8):
        add(music, 6.0 + k * 0.45, taiko(0.6, 0.25 + 0.04 * k, 85), 0.0)
    chord(music, 6.0, ('D3', 'F3', 'A3', 'D4'), 4.0, 'strings', 0.18, attack=1.5, release=0.3, bright=2000)
    add(amb, 6.0, wind(4.0, 0.55, 33, 300, 2500), 0.0)
    add(sfx, 9.0, breath(0.5, 0.3, True), 0.0)
    riser(8.4, 1.6, 0.8)
    add(music, 8.4, timp_roll(1.6, 0.5, 73), 0.0)
    # ---------------- S3 10-15 leap + dive
    add(sfx, 10.0, gust(1.0, 0.5), 0.0)
    add(sfx, 10.1, fall_whoosh(3.4, 1.0), 0.0)
    ostinato(10.0, 13.4, ['D2', 'D2', 'F2', 'D2', 'G2', 'D2', 'A2', 'C3'], 0.15, 0.3, 1600)
    riser(10.6, 2.85, 1.0)
    add(music, 11.5, cymbal_swell(1.95, 0.8), 0.0)
    chord(music, 10.0, ('D2', 'A2', 'D3', 'F3'), 3.45, 'strings', 0.22, attack=2.5, release=0.1, bright=2400)
    hit(13.45, 1.2)
    add(sfx, 13.45, mega_wingbeat(0.9), 0.0)
    # ---------------- S4 15-25 canyon run (driving)
    t0 = 13.45
    ostinato(t0, 23.6, ['D2', 'D2', 'D3', 'D2', 'F2', 'D2', 'A2', 'D2'], 0.1875, 0.3, 1700)
    tk = t0
    k = 0
    while tk < 23.6:
        v = 0.55 if k % 4 == 0 else 0.3
        add(music, tk, taiko(0.8, v, 60 if k % 4 == 0 else 90), 0.0)
        tk += 0.375
        k += 1
    mel(music, t0 + 0.75, HERO, 'brass', 0.5, 0.0, beat=0.375)
    mel(music, t0 + 0.75, HERO, 'strings', 0.4, 0.25, beat=0.375, oct=1, bright=4200)
    prog = [('D3', 'A3', 'D4', 'F4'), ('A#2', 'F3', 'A#3', 'D4'), ('F2', 'C3', 'F3', 'A3'), ('C3', 'G3', 'C4', 'E4')]
    for i in range(7):
        chd = prog[i % 4]
        chord(music, t0 + i * 1.5, chd, 1.55, 'strings', 0.22, attack=0.1, release=0.3, bright=2600)
        chord(music, t0 + i * 1.5, chd[1:], 1.4, 'choir', 0.22, attack=0.2, release=0.4)
    mel(music, t0 + 6.75, HERO, 'strings', 0.45, 0.25, beat=0.375, oct=1, bright=4200)
    mel(music, t0 + 6.75, HERO, 'piano', 0.45, -0.1, beat=0.375)
    tf = 15.0
    while tf < 25.0:
        add(sfx, tf, wing_flap(0.35, 0.3, 300), 0.0)
        tf += 0.42 if tf < 20 else 0.4
    rush = bp(noise(10.0), 300, 2500) * 0.12 * (0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * tt(10.0)))
    add(amb, 15.0, rush.astype(np.float32), 0.0)
    water = bp(noise(5.0), 800, 6000) * 0.08
    add(amb, 20.0, water.astype(np.float32), 0.0)
    riser(23.6, 3.5, 1.0)
    add(music, 23.6, timp_roll(3.5, 0.6, 73), 0.0)
    chord(music, 23.6, ('A2', 'E3', 'A3', 'C#4', 'E4'), 3.5, 'strings', 0.26, attack=2.0, release=0.1, bright=3000,
          cresc=lambda t: 0.3 + 0.7 * np.minimum(1, t / 3.5))
    # ---------------- S5 25-31 through the clouds -> burst at 28.0
    add(music, 25.5, cymbal_swell(2.4, 1.0), 0.0)
    hush = bp(noise(1.0), 200, 1500) * 0.15
    add(amb, 27.1, hush.astype(np.float32), 0.0)
    hit(28.0, 1.4)
    add(sfx, 28.0, mega_wingbeat(1.0), 0.0)
    chord(music, 28.0, ('D2', 'A2', 'D3', 'F#3', 'A3', 'D4'), 3.0, 'strings', 0.36, attack=0.05, release=0.6, bright=3500)
    chord(music, 28.0, ('D4', 'F#4', 'A4', 'D5'), 3.0, 'choir', 0.5, attack=0.15, release=0.8)
    chord(music, 28.0, ('D3', 'A3', 'F#4'), 2.8, 'brass', 0.5, attack=0.05, release=0.5)
    mel(music, 28.0, [('A5', 1), ('D6', 1), ('F#6', 2)], 'bell', 0.3, 0.3, beat=0.5)
    add(sfx, 28.4, shimmer(3.0, 0.4, 1760), 0.0)
    # ---------------- S6 31-37 above the world (D major theme)
    t0 = 31.0
    mel(music, t0, HERO_MAJ, 'brass', 0.55, 0.0, beat=0.5)
    mel(music, t0, HERO_MAJ, 'strings', 0.5, 0.25, beat=0.5, oct=1, bright=4400)
    mel(music, t0, HERO_MAJ, 'piano', 0.45, -0.15, beat=0.5)
    progM = [('D2', 'A2', 'D3', 'F#3', 'A3'), ('B1', 'F#2', 'B2', 'D3', 'F#3'), ('G1', 'D2', 'G2', 'B2', 'D3'),
             ('A1', 'E2', 'A2', 'C#3', 'E3')]
    for i, chd in enumerate(progM):
        tc = t0 + i * 1.5
        chord(music, tc, chd, 1.6, 'strings', 0.32, attack=0.1, release=0.5, bright=3200)
        chord(music, tc, [c[:-1] + str(int(c[-1]) + 2) for c in chd[2:]], 1.6, 'choir', 0.42, attack=0.2, release=0.6)
        add(music, tc, taiko(1.5, 0.75, 52), 0.0)
        add(music, tc + 0.75, taiko(1.0, 0.45, 75), 0.0)
    add(sfx, 31.5, shimmer(4.0, 0.3, 1568), -0.4)
    add(sfx, 33.0, flame_sfx(0.5), 0.2)
    add(sfx, 33.5, shimmer(3.5, 0.3, 2093), 0.4)
    tf = 31.0
    while tf < 37.0:
        add(sfx, tf, wing_flap(0.25, 0.4, 200), 0.0)
        tf += 0.7
    # ---------------- S7 37-43 EMBERWING
    hit(37.0, 1.0)
    chord(music, 37.0, ('D2', 'A2', 'D3', 'F#3', 'A3', 'D4', 'F#4'), 4.2, 'strings', 0.34, attack=0.1, release=1.8, bright=3200)
    chord(music, 37.0, ('A4', 'D5', 'F#5'), 4.2, 'choir', 0.45, attack=0.4, release=1.8)
    chord(music, 37.0, ('D3', 'A3', 'D4'), 3.0, 'brass', 0.45, attack=0.08, release=1.2)
    hit(38.6, 1.1)
    add(sfx, 38.6, magic_swell(0.01, 0.0), 0.0)
    mel(music, 38.6, [('D6', 1), ('A6', 1), ('D7', 2)], 'bell', 0.35, 0.0, beat=0.45)
    add(sfx, 38.8, shimmer(3.5, 0.35, 1760), 0.0)
    add(amb, 37.0, wind(6.0, 0.25, 35, 200, 900), 0.0)
    add(music, 41.2, piano(midi(nm('D5')), 1.6, 0.5), 0.0)
    add(music, 41.2, piano(midi(nm('A4')), 1.6, 0.25), 0.0)
    add(music, 41.2, piano(midi(nm('D4')), 1.6, 0.2), 0.0)


def main():
    build()
    m = reverb(music, 2.8, 0.3)
    s = reverb(sfx, 1.4, 0.18, seed=5)
    mix = m + s * 0.85 + amb * 1.1
    end = int(42.3 * SR)
    mix[:, end:] *= np.linspace(1, 0, N - end)[None, :]
    st = int(0.3 * SR)
    mix[:, :st] *= np.linspace(0, 1, st)[None, :]
    mix = mix / np.max(np.abs(mix)) * 0.95
    env = np.abs(mix).max(0)
    env = signal.filtfilt(*signal.butter(1, 8 / (SR / 2)), env)
    g = np.where(env > 0.45, 0.45 / np.maximum(env, 1e-6) * (env / 0.45) ** 0.35, 1.0)
    mix = mix * g[None, :]
    mix = mix / np.max(np.abs(mix)) * 0.9
    pcm = np.int16(np.clip(mix.T, -1, 1) * 32767)
    with wave.open('score3.wav', 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print('ok')


if __name__ == '__main__':
    main()
