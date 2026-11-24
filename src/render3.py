import sys, os, subprocess, time
import numpy as np
import film3
FPS = 24
OUT = 'seg3'
def render_shot(name, s, e):
    os.makedirs(OUT, exist_ok=True)
    fn = getattr(film3, name)
    tmp, final = f'{OUT}/{name}.tmp.mp4', f'{OUT}/{name}.mp4'
    if os.path.exists(final):
        return
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16',
                          '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
    t0 = time.time()
    for fi in range(int(round(s * FPS)), int(round(e * FPS))):
        T = fi / FPS
        img, fade = fn(T - s, T)
        a = np.asarray(img.convert('RGB'), np.float32) * fade
        p.stdin.write(np.uint8(np.clip(a, 0, 255)).tobytes())
    p.stdin.close()
    if p.wait() == 0:
        os.replace(tmp, final)
    print(name, 'done', round(time.time() - t0, 1), flush=True)
if __name__ == '__main__':
    for name, s, e in film3.SHOTS:
        render_shot(name, s, e)
