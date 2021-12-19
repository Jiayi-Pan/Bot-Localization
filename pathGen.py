# %%
import numpy as np

# %%
KeyPos = [[-6.5, 4], [-6.5, -7], [1, -7],
          [1, 0.5], [6.5, 0.5], [6.5, 4.5], [14, 4.5]]

# %%
# Path with Noise
STEP_SIZE = 0.25
NOISE_SCALE = 0.1
Path = []
Noise_2 = [0, 0]
for i in range(len(KeyPos)-1):
    s = KeyPos[i]
    e = KeyPos[i+1]
    isY = True  # check if going in y direction
    if (s[1]-e[1] == 0):
        isY = False
    idx = 1 if isY else 0
    Negidx = 0 if isY else 1
    dL = e[idx] - s[idx]
    n = int(np.ceil(np.abs(dL)/STEP_SIZE))
    pos = 1 if dL > 0 else -1  # if going in positive direction
    for i in range(n):
        step = s.copy()
        step.append(0.05)  # height
        step[idx] += i * STEP_SIZE * pos
        if (i-4) < n:
            if (i % 4 == 1) or (i % 4 == 0):
                noise = np.random.normal(0, NOISE_SCALE)
                Noise_2[i % 4] = noise
                step[Negidx] += noise
            else:
                step[Negidx] -= Noise_2[i % 4-2]
        Path.append(step)
print(Path)

# %%
