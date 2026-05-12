import json
import math
import random

import numpy as np

INPUT_SIZE = 8
HIDDEN_SIZE = 12
OUTPUT_SIZE = 2


def tanh(x):
    return np.tanh(x)


class FishBrain:
    def __init__(self, weights = None):
        n_w1 = INPUT_SIZE * HIDDEN_SIZE
        n_b1 = HIDDEN_SIZE
        n_w2 = HIDDEN_SIZE * OUTPUT_SIZE
        n_b2 = OUTPUT_SIZE
        self.n_params = n_w1 + n_b1 + n_w2 + n_b2

        if weights is None:
            weights = [random.uniform(-1, 1) for _ in range(self.n_params)]

        w = np.array(weights, dtype=np.float32)
        i = 0
        self.W1 = w[i: i + n_w1].reshape(INPUT_SIZE, HIDDEN_SIZE)
        i += n_w1
        self.b1 = w[i: i + n_b1]
        i += n_b1
        self.W2 = w[i: i + n_w2].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
        i += n_w2
        self.b2 = w[i: i + n_b2]

    def forward(self, inputs):
        x = np.array(inputs, dtype=np.float32)
        x = tanh(x @ self.W1 + self.b1)
        x = tanh(x @ self.W2 + self.b2)
        return float(x[0]), float(x[1])

    def get_weights(self):
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
        ]).tolist()

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'weights': self.get_weights()}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data['weights'])


SPEED_MAX = 0.8

POP_SIZE = 80
GENERATIONS = 900
ELITE_FRACTION = 0.2
MUTATION_RATE = 0.15
MUTATION_STD = 0.3
SIM_STEPS = 300


def _make_fish(x=None, y=None):
    return {
        'x': x if x is not None else random.uniform(10, 90),
        'y': y if y is not None else random.uniform(10, 90),
        'vx': random.uniform(-0.4, 0.4),
        'vy': random.uniform(-0.4, 0.4),
        'hp': 100.0,
    }


def _build_inputs(fish, angry_list):
    if angry_list:
        dists = [(math.hypot(fish['x'] - a['x'], fish['y'] - a['y']), a)
                 for a in angry_list]
        d, nearest = min(dists, key=lambda t: t[0])
        dx = (nearest['x'] - fish['x']) / 100.0
        dy = (nearest['y'] - fish['y']) / 100.0
        dn = d / 100.0
    else:
        dx, dy, dn = 0.0, 0.0, 1.0

    return [
        dx,
        dy,
        dn,
        (fish['x'] - 50) / 50.0,
        (fish['y'] - 50) / 50.0,
        fish['vx'] / SPEED_MAX,
        fish['vy'] / SPEED_MAX,
        fish['hp'] / 100.0,
    ]


def _simulate(brain):
    n_peaceful = 3
    n_angry = 2
    peaceful = [_make_fish() for _ in range(n_peaceful)]
    angry = [_make_fish() for _ in range(n_angry)]

    TOUCH_R = 2.5
    HUNT_SPD = 0.65
    fitness = 0.0

    for _ in range(SIM_STEPS):
        for a in angry:
            if not peaceful:
                break
            target = min(peaceful, key=lambda p: math.hypot(p['x'] - a['x'], p['y'] - a['y']))
            d = math.hypot(target['x'] - a['x'], target['y'] - a['y']) or 1e-9
            a['vx'] = ((target['x'] - a['x']) / d) * HUNT_SPD
            a['vy'] = ((target['y'] - a['y']) / d) * HUNT_SPD
            a['x'] = max(0, min(100, a['x'] + a['vx']))
            a['y'] = max(0, min(100, a['y'] + a['vy']))

        for p in peaceful:
            inputs = _build_inputs(p, angry)
            dvx, dvy = brain.forward(inputs)

            p['vx'] = 0.7 * p['vx'] + 0.3 * dvx * SPEED_MAX
            p['vy'] = 0.7 * p['vy'] + 0.3 * dvy * SPEED_MAX
            p['x'] = max(0, min(100, p['x'] + p['vx']))
            p['y'] = max(0, min(100, p['y'] + p['vy']))

        still_alive = []
        for p in peaceful:
            for a in angry:
                if math.hypot(p['x'] - a['x'], p['y'] - a['y']) <= TOUCH_R:
                    p['hp'] -= 25
                    fitness -= 50
                    break
            if p['hp'] > 0:
                if angry:
                    nearest_d = min(math.hypot(p['x'] - a['x'], p['y'] - a['y']) for a in angry)
                    fitness += nearest_d / 100.0
                centre_d = math.hypot(p['x'] - 50, p['y'] - 50)
                fitness -= centre_d / 200.0
                fitness += 1.0
                still_alive.append(p)

        peaceful = still_alive
        if not peaceful:
            break

    return fitness


def _mutate(weights):
    return [
        w + random.gauss(0, MUTATION_STD) if random.random() < MUTATION_RATE else w
        for w in weights
    ]


def _crossover(a, b):
    cut = random.randint(0, len(a))
    return a[:cut] + b[cut:]


def train(save_path = 'ml/brain.json'):
    print(f'Training: pop={POP_SIZE}, generations={GENERATIONS}')
    population = [FishBrain() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        scored = sorted(
            [(brain, _simulate(brain)) for brain in population],
            key=lambda t: t[1], reverse=True
        )
        best_score = scored[0][1]

        if gen % 20 == 0:
            print(f'  gen {gen:4d}  best={best_score:.1f}')

        n_elite = max(1, int(POP_SIZE * ELITE_FRACTION))
        elites = [b for b, _ in scored[:n_elite]]
        new_pop = list(elites)

        while len(new_pop) < POP_SIZE:
            a, b = random.sample(elites, 2)
            child_w = _crossover(a.get_weights(), b.get_weights())
            child_w = _mutate(child_w)
            new_pop.append(FishBrain(child_w))

        population = new_pop

    best = scored[0][0]
    best.save(save_path)
    print(f'Done. Saved to {save_path}')
    return best