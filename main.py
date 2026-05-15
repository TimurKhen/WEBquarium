import math
import os
import random
import threading
import time
from os import path, makedirs

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from data import db_session
from data.fish import Fish
from data.user import User
from ml.fish_brain import FishBrain
import os
from dotenv import load_dotenv

_BRAIN_PATH = path.join('ml', 'brain.json')
fish_brain: FishBrain | None = None
if os.path.exists(_BRAIN_PATH):
    fish_brain = FishBrain.load(_BRAIN_PATH)
    print(f'Loading brain.json')
else:
    print('No brain.json. Run ml/train.py first.')


def calc_movement(fish, angry_list):
    if angry_list:
        dists = [(math.hypot(fish.x - a.x, fish.y - a.y), a) for a in angry_list]
        d, nearest = min(dists, key=lambda t: t[0])
        dx = (nearest.x - fish.x) / 100.0
        dy = (nearest.y - fish.y) / 100.0
        dn = d / 100.0
    else:
        dx, dy, dn = 0.0, 0.0, 1.0

    return [
        dx, dy, dn,
        (fish.x - 50) / 50.0,
        (fish.y - 50) / 50.0,
        fish.vx / SPEED_MAX,
        fish.vy / SPEED_MAX,
        fish.health / 100.0,
    ]


load_dotenv()

app = Flask(__name__, static_folder='dist/WEBquarium/browser', static_url_path='')
app.config['SECRET_KEY'] = os.getenv("API_KEY")
app.config['UPLOAD_FOLDER'] = path.join('static', 'uploads')
makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_proxy(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')

TICK_INTERVAL = 0.1
SPEED_BASE = 0.5
SPEED_HUNT = 0.7
SPEED_FLEE = 0.6
HUNT_RADIUS = 30.0
FLEE_RADIUS = 31.0
TOUCH_RADIUS = 2.5
HIT_DAMAGE = 25.0
STARVATION_TICKS = 100
STARVATION_DAMAGE = 10.0
SPEED_MIN, SPEED_MAX = 0.2, 0.9

FISH_RADIUS = 2.0

COLLISION_PUSH = 0.8

STUCK_THRESHOLD = 0.5


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def serialize_fish(fish: Fish):
    return {
        'id': fish.id,
        'name': fish.name,
        'image_filename': fish.image_filename,
        'x': round(fish.x, 2),
        'y': round(fish.y, 2),
        'vx': round(fish.vx, 2),
        'vy': round(fish.vy, 2),
        'fish_type': fish.fish_type,
        'health': round(fish.health, 1),
    }


def get_all_fishes():
    session = db_session.create_session()
    try:
        return [serialize_fish(f) for f in session.query(Fish).all()]
    finally:
        session.close()


def dist(a: Fish, b: Fish) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def move_toward(fish: Fish, tx, ty, speed):
    dx, dy = tx - fish.x, ty - fish.y
    d = math.hypot(dx, dy) or 0.01
    fish.vx = (dx / d) * speed
    fish.vy = (dy / d) * speed


def move_away(fish: Fish, tx, ty, speed):
    dx, dy = fish.x - tx, fish.y - ty
    d = math.hypot(dx, dy) or 0.01
    fish.vx = (dx / d) * speed
    fish.vy = (dy / d) * speed


def bounce(fish: Fish):
    fish.x = max(0.0, min(100.0, fish.x))
    fish.y = max(0.0, min(100.0, fish.y))

    if fish.x <= 0:
        fish.vx = abs(fish.vx) or SPEED_BASE
    elif fish.x >= 100:
        fish.vx = -(abs(fish.vx) or SPEED_BASE)

    if fish.y <= 0:
        fish.vy = abs(fish.vy) or SPEED_BASE
    elif fish.y >= 100:
        fish.vy = -(abs(fish.vy) or SPEED_BASE)


def resolve_collisions(alive: list):
    diameter = FISH_RADIUS * 2

    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            a, b = alive[i], alive[j]

            dx = b.x - a.x
            dy = b.y - a.y
            d = math.hypot(dx, dy) or 1e-9

            if d >= diameter:
                continue

            overlap = diameter - d
            nx, ny = dx / d, dy / d

            correction = (overlap / 2) * COLLISION_PUSH
            a.x -= nx * correction
            a.y -= ny * correction
            b.x += nx * correction
            b.y += ny * correction

            a_along = a.vx * nx + a.vy * ny
            b_along = b.vx * nx + b.vy * ny

            if a_along - b_along > 0:
                a.vx -= (a_along - b_along) * nx
                a.vy -= (a_along - b_along) * ny
                b.vx += (a_along - b_along) * nx
                b.vy += (a_along - b_along) * ny

            if math.hypot(b.x - a.x, b.y - a.y) < STUCK_THRESHOLD:
                kick = SPEED_MAX * 1.5
                angle = random.uniform(0, 2 * math.pi)
                a.vx = -math.cos(angle) * kick
                a.vy = -math.sin(angle) * kick
                b.vx = math.cos(angle) * kick
                b.vy = math.sin(angle) * kick


def movement_loop():
    while True:
        time.sleep(TICK_INTERVAL)
        session = db_session.create_session()
        try:
            fishes = session.query(Fish).all()
            if not fishes:
                continue

            peaceful = [f for f in fishes if f.fish_type == 'peaceful']
            angry = [f for f in fishes if f.fish_type == 'angry']
            dead_ids = []

            for a in angry:
                a.ticks_since_kill += 1

                targets = sorted(peaceful, key=lambda p: dist(a, p))
                target = next((p for p in targets), None)

                if target and dist(a, target) <= HUNT_RADIUS:
                    move_toward(a, target.x, target.y, SPEED_HUNT)

                    if dist(a, target) <= TOUCH_RADIUS:
                        target.health -= HIT_DAMAGE
                        a.health = min(100.0, a.health + 20.0)
                        a.ticks_since_kill = 0
                        if target.health <= 0:
                            dead_ids.append(target.id)

                if a.ticks_since_kill >= STARVATION_TICKS:
                    a.health -= STARVATION_DAMAGE
                    a.ticks_since_kill = 0
                    if a.health <= 0:
                        dead_ids.append(a.id)

            for p in peaceful:
                if p.id in dead_ids:
                    continue

                threat = next(iter(sorted(angry, key=lambda a: dist(p, a))), None)

                if fish_brain and (not threat or dist(p, threat) <= FLEE_RADIUS):
                    inputs = calc_movement(p, angry)
                    dvx, dvy = fish_brain.forward(inputs)
                    p.vx = 0.7 * p.vx + 0.3 * dvx * SPEED_MAX
                    p.vy = 0.7 * p.vy + 0.3 * dvy * SPEED_MAX
                else:
                    p.vx += random.uniform(-0.05, 0.05)
                    p.vy += random.uniform(-0.05, 0.05)
                    speed = math.hypot(p.vx, p.vy)
                    if speed > SPEED_BASE:
                        p.vx = (p.vx / speed) * SPEED_BASE
                        p.vy = (p.vy / speed) * SPEED_BASE

            for f in fishes:
                if f.id not in dead_ids:
                    f.x += f.vx
                    f.y += f.vy
                    bounce(f)

            alive = [f for f in fishes if f.id not in dead_ids]
            resolve_collisions(alive)

            for f in alive:
                bounce(f)

            if dead_ids:
                for fid in set(dead_ids):
                    dead = session.get(Fish, fid)
                    if dead:
                        session.delete(dead)

            session.commit()

            payload = [serialize_fish(f) for f in fishes if f.id not in dead_ids]
        finally:
            session.close()

        socketio.emit('fish_positions', payload)

        if dead_ids:
            socketio.emit('fishes_updated', get_all_fishes())


def check_credentials():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')

    if not username or not password:
        return False, 'username and password are required'

    session = db_session.create_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user or not user.check_password(password):
            return False, 'invalid username or password'
        return True, ''
    finally:
        session.close()


@app.route('/fish', methods=['POST'])
def create_fish():
    ok, err = check_credentials()
    if not ok:
        return jsonify({'error': err}), 401

    session = db_session.create_session()
    try:
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({'error': 'name is required'}), 400

        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({'error': 'image is required'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': f'file type not allowed; permitted: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        requested_type = request.form.get('fish_type', '').strip().lower()
        fish_type = requested_type if requested_type in ('peaceful', 'angry') else random.choice(['peaceful', 'angry'])

        filename = secure_filename(file.filename)
        file.save(path.join(app.config['UPLOAD_FOLDER'], filename))

        fish = Fish(
            name=name,
            image_filename=filename,
            fish_type=fish_type,
            health=100,
            x=random.uniform(10, 90),
            y=random.uniform(10, 90),
            vx=random.uniform(SPEED_MIN, SPEED_MAX) * random.choice([-1, 1]),
            vy=random.uniform(SPEED_MIN, SPEED_MAX) * random.choice([-1, 1]),
            ticks_since_kill=0,
        )
        session.add(fish)
        session.commit()

        socketio.emit('fishes_updated', get_all_fishes())
        return jsonify({'id': fish.id, 'type': fish.fish_type, 'message': 'Fish created'}), 201
    finally:
        session.close()


@app.route('/users', methods=['POST'])
def create_user():
    session = db_session.create_session()
    try:
        data = request.json or {}

        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'error': 'username and password are required'}), 400

        if len(password) < 6:
            return jsonify({'error': 'password must be at least 6 characters'}), 400

        if session.query(User).filter(User.username == username).first():
            return jsonify({'error': 'username already exists'}), 400

        user = User(username=username)
        user.set_password(password)
        session.add(user)
        session.commit()

        return jsonify({'id': user.id, 'username': user.username}), 201
    finally:
        session.close()


@app.route('/login', methods=['POST'])
def login_user():
    session = db_session.create_session()
    try:
        data = request.json or {}

        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'error': 'username and password are required'}), 400

        user = session.query(User).filter(User.username == username).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'invalid username or password'}), 401

        return jsonify({'id': user.id, 'username': user.username, 'message': 'login successful'})
    finally:
        session.close()


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/stats/', methods=['GET'])
def stats():
    return jsonify({
        'TICK_INTERVAL': TICK_INTERVAL,
        'SPEED_BASE': SPEED_BASE,
        'SPEED_HUNT': SPEED_HUNT,
        'SPEED_FLEE': SPEED_FLEE,
        'HUNT_RADIUS': HUNT_RADIUS,
        'FLEE_RADIUS': FLEE_RADIUS,
        'TOUCH_RADIUS': TOUCH_RADIUS,
        'HIT_DAMAGE': HIT_DAMAGE,
        'STARVATION_TICKS': STARVATION_TICKS,
        'STARVATION_DAMAGE': STARVATION_DAMAGE,
        'SPEED_MIN': SPEED_MIN,
        'SPEED_MAX': SPEED_MAX,
        'FISH_RADIUS': FISH_RADIUS,
        'COLLISION_PUSH': COLLISION_PUSH,
    })


@socketio.on('connect')
def handle_connect():
    emit('all_fishes', get_all_fishes())


@socketio.on('request_all_fishes')
def handle_request_all_fishes():
    emit('all_fishes', get_all_fishes())


def main():
    db_path = path.join('db', 'app.db')
    makedirs('db', exist_ok=True)
    db_session.global_init(db_path)

    t = threading.Thread(target=movement_loop, daemon=True)
    t.start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
