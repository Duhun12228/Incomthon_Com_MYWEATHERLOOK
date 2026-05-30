from flask import Flask, jsonify, request, render_template
import requests
from datetime import datetime, timedelta
import json
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import base64
import math
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import func, case
import pandas as pd
import random
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'closet.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Cloth(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50))
    thickness = db.Column(db.String(50))
    nickname = db.Column(db.String(100))
    photo_path = db.Column(db.String(200))


with app.app_context():
    db.create_all()


# --- ML 모델 ---
temperature_ranges = [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 40)]
humidity_levels = [10, 30, 50, 70, 90]
wind_speeds = [0, 2, 5, 8, 12]
rainfall_levels = [0, 5, 15, 30, 50]

outerwear_options = ['매우 두꺼움', '두꺼움', '보통', '얇음', '매우 얇음', 'None']
top_options = ['매우 두꺼움', '두꺼움', '보통', '얇음', '매우 얇음']
bottom_options = ['긴바지', '반바지']
shoes_options = ['운동화', '구두', '슬리퍼', '레인부츠']


def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        temp = random.uniform(*random.choice(temperature_ranges))
        humidity = random.choice(humidity_levels)
        wind = random.choice(wind_speeds)
        rain = random.choice(rainfall_levels)

        if temp < 15 or rain > 20:
            outerwear = random.choice(outerwear_options[:3])
            shoes = '레인부츠' if rain > 20 else random.choice(shoes_options)
        else:
            outerwear = 'None'
            shoes = random.choice(['운동화', '구두', '슬리퍼'])

        top = random.choice(top_options[:3] if temp < 15 else top_options[2:])
        bottom = '긴바지' if temp < 20 else '반바지'
        data.append([temp, humidity, wind, rain, outerwear, top, bottom, shoes])

    columns = ['온도', '습도', '풍속', '강수량', 'outer', 'top', 'bottom', 'shoes']
    return pd.DataFrame(data, columns=columns)


random.seed(42)
df = generate_data()
X = df[['온도', '습도', '풍속', '강수량']]

model_outer = RandomForestClassifier(random_state=42)
model_top = RandomForestClassifier(random_state=42)
model_bottom = RandomForestClassifier(random_state=42)
model_shoes = RandomForestClassifier(random_state=42)

model_outer.fit(X, df['outer'])
model_top.fit(X, df['top'])
model_bottom.fit(X, df['bottom'])
model_shoes.fit(X, df['shoes'])


# --- 격자 좌표 변환 ---
Re = 6371.00877
grid = 5.0
PI = math.asin(1.0) * 2.0
DEGRAD = PI / 180.0

re = Re / grid
slat1 = 30.0 * DEGRAD
slat2 = 60.0 * DEGRAD
olon = 126.0 * DEGRAD
olat = 38.0 * DEGRAD
xo = 210 / grid
yo = 675 / grid

sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(
    math.tan(PI * 0.25 + slat2 * 0.5) / math.tan(PI * 0.25 + slat1 * 0.5)
)
sf = math.pow(math.tan(PI * 0.25 + slat1 * 0.5), sn) * math.cos(slat1) / sn
ro = re * sf / math.pow(math.tan(PI * 0.25 + olat * 0.5), sn)


def convert_grid(lat, lon):
    ra = re * sf / math.pow(math.tan(PI * 0.25 + float(lat) * DEGRAD * 0.5), sn)
    theta = float(lon) * DEGRAD - olon
    if theta > PI:
        theta -= 2.0 * PI
    if theta < -PI:
        theta += 2.0 * PI
    theta *= sn
    x = int(ra * math.sin(theta) + xo + 1.5)
    y = int(ro - ra * math.cos(theta) + yo + 1.5)
    return x, y


serviceKey = os.environ.get('WEATHER_API_KEY', '')


# --- 라우트 ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/closet')
def closet_page():
    return render_template('closet.html')


@app.route('/add_cloth')
def add_cloth_page():
    return render_template('Add_cloth.html')


@app.route('/outer_page')
def outer_page():
    return render_template('outer.html')


@app.route('/top_page')
def top_page():
    return render_template('top.html')


@app.route('/bottom_page')
def bottom_page():
    return render_template('bottom.html')


@app.route('/shoes_page')
def shoes_page():
    return render_template('shoes.html')


@app.route('/edit_outer_page')
def edit_outer_page():
    item_id = int(request.args.get('id', 0))
    return render_template('edit_outer.html', item_id=item_id)


@app.route('/edit_top_page')
def edit_top_page():
    item_id = int(request.args.get('id', 0))
    return render_template('edit_top.html', item_id=item_id)


@app.route('/edit_bottom_page')
def edit_bottom_page():
    item_id = int(request.args.get('id', 0))
    return render_template('edit_bottom.html', item_id=item_id)


@app.route('/edit_shoes_page')
def edit_shoes_page():
    item_id = int(request.args.get('id', 0))
    return render_template('edit_shoes.html', item_id=item_id)


# --- 추천 로직 ---
def recommend_outfit(user_weather_data):
    temp, humidity, wind_speed, precipitation = user_weather_data
    data = [[temp, humidity, wind_speed, precipitation]]
    return {
        'outer': model_outer.predict(data)[0],
        'top': model_top.predict(data)[0],
        'bottom': model_bottom.predict(data)[0],
        'shoes': model_shoes.predict(data)[0],
    }


def match_recommendation(user_weather_data):
    recommended = recommend_outfit(user_weather_data)
    matched = {}
    for category, item in recommended.items():
        if item == 'None':
            matched[category] = '착용 안함'
            continue
        cloth = Cloth.query.filter_by(category=category, thickness=item).first()
        if not cloth:
            cloth = find_closest_cloth(category, item)
        matched[category] = cloth.nickname if cloth else f"{item}에 맞는 옷이 없습니다."
    return matched


def find_closest_cloth(category, target_attribute):
    if category in ['outer', 'top']:
        thickness_levels = {
            '매우 두꺼움': 4, '두꺼움': 3, '보통': 2, '얇음': 1, '매우 얇음': 0
        }
        target_value = thickness_levels.get(target_attribute, 2)
        thickness_case = case(
            *[(Cloth.thickness == key, value) for key, value in thickness_levels.items()],
            else_=2
        )
        return (Cloth.query
                .filter(Cloth.category == category)
                .order_by(func.abs(thickness_case - target_value))
                .first())
    else:
        return Cloth.query.filter(Cloth.category == category).first()


@app.route('/weather', methods=['GET'])
def get_weather():
    now = datetime.now()
    base_time_dt = now - timedelta(hours=3)
    base_date = base_time_dt.strftime('%Y%m%d')
    base_time = base_time_dt.strftime('%H%M')

    fcst_time_dt = now - timedelta(hours=1)
    fcst_time = fcst_time_dt.replace(minute=0, second=0).strftime('%H%M')

    lon = request.args.get('lon', '').strip()
    lat = request.args.get('lat', '').strip()

    if not lat or not lon:
        return jsonify({"error": "위도 또는 경도가 제공되지 않았습니다."}), 400

    nx, ny = convert_grid(float(lat), float(lon))
    url = (
        f"http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
        f"?serviceKey={serviceKey}&numOfRows=60&pageNo=1&dataType=json"
        f"&base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}"
    )

    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({"error": "기상청 API 요청 실패"}), 500

    res = json.loads(response.text)
    informations = {}
    for item in res['response']['body']['items']['item']:
        t = item['fcstTime']
        if t not in informations:
            informations[t] = {}
        informations[t][item['category']] = item['fcstValue']

    max_temperature = max(
        (float(v['T1H']) for v in informations.values() if 'T1H' in v),
        default=None
    )

    desired = informations.get(fcst_time, {})
    A1 = max_temperature
    A2 = desired.get('REH')
    A3 = desired.get('WSD')
    A4 = desired.get('RN1')

    precip = 0.0 if A4 == '강수없음' else (float(A4) if A4 else 0.0)
    matched = match_recommendation([A1, A2, A3, precip])

    return jsonify({
        'A1': A1, 'A2': A2, 'A3': A3, 'A4': A4,
        'outer': matched.get('outer', ''),
        'top': matched.get('top', ''),
        'bottom': matched.get('bottom', ''),
        'shoes': matched.get('shoes', ''),
    })


@app.route('/add_cloth', methods=['POST'])
def add_cloth():
    data = request.json
    category = data['category']
    thickness = data['thickness']
    nickname = data['nickname']
    photo_base64 = data.get('photo')

    photo_filename = None
    if photo_base64:
        photo_data = photo_base64.split(',')[1]
        photo_filename = f"{nickname}.png"
        with open(os.path.join(UPLOAD_FOLDER, photo_filename), 'wb') as fh:
            fh.write(base64.b64decode(photo_data))

    db.session.add(Cloth(
        category=category,
        thickness=thickness,
        nickname=nickname,
        photo_path=photo_filename,
    ))
    db.session.commit()
    return jsonify({'message': '옷 정보가 저장되었습니다!'}), 201


@app.route('/get_clothes', methods=['GET'])
def get_clothes():
    return jsonify([{
        'id': c.id, 'category': c.category,
        'thickness': c.thickness, 'nickname': c.nickname,
        'photo_path': c.photo_path,
    } for c in Cloth.query.all()])


def _cloth_response(cloths, index):
    if not cloths or index < 0 or index >= len(cloths):
        return jsonify(None), 404
    c = cloths[index]
    return jsonify({
        'id': c.id, 'nickname': c.nickname,
        'category': c.category, 'thickness': c.thickness,
        'photo_path': c.photo_path, 'total': len(cloths),
    })


@app.route('/outer_get', methods=['GET'])
def get_outer_cloth():
    return _cloth_response(Cloth.query.filter_by(category='outer').all(), int(request.args.get('index', 0)))


@app.route('/top_get', methods=['GET'])
def get_top_cloth():
    return _cloth_response(Cloth.query.filter_by(category='top').all(), int(request.args.get('index', 0)))


@app.route('/bottom_get', methods=['GET'])
def get_bottom_cloth():
    return _cloth_response(Cloth.query.filter_by(category='bottom').all(), int(request.args.get('index', 0)))


@app.route('/shoes_get', methods=['GET'])
def get_shoes_cloth():
    return _cloth_response(Cloth.query.filter_by(category='shoes').all(), int(request.args.get('index', 0)))


@app.route('/update_cloth', methods=['POST'])
def update_cloth():
    cloth = db.session.get(Cloth, int(request.args.get('id')))
    if not cloth:
        return jsonify(None), 404
    cloth.category = request.form.get('category')
    cloth.thickness = request.form.get('thickness')
    cloth.nickname = request.form.get('nickname')
    db.session.commit()
    return jsonify(success=True)


@app.route('/delete_cloth', methods=['DELETE'])
def delete_cloth():
    cloth = db.session.get(Cloth, int(request.args.get('id')))
    if not cloth:
        return jsonify(None), 404
    db.session.delete(cloth)
    db.session.commit()
    return jsonify(success=True)


@app.route('/it_db', methods=['GET'])
def initialize_db():
    with app.app_context():
        db.drop_all()
        db.create_all()
    return jsonify({"message": "데이터베이스가 초기화되었습니다."}), 200


if __name__ == '__main__':
    app.run(debug=True)
