from flask import Flask, jsonify, request, render_template,redirect,url_for
import requests
from datetime import datetime, timedelta
import json
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import base64
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sqlalchemy import func
import pandas as pd
import random
from sqlalchemy import case

app = Flask(__name__, template_folder='/MYWEATHERLOOK/templates',static_folder='/MYWEATHERLOOK/static')
CORS(app)  # 모든 도메인에 대해 CORS 허용




# SQLite 데이터베이스 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///closet.db'  # SQLite 데이터베이스 파일
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 데이터베이스 모델 정의
class Cloth(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50))
    thickness = db.Column(db.String(50))
    nickname = db.Column(db.String(100))
    photo_path = db.Column(db.String(200))

# 데이터베이스 생성
with app.app_context():
    db.create_all()

###머신러닝 모델 설정
# 설정된 날씨 및 의류 특성 옵션
temperature_ranges = [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 40)]  # 예: 매우 춥고, 추운, 보통, 따뜻한, 매우 더운 날씨
humidity_levels = [10, 30, 50, 70, 90]
wind_speeds = [0, 2, 5, 8, 12]
rainfall_levels = [0, 5, 15, 30, 50]  # 강수량

# 의류 조합 옵션
outerwear_options = ['매우 두꺼움', '두꺼움', '보통', '얇음', '매우 얇음', 'None']
top_options = ['매우 두꺼움', '두꺼움', '보통', '얇음', '매우 얇음']
bottom_options = ['긴바지', '반바지']
shoes_options = ['운동화', '구두', '슬리퍼', '레인부츠']

# 학습 데이터셋 생성 함수
def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        # 랜덤 날씨 생성
        temp = random.uniform(*random.choice(temperature_ranges))
        humidity = random.choice(humidity_levels)
        wind = random.choice(wind_speeds)
        rain = random.choice(rainfall_levels)

        # 조건에 따른 의류 조합 설정
        if temp < 15 or rain > 20:  # 추운 날 또는 비가 많이 오는 날
            outerwear = random.choice(outerwear_options[:3])  # 두꺼운 아우터
            shoes = '레인부츠' if rain > 20 else random.choice(shoes_options)
        else:
            outerwear = 'None'  # 아우터 없음
            shoes = random.choice(['운동화', '구두', '슬리퍼'])

        top = random.choice(top_options[:3] if temp < 15 else top_options[2:])  # 기온에 따른 상의 두께
        bottom = '긴바지' if temp < 20 else '반바지'  # 기온에 따른 하의 길이

        # 데이터 추가
        data.append([temp, humidity, wind, rain, outerwear, top, bottom, shoes])

    columns = ['온도', '습도', '풍속', '강수량', 'outer', 'top', 'bottom', 'shoes']
    return pd.DataFrame(data, columns=columns)

# 학습 데이터셋 생성
df = generate_data()
print(df.head())  # 데이터셋 확인


# 학습 데이터 준비
training_data = df
X = training_data[['온도', '습도', '풍속', '강수량']]
y_outer = training_data['outer']
y_top = training_data['top']
y_bottom = training_data['bottom']
y_shoes = training_data['shoes']

# 랜덤 포레스트 모델 학습
model_outer = RandomForestClassifier()
model_top = RandomForestClassifier()
model_bottom = RandomForestClassifier()
model_shoes = RandomForestClassifier()

model_outer.fit(X, y_outer)
model_top.fit(X, y_top)
model_bottom.fit(X, y_bottom)
model_shoes.fit(X, y_shoes)
#-------------------------------------------------------------------------

##격좌좌표 변환 
NX = 149            ## X축 격자점 수
NY = 253            ## Y축 격자점 수

Re = 6371.00877     ##  지도반경
grid = 5.0          ##  격자간격 (km)
slat1 = 30.0        ##  표준위도 1
slat2 = 60.0        ##  표준위도 2
olon = 126.0        ##  기준점 경도
olat = 38.0         ##  기준점 위도
xo = 210 / grid     ##  기준점 X좌표
yo = 675 / grid     ##  기준점 Y좌표
first = 0

if first == 0 :
    PI = math.asin(1.0) * 2.0
    DEGRAD = PI/ 180.0
    RADDEG = 180.0 / PI


    re = Re / grid
    slat1 = slat1 * DEGRAD
    slat2 = slat2 * DEGRAD
    olon = olon * DEGRAD
    olat = olat * DEGRAD

    sn = math.tan(PI * 0.25 + slat2 * 0.5) / math.tan(PI * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(PI * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(PI * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)
    first = 1

#위도,경도 -> 격좌좌표
def convert_grid(lat, lon):
    ra = math.tan(PI * 0.25 + float(lat) * DEGRAD * 0.5)
    ra = re * sf / pow(ra, sn)
    theta = float(lon) * DEGRAD - olon
    if theta > PI :
        theta -= 2.0 * PI
    if theta < -PI :
        theta += 2.0 * PI
    theta *= sn
    x = (ra * math.sin(theta)) + xo
    y = (ro - ra * math.cos(theta)) + yo
    x = int(x + 1.5)
    y = int(y + 1.5)
    return x, y


serviceKey = "exBFq7a26F5gRk9CcioJM5kGSnfcxKS%2B%2FOc0gIbbq4HlXFtQw6GSl3raqQT3QqYE1sZqMjGg6hh11JBhX4y59g%3D%3D"  # 본인의 서비스 키 입력

#----------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')
#----------------------------------------------------

@app.route('/closet')
def closet_page():
    return render_template('closet.html')  # closet.html 파일을 templates 폴더에 넣어야 합니다
#----------------------------------------------------

@app.route('/add_cloth')
def add_cloth_page():
    return render_template('Add_cloth.html')  # Add_cloth.html 파일을 templates 폴더에 넣어야 합니다
#----------------------------------------------------

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

#----------------------------------------------------
@app.route('/edit_outer_page')
def edit_outer_page():
    item_id = int(request.args.get('id'))
    print(item_id)
    return render_template('edit_outer.html', item_id=item_id)


# edit_outer.html 렌더링 처리
@app.route('/render_edit_outer')
def render_edit_outer():
    return render_template('edit_outer.html', item_id=2)


@app.route('/edit_top_page', methods=['POST'])
def edit_top_page():
    data = request.get_json()
    global top_id
    top_id = data.get('id') if data else None   
    if top_id is None:
        print('NO item_id')
        return jsonify({'error': 'No item_id provided'}), 400
    else:
        print(f'Item ID: {top_id}')
        return jsonify({'success': 'ID received', 'top_id': top_id}), 200
    
@app.route('/render_edit_top')
def render_edit_top():
    return render_template('edit_top.html', item_id=top_id)

@app.route('/edit_bottom_page', methods=['POST'])
def edit_bottom_page():
    data = request.get_json()
    global bottom_id
    bottom_id = data.get('id') if data else None   
    if bottom_id is None:
        print('NO item_id')
        return jsonify({'error': 'No item_id provided'}), 400
    else:
        print(f'Item ID: {bottom_id}')

@app.route('/render_edit_bottom')
def render_edit_bottom():
    return render_template('edit_bottom.html', item_id=bottom_id)


@app.route('/edit_shoes_page', methods=['POST'])
def edit_shoes_page():
    data = request.get_json()
    global shoes_id
    shoes_id = data.get('id') if data else None   
    if shoes_id is None:
        print('NO item_id')
        return jsonify({'error': 'No item_id provided'}), 400
    else:
        print(f'Item ID: {shoes_id}')

@app.route('/render_edit_shoes')
def render_edit_shoes():
    return render_template('edit_shoes.html', item_id=shoes_id)

#----------------------------------------------------
# 사용자의 날씨 데이터에 대한 옷 추천 함수
def recommend_outfit(user_weather_data):
    temp, humidity, wind_speed, precipitation = user_weather_data

    outer_pred = model_outer.predict([[temp, humidity, wind_speed, precipitation]])[0]
    top_pred = model_top.predict([[temp, humidity, wind_speed, precipitation]])[0]
    bottom_pred = model_bottom.predict([[temp, humidity, wind_speed, precipitation]])[0]
    shoes_pred = model_shoes.predict([[temp, humidity, wind_speed, precipitation]])[0]

    return {'outer': outer_pred, 'top': top_pred, 'bottom': bottom_pred, 'shoes': shoes_pred}


# 데이터베이스에서 사용자 소유 옷과 추천 옷을 매칭하고 비슷한 조건의 옷을 찾는 함수
def match_recommendation(user_weather_data):
    recommended_outfit = recommend_outfit(user_weather_data)

    matched_clothes = {}
    for category, recommended_item in recommended_outfit.items():
        if recommended_item == 'None':
            matched_clothes[category] = '착용 안함'
            continue

        # 데이터베이스에서 추천된 조건과 일치하는 옷 검색
        matched_cloth = Cloth.query.filter_by(category=category, thickness=recommended_item).first()

        # 만약 일치하는 옷이 없으면, 가장 가까운 조건의 옷을 검색
        if not matched_cloth:
            matched_cloth = find_closest_cloth(category, recommended_item)
        # 결과 저장
        matched_clothes[category] = matched_cloth.nickname if matched_cloth else f"{recommended_item}에 맞는 옷이 없습니다."
    print(matched_clothes)
    return matched_clothes


# 유사 조건의 옷을 찾기 위한 함수 (하의와 신발은 아무 옷 추천)
def find_closest_cloth(category, target_attribute):
    if category in ['outer', 'top']:
        # 두께에 따른 유사 조건 검색
        thickness_levels = {
            '매우 두꺼움': 4, '두꺼움': 3, '보통': 2, '얇음': 1, '매우 얇음': 0
        }
        target_value = thickness_levels.get(target_attribute, 2)  # '보통'을 기본 값으로 설정
        
        # 두께 값 변환을 위해 case문 사용
        thickness_case = case(
            *[(Cloth.thickness == key, value) for key, value in thickness_levels.items()],
            else_=2  # 기본값을 '보통'으로 설정
        )

        closest_cloth = (
            Cloth.query \
            .filter(Cloth.category == category) \
            .order_by(func.abs(thickness_case - target_value))\
            .first()
        )
    elif category == 'bottom':
        # 아무 하의 추천
        closest_cloth = Cloth.query.filter(Cloth.category == category).first()
    elif category == 'shoes':
        # 아무 신발 추천
        closest_cloth = Cloth.query.filter(Cloth.category == category).first()
    else:
        closest_cloth = None

    return closest_cloth


@app.route('/weather', methods=['GET'])

def get_weather():
        # 현재 시간 설정
    now = datetime.now()

    # Base_date와 Base_time 설정 (현재로부터 3시간 전)
    base_time_dt = now - timedelta(hours=3)
    base_date = base_time_dt.strftime('%Y%m%d')  # YYYYMMDD 형식
    base_time = base_time_dt.strftime('%H%M')     # HHMM 형식

    # fcstDate와 fcstTime 설정 (현재로부터 1시간 전, 분은 00으로 설정)
    fcst_time_dt = now - timedelta(hours=1)
    fcstDate = fcst_time_dt.strftime('%Y%m%d')  # YYYYMMDD 형식
    fcstTime = fcst_time_dt.replace(minute=0, second=0).strftime('%H%M')  # HH00 형식

    print(f"Request args: {request.args}")  # 요청 인수 출력

    # 프론트엔드로부터 경도와 위도 값 받기
    lon = request.args.get('lon')  # 경도
    lat = request.args.get('lat')   # 위도
    
    # lat와 lon이 None인 경우 처리
    if lon is None or lat is None or lon.strip() == '' or lat.strip() == '':
        return jsonify({"error": "Latitude or longitude not provided"}), 400


    lat = float(lat)
    lon = float(lon)

    print(f"Converted values: lat={lat}, lon={lon}")  # 변환된 값 출력
    print(f"Types: lat type={type(lat)}, lon type={type(lon)}")
    
    nx, ny = convert_grid(lat, lon) #격좌과표로 변환
    nx = int(nx)
    ny = int(ny)
    print(f"nx:{nx}, ny: {ny}")
    # 요청 URL 설정
    url = f"http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst?serviceKey={serviceKey}&numOfRows=60&pageNo=1&dataType=json&base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}"
    
    # 날씨 데이터 요청
    response = requests.get(url)
    print("응답 상태 코드:", response.status_code)
    # print("응답 내용:", response.text)  # api 응답 내용 출력(디버깅용)

    if response.status_code != 200:
        print(f"API 요청 실패: {response.status_code}")
        return jsonify({"error": "API 요청 실패"}), 500

    res = json.loads(response.text)

    informations = dict()
    for items in res['response']['body']['items']['item']:
        cate = items['category']
        fcstTime = items['fcstTime']
        fcstValue = items['fcstValue']
        if fcstTime not in informations.keys():
            informations[fcstTime] = dict()
        informations[fcstTime][cate] = fcstValue


    # 최고 기온 찾기
    max_temperature = float('-inf')  # 음의 무한대
    for val in informations.values():
        if 'T1H' in val and val['T1H'] is not None:
            current_temp = float(val['T1H'])
            if current_temp > max_temperature:
                max_temperature = current_temp

    desired_fcst_data = informations.get(fcstTime)


    # 변수 A1, A2, A3, A4에 값 저장
    A1 = max_temperature if max_temperature != float('-inf') else None  # 최고 기온
    A2 = desired_fcst_data.get('REH', None) if desired_fcst_data else None  # 습도
    A3 = desired_fcst_data.get('WSD', None) if desired_fcst_data else None  # 풍속   
    A4 = desired_fcst_data.get('RN1', None) if desired_fcst_data else None  # 강수량
    
    user_weather_data = [A1,A2,A3,A4 if A4 != '강수없음' else 0]
    matched_clothes = match_recommendation(user_weather_data)
    outer = matched_clothes.get('outer', '기본값')  # 기본값은 항목이 없을 때 반환될 값입니다.
    top = matched_clothes.get('top', '기본값')
    bottom = matched_clothes.get('bottom', '기본값')
    shoes = matched_clothes.get('shoes', '기본값')
    print(outer,top,bottom,shoes)
    # JSON 응답
    return jsonify({
        'A1': A1,
        'A2': A2,
        'A3': A3,
        'A4': A4,
        'outer': outer,
        'top': top,
        'bottom': bottom,
        'shoes': shoes        
    }
    )


#새로운 옷 추가 
@app.route('/add_cloth', methods=['POST'])
def add_cloth():
    if request.method == 'POST':
        data = request.json  # JSON 데이터 가져오기
        category = data['category']  # JSON에서 category 추출
        thickness = data['thickness']
        nickname = data['nickname']
        
        photo_base64 = data['photo']  # Base64 이미지 데이터
        
        # 이미지 파일 저장 로직 (Base64 디코딩)
        if photo_base64:
            photo_data = photo_base64.split(',')[1]  # Base64 데이터에서 메타데이터 제거
            photo_filename = f"{nickname}.png"  # 파일 이름 설정
            photo_path = os.path.join('/MYWEATHERLOOK/static/uploads', photo_filename)
            with open(photo_path, "wb") as fh:
                fh.write(base64.b64decode(photo_data))  # Base64 데이터를 디코딩하여 저장
        else:
            photo_path = None

        # 옷 정보를 데이터베이스에 추가
        new_cloth = Cloth(
            category=category,
            thickness=thickness,
            nickname=nickname,
            photo_path=photo_path
        )
        db.session.add(new_cloth)
        db.session.commit()

        return jsonify({'message': '옷 정보가 저장되었습니다!'}), 201
    


#서버에서 데이터베이스 확인용    
@app.route('/get_clothes', methods=['GET'])
def get_clothes():
    clothes = Cloth.query.all()  # 모든 옷 정보 조회
    results = [
        {
            'id': cloth.id,
            'category': cloth.category,
            'thickness': cloth.thickness,
            'nickname': cloth.nickname,
            'photo_path': cloth.photo_path
        } for cloth in clothes
    ]
    return jsonify(results)


@app.route('/outer_get', methods=['GET'])
def get_outer_cloth():
    index = int(request.args.get('index'))
    # 인덱스를 사용하여 특정 옷을 찾으려면 우선 아우터 카테고리의 모든 옷을 가져옵니다.
    outer_cloths = Cloth.query.filter_by(category='outer').all()
    total_count = Cloth.query.filter_by(category='outer').count()
    # 아우터가 없는 경우
    if not outer_cloths:
        return jsonify(None), 404

    # 인덱스를 사용하여 아우터 중에서 특정 옷을 선택합니다.
    if 0 <= index < len(outer_cloths):
        cloth = outer_cloths[index]
        return jsonify({
            'id': cloth.id,  # DB의 고유 ID 추가
            'nickname': cloth.nickname,
            'category': cloth.category,
            'thickness': cloth.thickness,
            'photo_path': cloth.photo_path,
            'total': total_count
        })
    
    return jsonify(None), 404  # 유효하지 않은 인덱스의 경우

@app.route('/top_get', methods=['GET'])
def get_top_cloth():
    index = int(request.args.get('index'))
    # 인덱스를 사용하여 특정 옷을 찾으려면 우선 아우터 카테고리의 모든 옷을 가져옵니다.
    top_cloths = Cloth.query.filter_by(category='top').all()
    total_count = Cloth.query.filter_by(category='top').count()
    # 아우터가 없는 경우
    if not top_cloths:
        return jsonify(None), 404

    # 인덱스를 사용하여 아우터 중에서 특정 옷을 선택합니다.
    if 0 <= index < len(top_cloths):
        cloth = top_cloths[index]
        return jsonify({
            'id': cloth.id,  # DB의 고유 ID 추가            
            'nickname': cloth.nickname,
            'category': cloth.category,
            'thickness': cloth.thickness,
            'photo_path': cloth.photo_path,
            'total':total_count
        })
    
    return jsonify(None), 404  # 유효하지 않은 인덱스의 경우

@app.route('/bottom_get', methods=['GET'])
def get_bottom_cloth():
    index = int(request.args.get('index'))
    # 인덱스를 사용하여 특정 옷을 찾으려면 우선 아우터 카테고리의 모든 옷을 가져옵니다.
    bottom_cloths = Cloth.query.filter_by(category='bottom').all()
    total_count = Cloth.query.filter_by(category='bottom').count()
    # 아우터가 없는 경우
    if not bottom_cloths:
        return jsonify(None), 404

    # 인덱스를 사용하여 아우터 중에서 특정 옷을 선택합니다.
    if 0 <= index < len(bottom_cloths):
        cloth = bottom_cloths[index]
        return jsonify({
            'id': cloth.id,  # DB의 고유 ID 추가
            'nickname': cloth.nickname,
            'category': cloth.category,
            'thickness': cloth.thickness,
            'photo_path': cloth.photo_path,
            'total': total_count
        })
    
    return jsonify(None), 404  # 유효하지 않은 인덱스의 경우

@app.route('/shoes_get', methods=['GET'])
def get_shoes_cloth():
    index = int(request.args.get('index'))
    # 인덱스를 사용하여 특정 옷을 찾으려면 우선 아우터 카테고리의 모든 옷을 가져옵니다.
    shoes_cloths = Cloth.query.filter_by(category='shoes').all()
    total_count = Cloth.query.filter_by(category='shoes').count()
    # 아우터가 없는 경우
    if not shoes_cloths:
        return jsonify(None), 404

    # 인덱스를 사용하여 아우터 중에서 특정 옷을 선택합니다.
    if 0 <= index < len(shoes_cloths):
        cloth = shoes_cloths[index]
        return jsonify({
            'id': cloth.id,  # DB의 고유 ID 추가
            'nickname': cloth.nickname,
            'category': cloth.category,
            'thickness': cloth.thickness,
            'photo_path': cloth.photo_path,
            'total':total_count
        })
    
    return jsonify(None), 404  # 유효하지 않은 인덱스의 경우




# 옷 정보 업데이트
@app.route('/update_cloth', methods=['POST'])
def update_cloth():
    id = int(request.args.get('id'))
    cloth = Cloth.query.get(id)
    if not cloth:
        return jsonify(None), 404

    # 업데이트할 데이터 받기
    cloth.category = request.form.get('category')
    cloth.thickness = request.form.get('thickness')
    cloth.nickname = request.form.get('nickname')

    # 새로운 사진 데이터(Base64)가 포함되어 있다면 사진 업데이트
    # # photo_base64 = request.form.get('photo')  # Base64 인코딩된 사진 데이터
    # if photo_base64:
    #     photo_data = photo_base64.split(',')[1]  # 메타데이터 제거
    #     photo_filename = f"{cloth.nickname}.png"  # 새로운 파일 이름
    #     photo_path = os.path.join('C:/Users/user/Desktop/WEATHERLOOK/static/uploads', photo_filename)
    #     with open(photo_path, "wb") as fh:
    #         fh.write(base64.b64decode(photo_data))  # Base64 데이터를 디코딩하여 저장
    #     cloth.photo_path = photo_path  # DB에 저장할 경로
    # else: 
    #     print('사진 업데이트는 왜 안되는거짇')
    db.session.commit()
    return jsonify(success=True)

# 옷 삭제하기
@app.route('/delete_cloth', methods=['DELETE'])
def delete_cloth():
    id = int(request.args.get('id'))
    print(id)
    cloth = Cloth.query.get(id)
    if not cloth:
        return jsonify(None), 404

    db.session.delete(cloth)
    db.session.commit()
    return jsonify(success=True)


#DB초기화
@app.route('/it_db', methods=['GET'])
def initialize_db():
    with app.app_context():
        db.drop_all()  # 기존 모든 테이블 삭제
        db.create_all()  # 테이블 다시 생성
    return jsonify({"message": "Database initialized."}), 500

if __name__ == '__main__':
    app.run(debug=True)
