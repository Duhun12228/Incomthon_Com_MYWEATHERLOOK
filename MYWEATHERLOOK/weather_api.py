from flask import Flask, jsonify
import requests
from datetime import datetime, timedelta
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 모든 도메인에 대해 CORS 허용

serviceKey = "exBFq7a26F5gRk9CcioJM5kGSnfcxKS%2B%2FOc0gIbbq4HlXFtQw6GSl3raqQT3QqYE1sZqMjGg6hh11JBhX4y59g%3D%3D"  # 본인의 서비스 키 입력
# 날씨를 알고 싶은 시간 입력
base_date = '20241015'  # 발표 일자
base_time = '0700'  # 발표 시간
nx = '62'  # 예보 지점 x좌표
ny = '123'  # 예보 지점 y좌표

@app.route('/weather', methods=['GET'])
def get_weather():
    # 현재 입력 시간 설정
    input_d = datetime.strptime(base_date + base_time, "%Y%m%d%H%M") - timedelta(hours=1)
    input_datetime = datetime.strftime(input_d, "%Y%m%d%H%M")
    
    # 요청 URL 설정
    url = f"http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst?serviceKey={serviceKey}&numOfRows=60&pageNo=1&dataType=json&base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}"
    
    # 날씨 데이터 요청
    response = requests.get(url)
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

    # 최근 데이터 추출
    latest_time = max(informations.keys())
    latest_data = informations[latest_time]
    
    # 변수 A1, A2, A3, A4에 값 저장
    A1 = max_temperature if max_temperature != float('-inf') else None  # 최고 기온
    A2 = latest_data.get('REH', None)  # 습도
    A3 = latest_data.get('WSD', None)  # 풍속
    A4 = latest_data.get('RN1', None)  # 강수량
    # JSON 응답
    return jsonify({
        'A1': A1,
        'A2': A2,
        'A3': A3,
        'A4': A4,
        '조회 시간':latest_time
    })

if __name__ == '__main__':
    app.run(debug=True)
