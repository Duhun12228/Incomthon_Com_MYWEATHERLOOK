# MYWEATHERLOOK

날씨 기반 옷 추천 웹 애플리케이션 — Incomthon 팀 프로젝트

## 개요

사용자의 현재 위치를 기반으로 기상청 초단기예보 API에서 날씨 데이터를 받아오고, RandomForest 머신러닝 모델로 상의·하의·아우터·신발을 추천합니다. 추천 결과는 사용자가 직접 등록한 옷 DB와 매칭되어 본인 옷 중에서 가장 적합한 코디를 제안합니다.

## 기술 스택

| 분류 | 기술 |
|------|------|
| 백엔드 | Python, Flask |
| DB | SQLite (Flask-SQLAlchemy) |
| ML | scikit-learn (RandomForestClassifier) |
| 외부 API | 기상청 초단기예보 API |
| 프론트엔드 | HTML / CSS / JavaScript (Jinja2) |

## 프로젝트 구조

```
.
├── weather_api.py       # Flask 앱 진입점 (백엔드 전체)
├── templates/           # Jinja2 HTML 템플릿
│   ├── index.html       # 메인 (날씨 + 추천 결과)
│   ├── closet.html      # 옷장 메뉴
│   ├── Add_cloth.html   # 옷 등록
│   ├── outer/top/bottom/shoes.html   # 카테고리별 조회
│   └── edit_*.html      # 카테고리별 수정
├── static/
│   ├── LOGO.PNG
│   └── uploads/         # 사용자 업로드 옷 사진
└── instance/
    └── closet.db        # SQLite DB (자동 생성)
```

## 시작하기

### 1. 가상환경 생성 및 의존성 설치

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. 환경변수 설정

프로젝트 루트의 `.env` 파일에 기상청 API 키를 입력합니다.

```
WEATHER_API_KEY=발급받은_서비스키
```

> 기상청 공공데이터포털에서 **초단기예보조회서비스** API 키를 발급받아야 합니다.

### 3. 실행

```bash
python weather_api.py
```

브라우저에서 `http://127.0.0.1:5000` 접속

## 머신러닝 모델

### 개요

서버 시작 시 규칙 기반으로 합성 데이터 1,000개를 생성하고, 카테고리별로 4개의 RandomForestClassifier를 학습시킵니다. 별도의 학습 데이터 파일이나 사전 학습 과정 없이 즉시 실행됩니다.

### 입력 피처

| 피처 | 설명 |
|------|------|
| 온도 (°C) | -10 ~ 40 범위 |
| 습도 (%) | 10 / 30 / 50 / 70 / 90 |
| 풍속 (m/s) | 0 / 2 / 5 / 8 / 12 |
| 강수량 (mm) | 0 / 5 / 15 / 30 / 50 |

### 출력 레이블

| 모델 | 예측값 |
|------|--------|
| 아우터 | 매우 두꺼움 / 두꺼움 / 보통 / 얇음 / 매우 얇음 / None |
| 상의 | 매우 두꺼움 / 두꺼움 / 보통 / 얇음 / 매우 얇음 |
| 하의 | 긴바지 / 반바지 |
| 신발 | 운동화 / 구두 / 슬리퍼 / 레인부츠 |

### 데이터 생성 규칙

- 온도 15°C 미만 또는 강수량 20mm 초과 → 두꺼운 아우터 착용
- 강수량 20mm 초과 → 레인부츠
- 온도 20°C 미만 → 긴바지, 이상 → 반바지
- 온도에 따라 상의 두께 범위 결정

### DB 매칭

예측된 두께/종류와 일치하는 옷을 사용자 DB에서 검색합니다. 정확히 일치하는 옷이 없으면 두께 단계(`매우 두꺼움=0` ~ `매우 얇음=4`)를 수치화하여 가장 가까운 옷을 추천합니다.

## 주요 기능

- **날씨 조회**: 브라우저 Geolocation으로 위경도를 가져와 기상청 격자 좌표로 변환 후 API 호출
- **옷 추천**: 온도·습도·풍속·강수량 4가지 입력으로 아우터·상의·하의·신발 각각 RandomForest 예측
- **DB 매칭**: 예측된 두께/종류와 일치하는 옷을 사용자 DB에서 검색, 없으면 가장 유사한 두께의 옷 추천
- **옷장 관리**: 옷 등록(카테고리·두께·별칭·사진), 조회, 수정, 삭제

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/weather?lat=&lon=` | 날씨 조회 및 옷 추천 |
| POST | `/add_cloth` | 옷 등록 |
| GET | `/outer_get?index=` | 아우터 조회 |
| GET | `/top_get?index=` | 상의 조회 |
| GET | `/bottom_get?index=` | 하의 조회 |
| GET | `/shoes_get?index=` | 신발 조회 |
| POST | `/update_cloth?id=` | 옷 수정 |
| DELETE | `/delete_cloth?id=` | 옷 삭제 |
| GET | `/it_db` | DB 초기화 (개발용) |
