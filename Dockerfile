# Python slim base
FROM python:3.10-slim

# 환경 변수 (thread 줄여서 메모리 절약)
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1

WORKDIR /app

# requirements 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install gunicorn

# 앱 소스 복사
COPY . /app

# 모델 파일이 반드시 backend/models/best.pt 에 있어야 함
# 만약 repo에 포함시키지 않았다면 여기에서 curl 로 다운받도록 수정 가능
# RUN curl -L -o models/best.pt "https://YOUR_PUBLIC_URL/best.pt"

# Flask/Gunicorn 실행 (HuggingFace Spaces는 기본 PORT=7860, Render/Cloud Run은 $PORT 환경변수 제공)
ENV PORT=7860
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 180 app:app
