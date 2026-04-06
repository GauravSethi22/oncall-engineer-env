FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    websockets==12.0 \
    pydantic==2.10.0
COPY models.py .
COPY server/ ./server/
ENV ONCALL_TASK=easy_crash
EXPOSE 7860
CMD uvicorn server.app:app --host 0.0.0.0 --port 7860