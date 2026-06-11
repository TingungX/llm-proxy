FROM python:3.11-slim

# 避免 .pyc 与强制 stdout 不缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY llm_proxy/ ./llm_proxy/
COPY proxy.py .

# 复制前端构建产物（dist/ 必须已存在；构建时由 docker-compose 通过 build-arg 或外部构建）
COPY static/dist/ ./static/dist/

# 创建非 root 用户
RUN useradd -m -u 1000 llmproxy && \
    chown -R llmproxy:llmproxy /app
USER llmproxy

EXPOSE 4000

# 健康检查：访问 /api/config（无认证可读）
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:4000/api/config', timeout=5).raise_for_status()" || exit 1

CMD ["uvicorn", "llm_proxy.main:app", "--host", "0.0.0.0", "--port", "4000"]

