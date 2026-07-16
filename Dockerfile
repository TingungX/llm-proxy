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

# 健康检查脚本
COPY docker_healthcheck.py .
RUN chmod +x docker_healthcheck.py

# 创建非 root 用户
RUN useradd -m -u 1000 llmproxy && \
    chown -R llmproxy:llmproxy /app
USER llmproxy

EXPOSE 4000

# 健康检查：访问 /api/config（无认证可读）
# 健康检查：使用专用脚本，失败时输出详细信息到 stderr（可见于 docker logs）
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/docker_healthcheck.py

# 启动命令：--no-access-log 让 uvicorn 不重复输出 access log（由 access_log_middleware 统一管理）
CMD ["uvicorn", "llm_proxy.main:app", "--host", "0.0.0.0", "--port", "4000", "--no-access-log"]
