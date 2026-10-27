FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Evita arquivos .pyc e força logs em tempo real no console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Copia os arquivos de configuração de dependências primeiro (aproveita o cache do Docker)
COPY pyproject.toml uv.lock ./

# Instala apenas as dependências de produção usando o uv.lock (sem instalar o pacote raiz)
RUN uv sync --frozen --no-dev --no-install-project

# Copia o código-fonte e os modelos treinados para dentro do container
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Executa a API FastAPI utilizando o uv run
CMD ["uv", "run", "--no-dev", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Instalar o pacote de idiomas exigido pelo ONNX Runtime (C++)
RUN apt-get update && apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

# Definir as variáveis de ambiente do idioma
ENV LANG="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"