# ==============================================================================
# Base Build
# ==============================================================================
FROM python:3.11.7-slim-bookworm as builder

# Install pipx
RUN python -m pip install --upgrade pip
RUN python -m pip install pipx
RUN python -m pipx ensurepath

# Install poetry
RUN pipx install poetry
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry --version
RUN poetry config virtualenvs.create false --local

# Copy Poetry files
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Create requirements.txt
RUN poetry update
RUN poetry export -f requirements.txt --output requirements.txt

# ==============================================================================
# Final Build
# ==============================================================================

FROM python:3.11.7-bookworm as final

ENV DASH_DEBUG_MODE True

# Copy project
COPY . /app
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Install requirements
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt --trusted-host pypi.python.org --no-cache-dir

# Install gunicorn for production
RUN pip install gunicorn

EXPOSE 8050

WORKDIR /app/src

# Run app
CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server"]
