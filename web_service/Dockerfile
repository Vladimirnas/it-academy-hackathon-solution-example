FROM python:3.13-slim

WORKDIR /app

COPY server.py .
COPY static ./static

ENV PORT=80
ENV STATIC_ROOT=/app/static

EXPOSE 80

CMD ["python", "server.py"]
