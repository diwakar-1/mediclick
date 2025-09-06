FROM python:3.11-silm
WORKDIR /app
COPY requirement.txt .
RUN pip install -no-cache-dir -r requirement.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn","app.main:app", "--host","0.0.0.0"]
