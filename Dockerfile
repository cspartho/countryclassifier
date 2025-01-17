FROM python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install transformers torch gradio scikit-learn

COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]
