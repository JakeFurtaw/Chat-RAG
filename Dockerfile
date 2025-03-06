FROM nvcr.io/nvidia/pytorch:25.02-py3
WORKDIR /home/jake/Programming/Personal/Chat-RAG

COPY . .
RUN true
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000

RUN useradd app
USER app

CMD ["uvicorn", "app.demo.launch:app", "--host", "0.0.0.0", "--port", "8080"]