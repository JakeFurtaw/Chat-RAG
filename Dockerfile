FROM nvcr.io/nvidia/pytorch:25.02-py3
WORKDIR /home/jake/Programming/Personal/Chat-RAG

COPY . .
RUN true
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
EXPOSE 5000

RUN useradd Jake
USER Jake

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080"]