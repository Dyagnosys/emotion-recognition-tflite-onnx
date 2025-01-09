FROM tensorflow/tensorflow:2.12.0-cpu

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main_tflite.py"]