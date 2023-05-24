FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade setuptools 

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "FastApiServer.main:FastApiServer", "--host", "0.0.0.0", "--port", "80"]
