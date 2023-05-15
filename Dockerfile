FROM python:latest
RUN mkdir myapp/
COPY app.py myapp/app.py
COPY recoomand.py myapp/recoomand.py
COPY data myapp/data
COPY recomand.py myapp/recomand.py
COPY key_extraction.py myapp/key_extraction.py
COPY make_pickle.py myapp/make_pickle.py
COPY model.py myapp/model.py
COPY w2v.py myapp/w2v.py
COPY requirements.txt myapp/requirements.txt

WORKDIR /myapp/
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]