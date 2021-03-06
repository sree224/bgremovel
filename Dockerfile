FROM python:latest
WORKDIR /testapp
ADD . /testapp
RUN pip3 install -r requirements.txt
CMD [ "python", "app.py" ]