# FROM python:3.6-alpine
# RUN apk update

# WORKDIR /app
# COPY . /app
# RUN apk add --no-cache build-base
# RUN pip3 --no-cache-dir install -r requirements.txt --ignore-installed six

# EXPOSE 3000

# # ENTRYPOINT [ "python3" ]
# # CMD [ "app.py" ]
# CMD [ "uwsgi", "--ini" "wsgi.ini" ]

FROM ubuntu:20.04

RUN apt-get upgrade && apt-get update

RUN apt-get install -y python3-pip 

ADD requirements.txt requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

WORKDIR /app
COPY . /app

EXPOSE 5000

# RUN chmod +x /app/feature-download.sh

# ENTRYPOINT ["/app/feature-download.sh"]

# RUN curl -o feature-config.json http://13.233.109.194:4001/tempfunction/message_server | grep -Po '(?<=url: )[^,]*'

# CMD ["uwsgi", "--ini", "wsgi.ini"]

# CMD ["flask", "--app", "app.py","--debug","run","--port=5000"]

CMD ["python3", "API/app.py"]