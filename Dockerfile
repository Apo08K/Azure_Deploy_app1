
FROM python:3.11
RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app
RUN pip3 install -r requirements.txt
ADD . /app
EXPOSE 8080
ENTRYPOINT ["streamlit", "run" ,"test_import.py", "--server.port=8080","–-server.address=0.0.0.0"]


#Run this in terminal 
#docker build -t my-app . ## . - current working dira