#https://github.com/xiaohanc/docker-circleci-python3.6-java8/blob/master/Dockerfile
FROM hancheng/circleci-python3.6-java8
RUN sudo pip install -U pip setuptools --user
COPY requirements.txt /tmp/requirements.txt
RUN sudo pip install -U -r /tmp/requirements.txt
RUN sudo python3 -c "from orbdetpy.astrodata import update_data; update_data();"
WORKDIR /home
ENTRYPOINT ["python3"]
