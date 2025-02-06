FROM gcc:latest

RUN git clone https://github.com/jordiae/exebench.git

RUN apt-get update
RUN apt-get install -y python3 python-is-python3

# working directory for building and running.
WORKDIR /exebench/exebench

COPY run_tests.py .
