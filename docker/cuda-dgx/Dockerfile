FROM nvidia/cuda:9.0-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
	git python cmake vim

# build trng
WORKDIR /tmp
RUN git clone https://github.com/rabauke/trng4.git
RUN cd trng4 && ./configure && make && make install

# build spdlog
RUN git clone https://github.com/gabime/spdlog.git
RUN cd spdlog && mkdir build && cd build && \
	cmake .. && make && make install

# build json
RUN git clone https://github.com/nlohmann/json.git
RUN cd json && mkdir build && cd build && \
	cmake .. && make && make install

# install gcc-6
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt install g++-6 -y

# set env
ENV CXX=g++-6
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/$LD_LIBRARY_PATH


WORKDIR /
