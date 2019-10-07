# specify the dependency versions (can be overriden with --build-arg)
ARG quilc_version=1.12.0
ARG qvm_version=1.12.0
ARG python_version=3.6

# use multi-stage builds to independently pull dependency versions
FROM rigetti/quilc:$quilc_version as quilc
FROM rigetti/qvm:$qvm_version as qvm
FROM python:$python_version

# copy over the pre-built quilc binary from the first build stage
COPY --from=quilc /src/quilc /src/quilc

# copy over the pre-built qvm binary from the second build stage
COPY --from=qvm /src/qvm /src/qvm

# install the missing apt requirements that can't be copied over
RUN echo "deb http://http.us.debian.org/debian/ testing non-free contrib main" >> /etc/apt/sources.list && \
    apt-get update && apt-get -yq dist-upgrade && \
    apt-get install --no-install-recommends -yq \
    clang-7 git libblas-dev libffi-dev liblapack-dev libzmq3-dev && \
    rm -rf /var/lib/apt/lists/*

# install ipython
RUN pip install --no-cache-dir ipython

# copy over files and install requirements
ADD . /src/pyquil
WORKDIR /src/pyquil
RUN pip install -e .

# use an entrypoint script to add startup commands (qvm & quilc server spinup)
ENTRYPOINT ["./entrypoint.sh"]
CMD ["ipython"]
