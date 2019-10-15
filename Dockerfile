# specify the dependency versions (can be overriden with --build-arg)
ARG quilc_version=1.12.1
ARG qvm_version=1.12.0
ARG python_version=3.6

# use multi-stage builds to independently pull dependency versions
FROM rigetti/quilc:$quilc_version as quilc
FROM rigetti/qvm:$qvm_version as qvm
FROM python:$python_version

# copy over the pre-built quilc binary and tweedledum library from the first build stage
COPY --from=quilc /usr/local/lib/libtweedledum.so /usr/local/lib/libtweedledum.so
COPY --from=quilc /src/quilc/quilc /src/quilc/quilc

# copy over the pre-built qvm binary from the second build stage
COPY --from=qvm /src/qvm/qvm /src/qvm/qvm

# install the missing apt packages that aren't copied over
RUN apt-get update && apt-get -yq dist-upgrade && \
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
ENTRYPOINT ["/src/pyquil/entrypoint.sh"]
CMD ["ipython"]
