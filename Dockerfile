# use multi-stage builds to independently pull dependency versions
ARG quilc_version=1.20.0
ARG qvm_version=1.17.1
ARG python_version=3.7

# use multi-stage builds to independently pull dependency versions
FROM rigetti/quilc:$quilc_version as quilc
FROM rigetti/qvm:$qvm_version as qvm
FROM python:$python_version

ARG pyquil_version
ARG primary_index_url

# copy over the pre-built quilc binary from the first build stage
COPY --from=quilc /src/quilc/quilc /src/quilc/quilc

# copy over the pre-built qvm binary from the second build stage
COPY --from=qvm /src/qvm/qvm /src/qvm/qvm

# install the missing apt packages that aren't copied over
RUN apt-get update && apt-get -yq dist-upgrade && \
    apt-get install --no-install-recommends -yq \
    git libblas-dev libffi-dev liblapack-dev libzmq3-dev && \
    rm -rf /var/lib/apt/lists/*

# install ipython
RUN pip install --no-cache-dir ipython

# install pyquil
RUN pip install --index-url $primary_index_url --extra-index-url https://pypi.org/simple pyquil==$pyquil_version

# use an entrypoint script to add startup commands (qvm & quilc server spinup)
COPY ./entrypoint.sh /src/pyquil/entrypoint.sh
ENTRYPOINT ["/src/pyquil/entrypoint.sh"]
CMD ["ipython"]
