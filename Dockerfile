# specify the dependency versions (can be overriden with --build-arg)
ARG quicklisp_version=2019-07-11
ARG rpcq_version=2.7.3
ARG quilc_version=1.12.0
ARG qvm_version=1.12.0
ARG python_version=3.6

# use multi-stage builds to independently pull dependency versions
FROM rigetti/lisp:$quicklisp_version as lisp
FROM rigetti/rpcq:$rpcq_version as rpcq
FROM rigetti/quilc:$quilc_version as quilc
FROM rigetti/qvm:$qvm_version as qvm
FROM python:$python_version

# copy over SBCL and Quicklisp from the first build stage
COPY --from=lisp /src/sbcl /src/sbcl
COPY --from=lisp /usr/local/bin/sbcl /usr/local/bin/sbcl
COPY --from=lisp /usr/local/lib/sbcl /usr/local/lib/sbcl
COPY --from=lisp /root/quicklisp /root/quicklisp

# copy over rpcq source from the second build stage
COPY --from=rpcq /src/rpcq /src/rpcq

# copy over the quilc source from the third build stage
COPY --from=quilc /src/quilc /src/quilc

# copy over the pre-built qvm binary from the fourth build stage
COPY --from=qvm /src/qvm /src/qvm

# install the missing apt requirements that can't be copied over
RUN apt-get update && apt-get -yq dist-upgrade && \
    apt-get install --no-install-recommends -yq \
    clang-7 git libblas-dev libffi-dev liblapack-dev libzmq3-dev && \
    rm -rf /var/lib/apt/lists/*

# rebuild the quilc binary to fix tweedledum linkage
WORKDIR /src/quilc
RUN CXX=clang++-7 make && make install-tweedledum && ldconfig

# install ipython
RUN pip install --no-cache-dir ipython

# copy over files and install requirements
ADD . /src/pyquil
WORKDIR /src/pyquil
RUN pip install -e .

# use an entrypoint script to add startup commands (qvm & quilc server spinup)
ENTRYPOINT ["./entrypoint.sh"]
CMD ["ipython"]
