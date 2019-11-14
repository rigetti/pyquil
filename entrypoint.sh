#!/bin/bash

echo -e "========================================="
echo -e "=   WELCOME TO THE RIGETTI FOREST SDK   ="
echo -e "========================================="
echo -e "Copyright (c) 2016-2019 Rigetti Computing\n"

/src/quilc/quilc --quiet --check-sdk-version=no -R 2> quilc.log &
/src/qvm/qvm --quiet --check-sdk-version=no -S 2> qvm.log &

exec "$@"
