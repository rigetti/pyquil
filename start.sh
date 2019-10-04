#!/bin/bash

echo -e "========================================="
echo -e "=   WELCOME TO THE RIGETTI FOREST SDK   ="
echo -e "========================================="
echo -e "Copyright (c) 2016-2019 Rigetti Computing\n"
/src/quilc/quilc --quiet -R &
/src/qvm/qvm --quiet -S &

exec "$@"