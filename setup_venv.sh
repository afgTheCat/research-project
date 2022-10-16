#!/bin/sh

set -e
cd `dirname $0`

VENV_DIR=.env
REQUIREMENTS=./requirements.txt

if [ "$COMSPEC" = "" ]; then
    PYTHON=python3
    PIP=pip3
else # Windows
    PYTHON=python
    PIP=pip
fi

rm -rf $VENV_DIR
$PYTHON -m venv $VENV_DIR
. $VENV_DIR/*/activate
$PYTHON -m pip install -U pip
$PIP install -r $REQUIREMENTS
