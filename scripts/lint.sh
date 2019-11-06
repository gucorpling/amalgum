#!/bin/sh
if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 DIRECTORY" >&2
  exit 1
fi
if ! type "xmllint" > /dev/null; then
  echo "You must have xmllint installed."
  exit 1
fi
xmllint $1/*.xml --noout
