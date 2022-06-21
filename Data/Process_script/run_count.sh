#!/bin/bash
#testinput.sh
echo "##################row count is:#####################:"
less $1 | cut -f 1 | sort -u | wc 

h=${2:-' '}

echo "###################column count is:##############"
less $1 | head -n 1 | awk -F "$h" '{print NF}'

echo ">>>>>>>>>>>>>>>>>>>>>>>>less -S<<<<<<<<<<<<<<<<<<"
less $1 | cut -f 1 | sort -u | cat
