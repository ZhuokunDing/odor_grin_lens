#!/bin/bash
cd /src || {
	echo "cd failed"
	exit
}
git config --global --add safe.directory '*'
cd /src || {
	echo "cd failed"
	exit
}
jupyter lab build
jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token="${JUPYTER_PASSWORD:-}" --no-browser
