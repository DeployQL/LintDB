set -e

LD_DEBUG=libs python -c "import lintdb" 2>&1 | grep pylintdb.so