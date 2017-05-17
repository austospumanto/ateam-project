#!/bin/bash
# This script will fetch the TIDIGITs dataset from Stanford AFS, with proper authentication
# IMPORTANT: do NOT check the dataset into version control, do NOT share the dataset with anyone,
# and do NOT use the dataset for anything other than this project. Delete the data once you have
# finished using it.

echo "Type in your SUNet ID (e.g. 'viggy'), followed by [Enter]:"

read sunetid

scp -r $sunetid@corn.stanford.edu:/afs/ir.stanford.edu/data/linguistic-data/ldc/LDC93S10_TIDIGITS data/
