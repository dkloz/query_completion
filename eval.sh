#!/usr/bin/bash

#data="/g/ssli/data/LowResourceLM/aol/queries01.dev.txt.gz"
data="/g/ssli/data/LowResourceLM/aol/queries07.train.txt.gz --data /g/ssli/data/LowResourceLM/aol/queries07.dev.txt.gz --data /g/ssli/data/LowResourceLM/aol/queries07.test.txt.gz"

suffix="7dynamic"

for path in /n/falcon/s0/ajaech/aolexps/g*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"
    
    #cmd="echo $path; python eval.py $path --data $data --threads 4 > $path/$suffix.txt 2> $path/error.$suffix.log"
    #cmd="echo $path; python generanker.py $path --threads 4 > $path/$suffix.txt 2> $path/error.$suffix.log"
    cmd="echo $path; python dynamic.py $path --threads 8 --data $data --limit 700000 > $path/$suffix.txt 2> $path/error.$suffix.log"
    if [ -f $path/$suffix.txt ]; then
        if test $path/model.bin.index -nt $path/$suffix.txt; then
            echo $cmd
        fi
    else
        echo $cmd
    fi
done
