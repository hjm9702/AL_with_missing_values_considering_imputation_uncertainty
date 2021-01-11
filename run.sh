#! /bin/bash

for i in {0..19}
do
  for j in {0..19}
  do
    for k in {0..4}
    do
      for l in {0..2}
      do
        for m in {0..6}
        do
          python main.py $i $j $k $l $m
        done
      done
    done
  done
done