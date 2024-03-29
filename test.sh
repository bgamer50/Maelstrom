for f in test/bin/*.exe; do
    echo $f
    ./$f
done

for f in test/bin/dist/*.exe; do
    echo $f
    mpirun --allow-run-as-root -np 2 $f
done