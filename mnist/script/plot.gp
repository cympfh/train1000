set terminal pngcairo size 1200,600
set output 'images/' . date . '.png'

set xlabel '#epoch'
set grid ytics
unset border
unset key

set style line 1 lc rgb '#ff4444'
set style line 2 lc rgb '#884444'
set style line 3 lc rgb '#ff4444'
set style line 4 lc rgb '#884444'

set multiplot layout 1,2 title date

set title 'Accuracy'
plot '<grep ^Epoch logs/' . date \
     u 2:6 w l ls 1  title 'train-acc' ,\
  '' u 2:10 w l ls 2 title 'test-acc'

set title 'CrossEntropy'
plot '<grep ^Epoch logs/' . date \
     u 2:8 w l ls 3 title 'train-ent' ,\
  '' u 2:12 w l ls 4 title 'test-ent'
