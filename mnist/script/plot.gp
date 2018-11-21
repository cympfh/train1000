set terminal pngcairo size 1200,1200
set output dest

set xlabel '#epoch' tc '#808080'
set grid
set tics out nomirror
unset border
set xtics scale 0

set style line 1 lc rgb '#ff4444' lw 2
set style line 2 lc rgb '#884444' lw 3
set style line 3 lc rgb '#ff4444' lw 2
set style line 4 lc rgb '#884444' lw 3

set multiplot layout 2,2 title exname

set title 'Accuracy'
unset yrange
set key right bottom
plot '<grep ^Epoch ' . src \
     u 2:6 w l ls 1  title 'train' ,\
  '' u 2:10 w l ls 2 title 'test'

set title 'CrossEntropy'
unset yrange
set key right top
plot '<grep ^Epoch ' . src \
     u 2:8 w l ls 3 title 'train' ,\
  '' u 2:12 w l ls 4 title 'test'

set title 'Accuracy'
set yrange [0.92:0.95]
unset key
plot '<grep ^Epoch ' . src \
     u 2:6 w l ls 1  title 'train' ,\
  '' u 2:10 w l ls 2 title 'test'

set title 'CrossEntropy'
unset key
set yrange [0.22:0.42]
plot '<grep ^Epoch ' . src \
     u 2:8 w l ls 3 title 'train' ,\
  '' u 2:12 w l ls 4 title 'test'
