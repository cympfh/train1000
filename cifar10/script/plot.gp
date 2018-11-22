set terminal pngcairo size 1200,1200
set output dest

set xlabel '#epoch' tc '#808080'
set grid
set tics out nomirror
unset border
set xtics scale 0

set style line 1 lc rgb '#ff4444' lw 1
set style line 2 lc rgb '#884444' lw 2
set style line 3 lc rgb '#ff4444' lw 1
set style line 4 lc rgb '#884444' lw 2
set style line 5 lc rgb '#888888' lw 1  # border
set style line 6 lc rgb '#884444' lw 3

set multiplot layout 2,2 title exname

set title 'Accuracy'
unset yrange
set key right bottom
plot 0.5295 ls 5 ,\
     '<grep ^Epoch ' . src \
     u 2:6 w l ls 1  title 'train' ,\
  '' u 2:10 w l ls 2 title 'test'

set title 'CrossEntropy'
unset yrange
set key left top
plot 1.5017 ls 5 ,\
     '<grep ^Epoch ' . src \
     u 2:8 w l ls 3 title 'train' ,\
  '' u 2:12 w l ls 4 title 'test'

set title 'Accuracy'
set yrange [0.5:0.54]
unset key
plot 0.5295 ls 5 ,\
     '<grep ^Epoch ' . src \
     u 2:6 w l ls 1  title 'train' ,\
  '' u 2:10 w l ls 2 title 'test'

set title 'CrossEntropy'
unset key
set yrange [1.48:1.7]
plot 1.5017 ls 5 ,\
     '<grep ^Epoch ' . src \
     u 2:8 w l ls 3 title 'train' ,\
  '' u 2:12 w l ls 4 title 'test'
