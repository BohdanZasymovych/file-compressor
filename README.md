# How to use compressor

To compress input_file and write compressed data to output_file
```
compress input_file output_file
```

Flags to show size of input_file and output_file
```
--v --verbose
```

Flags to show content of output_file
```
--c --content
```


# Tests
### Compressing text_file_test.txt
20.53kb -> 10.07kb\
Size of a file decreased by 10.46kb or about 50.9 percents

### Compressing image_test.png
4.72kb -> 0.91kb\
Size of a file decreased by 3.81kb or about 80.7 percents


# Comparison
I have compared my algorithm with compression algorithm from python's zlib module
### Results on txt file
![alt text](https://raw.githubusercontent.com/BohdanZasymovych/file-compressor/refs/heads/main/tests/image_comparison.png)