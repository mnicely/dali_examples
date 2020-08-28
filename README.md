# dali_examples
Parallelize ExternalSource to maximize the loading pipeline

### Note:
1. This example expects clean data
2. More comments in files

### Environment
```bash
conda create --name dali
conda activate dali
conda install cupy -y
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly nvidia-dali-weekly-cuda100
pip install aiapy
```

### Input data
https://drive.google.com/file/d/1IMiCcm49WEw_cyJF4GCZLVW9_Gq77V-2/view?usp=sharing

### Results (Nsight Systems on DGX1 w/V100s)
Version 1 (**--num_read_processes=16 --batch_size=16**)
```bash
Time(%)     Time (ns)     Instances   Avg (ns)        Min (ns)      Max (ns)      Range                                                         
----------  ------------  ----------  --------------  ------------  ------------  ------
96.9        70224833901   1           70224833901.0   70224833901   70224833901   run
```

Version 2 (**--num_read_processes=16 --batch_size=16 --num_gpus=4**)
```bash
Time(%)     Time (ns)     Instances   Avg (ns)        Min (ns)      Max (ns)      Range 
----------  ------------  ----------  --------------  ------------  ------------  ------
100.0       23609338458   1           23609338458.0   23609338458   23609338458   run
```

Version 3 (**--num_read_processes=16 --batch_size=16**)
```bash
Time(%)     Time (ns)     Instances   Avg (ns)        Min (ns)      Max (ns)      Range                                                         
----------  ------------  ----------  --------------  ------------  ------------  ------
95.0        34849683649   1           34849683649.0   34849683649   34849683649   run 
```

Version 4 (**--num_read_processes=16 --batch_size=16 --num_gpus=4**)
```bash
Time(%)     Time (ns)     Instances   Avg (ns)        Min (ns)      Max (ns)      Range 
----------  ------------  ----------  --------------  ------------  ------------  ------
100.0       15935145551   1           15935145551.0   15935145551   15935145551   run
```

1. v2 is ~3.0x faster than v1
2. v3 is ~2.2x faster than v1
3. v4 is ~5.0x faster than v1
