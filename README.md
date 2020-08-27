# dali_examples
Parallelize ExternalSource to maximize loading pipeline

### Note:
This example expects clean data

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
97.7        81288488231   1           81288488231.0   81288488231   81288488231   run
```

Version 2 (**--num_read_processes=16 --batch_size=16 --num_gpus=4**)
```bash
Time(%)     Time (ns)     Instances   Avg (ns)        Min (ns)      Max (ns)      Range 
----------  ------------  ----------  --------------  ------------  ------------  ------
100.0       26487050531   1           26487050531.0   26487050531   26487050531   run
```

Version 3 (**--num_read_processes=16 --batch_size=16**)
```bash
Time(%)     Time (ns)     Instances   Avg (ns)        Min (ns)      Max (ns)      Range                                                         
----------  ------------  ----------  --------------  ------------  ------------  ------
93.4        35934472941   1           35934472941.0   35934472941   35934472941   run 
```

Version 4 (**--num_read_processes=16 --batch_size=16 --num_gpus=4**)
```bash
Time(%)     Time (ns)     Instances   Avg (ns)        Min (ns)      Max (ns)      Range 
----------  ------------  ----------  --------------  ------------  ------------  ------
100.0       16306502147   1           16306502147.0   16306502147   16306502147   run
```

1. V2 is ~3.0x faster than V1
2. V3 is ~2.2x faster than V1
3. V4 is ~5.0x faster than V1
