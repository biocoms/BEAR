# BEAR - Bootstrap and Attribute Ranking

## conda environment

## running the pipeline

 ```bash
 sh run_pipeline.sh $input_file_path $number_of_features
```

### example:
```bash
sh run_pipeline.sh inputs/test/multi_data.csv 100
```

## submitting as a job file
```bash
qsub pipeline_trail.job
```
