# Assignment 2: Argument Detection Classifier

This project is part of the Advanced NLP course and focuses on identifying and classifying arguments within sentences using logistic regression.


## Usage
``` bash
python3 A2_main.py *train-data-path* *test-data-path* *output-folder-path* *max_iter-number*
```

example: `python3 A2_main.py data/en_ewt-up-train.conllu data/en_ewt-up-test.conllu output 1000`

User will be asked to chose if they want to use GPU when executing.

**If you would like to use GPU, you will need a cuda-compatible device and then uncomment the lines in the requirements.txt**



## Contents

- `A2_main.py`: The main script for the task.

- `A2_main.ipynb`: A step-by-step separated notebook for the entire process.

- data

   - `en_ewt-up-train.conllu`: Training dataset

   - `en_ewt-up-dev.conllu`: Developement dataset

   - `en_ewt-up-test.conllu`: Test dataset
  
- output

   - `singlelogreg.csv`: example results from single model trained on cpu with max_iter = 100
     
   - `doublelogreg.csv`: example results from double model trained on cpu with max_iter = 100
   
   - `GPUsinglelogreg.csv`:  example results from single model trained on gpu with max_iter = 50000
   
   - `GPUdoublelogreg.csv`: example results from double model trained on gpu with max_iter = 50000



