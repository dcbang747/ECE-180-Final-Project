# ECE-180-Final-Project

In here we strive to create a traditional model that satisfies the requirements of the final project.   
Below is the initial file structure (proposed).  
  
nutribench_rnn/  
│  
├── data/               # the provided CSVs are here  
│   └── raw/  
│       ├── train.csv  
│       ├── val.csv  
│       └── test.csv  
│  
├── src/  
│   ├── __init__.py  
│   ├── config.py  
│   ├── utils.py  
│   ├── data.py  
│   ├── model.py  
│   ├── train.py  
│   ├── evaluate.py  
│   └── predict.py  
│  
├── outputs/  
│   ├── checkpoints/  
│   └── test_with_preds.csv    # created by predict.py  
│  
├── README.md  
└── requirements.txt  
