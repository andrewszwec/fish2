

# Loading on Mac
import numpy as np
predictions = np.load('/Users/andrew/Downloads/results/Submission_Class_Probabilities.npy')
# Head
predictions[0:3,:]


import pandas as pd
pred = pd.DataFrame(data=predictions,    # values
              index=range(0,1000),    # 1st column as index
            columns=range(0,8))  # 1st row as the column names


filenames = os.listdir('/Users/andrew/Documents/kaggle/fish/data/test_stg1')

files = pd.DataFrame(data=filenames)

result = pd.concat([files, pred], axis=1)

result.columns = ['image',  'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER',   'SHARK',   'YFT']

# Header: [image ALB BET DOL LAG NoF OTHER   SHARK   YFT]
result.to_csv('/Users/andrew/Downloads/results/fish_submission.csv', index = False)



