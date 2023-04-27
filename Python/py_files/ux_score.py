#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os


# In[96]:


def ux_score(score_path:str,model_name:str,p:float,c:float,s:float):
    
    for score in [p, c, s]:
        if score > 7 or score < 1:
            print("One of the scores is out of range. Please use numbers (1...7)")
            return
    
    # score is adjusted from 1-7 to 0-6.
    max_score = 18
    ux_score = (2*(p + c + s - 3) / max_score)-1

    scores_dict = {
        'Name': model_name, 
        'Predictability': p,
        'Correlation': c,
        'Satisfaction': s,
        'UX Score': f'{ux_score:.2f}'
    }

    # Convert to data frame.
    scores_frame = pd.DataFrame(scores_dict, index=[0])
    # Check if score csv exists.
    if (os.path.exists(score_path)):
        df = pd.read_csv(score_path, index_col=0)
        df = pd.concat([df, scores_frame], ignore_index=True)
        df.to_csv(score_path)

    else:
        scores_frame.to_csv(score_path)


# In[191]:


metrics = pd.read_csv("./models/systematic_metrics.csv")

ux_scores_path = "./models/systematic/ux_scores.csv"

# Use discrete values (1-7). Set all to 1 if accuracy is below 70%.
p = 6 # How much can you influence the model on purpose? 4 = neutral
c = 4 # How much does the music correlate to what you are doing in game? 4 = neutral
s = 3 # How does that influence the experience? 4 = neutral

ux_score(ux_scores_path,f"Model {metrics.shape[0]-1}", p, c, s)


# In[ ]:




