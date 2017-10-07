
# coding: utf-8

# ## Recopilacion parcial de resultados

# In[25]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:

results_df = pd.DataFrame([{
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 1,
                            'Perplexity': 92.86},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 2,
                            'Perplexity': 90.41},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 3,
                            'Perplexity': 91.20},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 4,
                            'Perplexity': 90.74},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 5,
                            'Perplexity': 90.34},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 6,
                            'Perplexity': 88.14},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 7,
                            'Perplexity': 90.15},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 8,
                            'Perplexity': 89.18},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 9,
                            'Perplexity': 90.82},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 325,
                            'Layers': 1,
                            'Iterations': 1,
                            'Perplexity': 84.56},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 325,
                            'Layers': 1,
                            'Iterations': 2,
                            'Perplexity': 83.05},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 325,
                            'Layers': 1,
                            'Iterations': 4,
                            'Perplexity': 82.11},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 250,
                            'Layers': 1,
                            'Iterations': 5,
                            'Perplexity': 84.12},
                          {'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 250,
                            'Layers': 1,
                            'Iterations': 6,
                            'Perplexity': 83.65},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 250,
                            'Layers': 1,
                            'Iterations': 7,
                            'Perplexity': 91.41},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 1,
                            'Perplexity': 83.61},
                          {
                            'Training epochs': 55, 
                            'Learning rate scheme': 'decay by 0.86 after 14 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 1,
                            'Perplexity': 83.25},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 3, #??
                            'Iterations': 1,
                            'Perplexity': 85.92},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 4, #??
                            'Iterations': 1,
                            'Perplexity': 91.16},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 650,
                            'Layers': 5, #??
                            'Iterations': 1,
                            'Perplexity': 97.15},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 1,
                            'Perplexity': 83.61},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 2,
                            'Perplexity': 81.06}, 
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 3,
                            'Perplexity': 79.65}, 
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 4,
                            'Perplexity': 79.22}, #doing at mac
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 5,
                            'Perplexity': 78.53},# HP. c' = c.. & out=tanh(newh+in)
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 6,
                            'Perplexity': 78.73},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 7,
                            'Perplexity': 78.50},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 8,
                            'Perplexity': 78.49},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 9,
                            'Perplexity': 78.88},# HP. c' = tanh(c).. & out=tanh(newh+in)
                          {
                            'Training epochs': 55, 
                            'Learning rate scheme': 'decay by 0.86 after 14 epochs',
                            'Units per layer': 1500,
                            'Layers': 2,
                            'Iterations': 1,
                            'Perplexity': 78.29}])
results_df['Total Units'] = results_df['Layers'] * results_df['Units per layer']


# In[34]:

g = sns.FacetGrid(results_df[results_df['Layers']==1].dropna(), hue='Units per layer', size=8)
g.map(plt.scatter, 'Iterations', 'Perplexity')
g.map(plt.plot, 'Iterations', 'Perplexity')
g.add_legend()

plt.show()


# In[10]:

g = sns.FacetGrid(results_df[results_df['Iterations']==1].dropna(), hue='Units per layer', size=8)
g.map(plt.scatter, 'Layers', 'Perplexity')
g.map(plt.plot, 'Layers', 'Perplexity')
g.add_legend()

plt.show()


# In[5]:

results_df.sort_values('Total Units').dropna()


# In[ ]:



