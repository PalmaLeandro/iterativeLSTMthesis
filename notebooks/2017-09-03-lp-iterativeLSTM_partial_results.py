
# coding: utf-8

# ## Recopilacion parcial de resultados

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
 
sns.set_context("talk", font_scale=1.4)


# In[3]:


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
                            'Perplexity': 89.56},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 4,
                            'Perplexity': 89.97},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 5,
                            'Perplexity': 89.94},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 6,
                            'Perplexity': 90.53},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 7,
                            'Perplexity': 93.28},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 8,
                            'Perplexity': 94.14},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 9,
                            'Perplexity': 98.23},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 14 epochs',
                            'Units per layer': 200,
                            'Layers': 1,
                            'Iterations': 10,
                            'Perplexity': 100.1},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 1,
                            'Perplexity': 83.61},
                          {
                            'Training epochs': 55, 
                            'Learning rate scheme': 'decay by 0.86 after 6 epochs',
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
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 5, #??
                            'Iterations': 1,
                            'Perplexity': 97.15},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 2,
                            'Perplexity': 81.05}, 
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
                            'Perplexity': 79.22},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 5,
                            'Perplexity': 78.52},# HP. c' = c.. & out=tanh(newh+in)
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 6,
                            'Perplexity': 78.76},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 7,
                            'Perplexity': 78.51},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 8,
                            'Perplexity': 78.62},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2, #??
                            'Iterations': 1,
                            'Perplexity': 84.06},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 2,
                            'Perplexity': 81.28}, 
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 3,
                            'Perplexity': 79.75}, 
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 4,
                            'Perplexity': 79.19},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 5,
                            'Perplexity': 78.78},# HP. c' = c.. & out=tanh(newh+in)
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 6,
                            'Perplexity': 78.87},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 7,
                            'Perplexity': 78.64},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 8,
                            'Perplexity': 81.12},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 9,
                            'Perplexity': 79.02},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 2,
                            'Iterations': 10,
                            'Perplexity': None},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 9,
                            'Perplexity': 78.88},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 650,
                            'Layers': 1,
                            'Iterations': 10,
                            'Perplexity': 78.50},# HP. c' = tanh(c).. & out=tanh(newh+in)
                          {
                            'Training epochs': 55, 
                            'Learning rate scheme': 'decay by 0.86 after 14 epochs',
                            'Units per layer': 1500,
                            'Layers': 2,
                            'Iterations': 1,
                            'Perplexity': 78.29},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 325,
                            'Layers': 1,
                            'Iterations': 1,
                            'Perplexity': 84.56},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 325,
                            'Layers': 1,
                            'Iterations': 2,
                            'Perplexity': 83.05},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 325,
                            'Layers': 1,
                            'Iterations': 4,
                            'Perplexity': 82.11},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 250,
                            'Layers': 1,
                            'Iterations': 5,
                            'Perplexity': 84.12},
                          {'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 250,
                            'Layers': 1,
                            'Iterations': 6,
                            'Perplexity': 83.65},
                          {
                            'Training epochs': 40, 
                            'Learning rate scheme': 'decay by 0.8 after 6 epochs',
                            'Units per layer': 250,
                            'Layers': 1,
                            'Iterations': 7,
                            'Perplexity': 91.41}])
results_df['Total Units'] = results_df['Layers'] * results_df['Units per layer']


# In[4]:


df = results_df[(results_df['Layers'].isin([1,2]))&(results_df['Units per layer'].isin([650]))].dropna()
g = sns.FacetGrid(df.groupby(['Iterations','Layers'])[ 'Perplexity'].min().reset_index(), hue='Layers', size=8, 
                 hue_kws={'color': ['b', 'r'], "ls" : ["-","--"]})
#g.map(plt.scatter, 'Iterations', 'Perplexity')
g.map(plt.plot, 'Iterations', 'Perplexity')
g.fig.suptitle('Performance versus iterations made', fontsize=25, y=1.0)
g.add_legend()
plt.xticks(range(11))
plt.show()


# In[4]:


g = sns.FacetGrid(results_df[results_df['Iterations']==1].dropna(), hue='Units per layer', size=8)
g.map(plt.scatter, 'Layers', 'Perplexity')
g.map(plt.plot, 'Layers', 'Perplexity')
g.add_legend()

plt.show()


# In[5]:


results_df.sort_values('Perplexity').dropna()


# In[ ]:




