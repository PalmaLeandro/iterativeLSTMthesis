
# coding: utf-8

# ## Recopilacion parcial de resultados

# In[66]:

results_of_one_layer_df = results_df[(results_df['Layers']==1)&(results_df['Units per layer'].isin([200, 650]))].dropna()
g = sns.FacetGrid(results_of_one_layer_df, hue='Units per layer', size=8)
g.map(plt.scatter, 'Iterations', 'Perplexity')
g.map(plt.plot, 'Iterations', 'Perplexity')
g.add_legend()

plt.show()


# In[63]:

g = sns.FacetGrid(results_df[results_df['Iterations']==1].dropna(), hue='Units per layer', size=8)
g.map(plt.scatter, 'Layers', 'Perplexity')
g.map(plt.plot, 'Layers', 'Perplexity')
g.add_legend()

plt.show()


# In[64]:

results_df.sort_values('Perplexity').dropna()

