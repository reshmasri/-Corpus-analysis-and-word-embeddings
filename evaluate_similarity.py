#!/usr/bin/env python
# coding: utf-8

# In[6]:


import logging
from six import iteritems
from web.datasets.similarity import *
from web.embeddings import *
from web.evaluate import evaluate_similarity
import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


# In[7]:


# Fetch GloVe embedding (warning: it might take few minutes)
w_glove = fetch_GloVe(corpus="wiki-6B", dim=300)


# In[10]:


t_glove = fetch_GloVe(corpus="twitter-27B", dim=100)


# In[11]:


cc_glove = fetch_GloVe(corpus="common-crawl-840B", dim=300)


# In[12]:


# Fetches SG (skip-gram with negative sampling) embeddings trained on GoogleNews dataset published on word2vec website
SG_googleNews = fetch_SG_GoogleNews()


# In[13]:


LexVec = fetch_LexVec("commoncrawl-W+C")


# In[14]:


CC_LexVec = fetch_LexVec("commoncrawl-ngramsubwords-W")


# In[15]:


PDC = fetch_PDC()


# In[16]:


HDC = fetch_HDC()


# In[17]:


conceptnet_numberbatch = fetch_conceptnet_numberbatch()


# In[30]:


FastText = fetch_FastText()


# In[19]:


# Define tasks
tasks = {
    "MTurk":fetch_MTurk(),
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "RG65":fetch_RG65(),
    "RW":fetch_RW(),
#     "multilingual_SimLex999":fetch_multilingual_SimLex999(),
    "SIMLEX999": fetch_SimLex999(),
    "TR9856":fetch_TR9856()
}


# In[20]:


# Print sample data
for name, data in iteritems(tasks):
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1], data.y[0]))


# In[31]:


# Calculate results using helper function -Fast Text
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(FastText, data.X, data.y)))
#     print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(t_glove, data.X, data.y)))


# In[21]:


# Calculate results using helper function - wiki glove
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(w_glove, data.X, data.y)))
#     print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(t_glove, data.X, data.y)))


# In[22]:


# Calculate results using helper function - twitter glove
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(t_glove, data.X, data.y)))


# In[23]:


# Calculate results using helper function 
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(CC_LexVec, data.X, data.y)))


# In[24]:


# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(cc_glove, data.X, data.y)))


# In[25]:


# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(SG_googleNews, data.X, data.y)))


# In[26]:


# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(PDC, data.X, data.y)))


# In[27]:


# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(conceptnet_numberbatch, data.X, data.y)))


# In[28]:


# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(HDC, data.X, data.y)))


# In[29]:


# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(LexVec, data.X, data.y)))


# In[ ]:





# In[ ]:




