#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
from six import iteritems
from web.datasets.analogy import *
from web.embeddings import *
from web.evaluate import *


# In[2]:


# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


# In[3]:


w_glove = fetch_GloVe(corpus="wiki-6B", dim=300)


# In[7]:


t_glove = fetch_GloVe(corpus="twitter-27B", dim=100)


# In[2]:


cc_glove = fetch_GloVe(corpus="common-crawl-840B", dim=300)


# In[9]:


# Fetches SG (skip-gram with negative sampling) embeddings trained on GoogleNews dataset published on word2vec website
SG_googleNews = fetch_SG_GoogleNews()


# In[10]:


LexVec = fetch_LexVec("commoncrawl-W+C")


# In[11]:


CC_LexVec = fetch_LexVec("commoncrawl-ngramsubwords-W")


# In[13]:


PDC = fetch_PDC()


# In[14]:


HDC = fetch_HDC()


# In[15]:


conceptnet_numberbatch = fetch_conceptnet_numberbatch()


# In[16]:


FastText = fetch_FastText()


# In[17]:


# Calculate results on analogy
    
analogy_tasks = {
    "MSR": fetch_msr_analogy(),
#     "wordrep":fetch_wordrep()
    "google_analogy":  fetch_google_analogy()
#     "semeval_2012_2": fetch_semeval_2012_2()
}

   


# In[19]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(FastText, data.X, data.y)
    print("Analogy prediction accuracy on FastText {} {}".format(name, analogy_results[name]))


# In[20]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(w_glove, data.X, data.y)
    print("Analogy prediction accuracy on wiki - glove {} {}".format(name, analogy_results[name]))


# In[21]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(t_glove, data.X, data.y)
    print("Analogy prediction accuracy on twitter glove {} {}".format(name, analogy_results[name]))


# In[22]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(cc_glove, data.X, data.y)
    print("Analogy prediction accuracy on common-crawl glove {} {}".format(name, analogy_results[name]))


# In[23]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(SG_googleNews, data.X, data.y)
    print("Analogy prediction accuracy on SG_googleNews {} {}".format(name, analogy_results[name]))


# In[24]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(LexVec, data.X, data.y)
    print("Analogy prediction accuracy on LexVec {} {}".format(name, analogy_results[name]))


# In[25]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(CC_LexVec, data.X, data.y)
    print("Analogy prediction accuracy on CC_LexVec {} {}".format(name, analogy_results[name]))


# In[26]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(PDC, data.X, data.y)
    print("Analogy prediction accuracy on PDC {} {}".format(name, analogy_results[name]))


# In[27]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(HDC, data.X, data.y)
    print("Analogy prediction accuracy on HDC {} {}".format(name, analogy_results[name]))


# In[28]:


analogy_results = {}
for name, data in iteritems(analogy_tasks):
    analogy_results[name] = evaluate_analogy(conceptnet_numberbatch, data.X, data.y)
    print("Analogy prediction accuracy on conceptnet_numberbatch {} {}".format(name, analogy_results[name]))


# In[31]:


print("Analogy prediction accuracy on FastText")
evaluate_on_semeval_2012_2(FastText)["all"]


# In[32]:


print("Analogy prediction accuracy on wiki-glove")
evaluate_on_semeval_2012_2(w_glove)["all"]


# In[33]:


print("Analogy prediction accuracy on twitter-golve")
evaluate_on_semeval_2012_2(t_glove)["all"]


# In[34]:


print("Analogy prediction accuracy on  common-crawl glove")
evaluate_on_semeval_2012_2(cc_glove)["all"]


# In[35]:


print("Analogy prediction accuracy on SG_googleNews")
evaluate_on_semeval_2012_2(SG_googleNews)["all"]


# In[36]:


print("Analogy prediction accuracy on LexVec")
evaluate_on_semeval_2012_2(LexVec)["all"]


# In[37]:


print("Analogy prediction accuracy on common-crawl LexVec")
evaluate_on_semeval_2012_2(CC_LexVec)["all"]


# In[38]:


print("Analogy prediction accuracy on PDC")
evaluate_on_semeval_2012_2(PDC)["all"]


# In[39]:


print("Analogy prediction accuracy on HDC")
evaluate_on_semeval_2012_2(HDC)["all"]


# In[40]:


print("Analogy prediction accuracy on conceptnet_numberbatch")
evaluate_on_semeval_2012_2(conceptnet_numberbatch)["all"]


# In[ ]:


print("Analogy prediction accuracy on FastText ")
evaluate_on_WordRep(FastText)


# In[ ]:


print("Analogy prediction accuracy on  wiki-glove ")
evaluate_on_WordRep(w_glove)


# In[ ]:


print("Analogy prediction accuracy on twitter-glove")
evaluate_on_WordRep(t_glove,max_pairs = 200)


# In[ ]:


print("Analogy prediction accuracy on common-crawl glove")
evaluate_on_WordRep(cc_glove,max_pairs = 200)


# In[ ]:


print("Analogy prediction accuracy on SG_googleNews")
evaluate_on_WordRep(SG_googleNews,max_pairs = 200)


# In[ ]:


print("Analogy prediction accuracy on LexVec")
evaluate_on_WordRep(LexVec,max_pairs = 200)


# In[ ]:


print("Analogy prediction accuracy on common-crawl LexVec")
evaluate_on_WordRep(CC_LexVec,max_pairs = 200)


# In[ ]:


print("Analogy prediction accuracy on PDC")
evaluate_on_WordRep(PDC,max_pairs = 200)


# In[ ]:


print("Analogy prediction accuracy on HDC")
evaluate_on_WordRep(HDC,max_pairs = 200)


# In[ ]:


print("Analogy prediction accuracy on conceptnet_numberbatch")
evaluate_on_WordRep(conceptnet_numberbatch,max_pairs = 200)["all"]


# In[ ]:





# In[ ]:





# In[ ]:




