# function 'compute' is distributed and executed with arguments
# supplied with 'cluster.submit' below
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from functools import partial
import time
import logging

start_time = time.time()

tstamp = str(time.time())
log_file = './logs/exec_log_knn_mp'+tstamp+'.log'
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

# size of the catalog we want to load
n = 100000

logging.info('Loading data from file...')

# on charge puis génère le corpus
corpus = pd.read_csv('/data/testdata.csv', sep='\t', usecols=['product_name'], nrows=n).dropna()
# on calcule le tf-idf
vectorizer_pn = TfidfVectorizer(smooth_idf=True, norm=None, strip_accents='unicode', max_df=0.1)
vectorizer_pn.fit(corpus['product_name'].values)
#vectorizer_pn = joblib.load('pickles/idf_product_name_l2.pkl')

cat1 = pd.read_csv('/data/testdata.csv', sep='\t', usecols=['product_id', 'product_name'], nrows=n)
cat1.fillna(' ', inplace=True)
#cat2 = cat1.copy()
logging.info('... Done')

# we vectorize the product names using the tf-idf
logging.info('TF-IDF vectorization and KNN model fit...')
tfidf1 = vectorizer_pn.transform(cat1.product_name.values)

# we fit a KNN model on this data
nbrs = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
nbrs.fit(tfidf1)
logging.info('... Done')

def kneighbors_mp(input_data, knn):
    import time, socket
    import numpy as np
    chunksize = 500
    m = input_data.shape[0]
    # we generate the list of chunk sizes
    l_cs = (m // chunksize)*[chunksize]
    if m%chunksize!=0 : 
        l_cs += [m%chunksize]
    indices_list = []
    k = 0
    for cs in l_cs:
        ind = knn.kneighbors(input_data[k:k+cs], return_distance=False)
        indices_list.append(ind)
        k += cs
        
    host = socket.gethostname()
    return (host, np.concatenate(indices_list))

njobs = 4
nn = 10000#tfidf1.shape[0]

# we distribute manually the load
loads = njobs * [nn // njobs]
if nn % njobs != 0:
    rest = nn % njobs
    loads[:rest] = [loads[i] + 1 for i in range(rest)]
i = 0
datarray = []
for l in loads:
    datarray.append(tfidf1[i:i+l])
    i += l

logging.info('Computing NN...')

if __name__ == '__main__':
    # executed on client only; variables created below, including modules imported,
    # are not available in job computations
    import dispy
    # distribute 'compute' to nodes; 'compute' does not have any dependencies (needed from client)
    cluster = dispy.JobCluster(kneighbors_mp)
    # import dispy's httpd module, create http server for this cluster
    import dispy.httpd
    http_server = dispy.httpd.DispyHTTPServer(cluster)
    
    # run 'compute' with 20 random numbers on available CPUs
    res = []
    jobs = []
    for i in range(njobs):
        job = cluster.submit(datarray[i], nbrs)
        job.id = i # associate an ID to identify jobs (if needed later)
        jobs.append(job)

    for job in jobs:
        host, l = job() # waits for job to finish and returns results
        res.append(l)
        print('%s executed job %s at %s with %s' % (host, job.id, job.start_time, l.size))
        # other fields of 'job' that may be useful:
        # job.stdout, job.stderr, job.exception, job.ip_addr, job.end_time
    cluster.print_status()  # shows which nodes executed how many jobs etc.
    cluster.wait() # wait for all jobs to finish
    #print(res)
    http_server.shutdown() # this waits until browser gets all updates
    cluster.close()
