import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Supervised:
    '''
    Get performane metrics of supervised benchmarks
    
    Args:
        test_idx (array): index for test documents
        genre (array): genres of interest 

    Attributes:
        test_idx (array): index for test documents
        genre (array): genres of interest 
        df_trn (dataframe): train set
        df_val (dataframe): validation set
        dtm (array): tfidf document-term matrix for train set 
        dtm_val (array): tfidf document-term matrix for validation set
    '''

    def __init__ (self, test_idx, genre):
        self.test_idx = test_idx
        self.genre = genre

        self.df_trn = None
        self.df_val = None
        self.dtm = None
        self.dtm_val = None
        
    def get_dtm(self, movies, train_idx0, min_df, max_df):
        ''' Get tfidf document-term matrix for train and validation set
        
        Args:
            movies (dataframe): pandas dataframe of pre-processed dataset
            train_idx0 (array): index for full train set, 10% of these will be used to train classifier
            min_df (int): parameter in CountVectorizer for minimum documents that keywords should appear in 
            max_df (int): parameter in CountVectorizer for maximum documents that keywords should appear in 
            
        Returns:
            None
        '''
        
        np.random.seed(1)
        train_idx = np.random.choice(train_idx0,int(0.1*movies.shape[0]),replace=False)

        self.df_trn = movies.iloc[train_idx,:]
        self.df_val = movies.iloc[self.test_idx,:]
        genre = movies.columns[1:]

        corpus = self.df_trn['overview']

        vectorizer = CountVectorizer(ngram_range=(1, 1), min_df =min_df, max_df = max_df)
        vectorizer.fit(corpus)
        tf_dtm = vectorizer.transform(corpus)

        transformer =  TfidfTransformer()
        self.dtm = transformer.fit_transform(tf_dtm)

        corpus_val = self.df_val['overview']
        tf_dtm_val = vectorizer.transform(corpus_val)
        transformer =  TfidfTransformer()
        self.dtm_val = transformer.fit_transform(tf_dtm_val)
              
    def classifier(self, classifier):
        ''' Get performane metrics of supervised benchmark using a particular classification model
        
        Args:
            classifier (sklearn classifier): classifer to be used as supervised benchmark
            
        Returns:
            yres_test (dataframe): Performance metrics for each genre 
        '''
    
        ypred_val_proba = np.zeros((self.dtm_val.shape[0],len(self.genre)))

        for i in range(len(self.genre)):
            classifier.fit(self.dtm,self.df_trn.iloc[:,i+1])
            ypred_val_proba[:,i] = classifier.predict_proba(self.dtm_val)[:,1]

        ypred_val_proba = pd.DataFrame(ypred_val_proba,columns = self.genre)
        ypred_val = np.round(ypred_val_proba)
        ytrue_val = self.df_val.iloc[:,1:]

        yres_test = pd.DataFrame(np.zeros((3,len(self.genre))), index = ['Precision','Recall', 'F1-score'],columns = self.genre)
        for i in range(len(self.genre)):

            yres_test.iloc[0,i] = round(precision_score(ytrue_val.iloc[:,i], ypred_val.iloc[:,i]),4) 
            yres_test.iloc[1,i] = round(recall_score(ytrue_val.iloc[:,i], ypred_val.iloc[:,i]),4)
            yres_test.iloc[2,i] = round(f1_score(ytrue_val.iloc[:,i], ypred_val.iloc[:,i]),4)

        return yres_test
    
