import pandas as pd
import spacy
import gensim.downloader as api
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report
wv=api.load('word2vec-google-news-300')
nlp=spacy.load("en_core_web_lg")

df=pd.read_json('./news.json',lines=True)

df['category_num']=df.category.map({
'POLITICS'  :0,        
'ENTERTAINMENT'  :1,   
'HEALTHY LIVING' :2,    
'QUEER VOICES'  :3,     
'BUSINESS'   :4,        
'SPORTS':5,              
'COMEDY'   :6,          
'PARENTS'   :7,         
'BLACK VOICES'   :8,    
'THE WORLDPOST'  :9,    
'WOMEN '        :10,     
'CRIME'        :11,      
'MEDIA '     :12,        
'WEIRD NEWS'   :13  ,    
'GREEN'   :14,           
'IMPACT'  :15,           
'WORLDPOST ' :16,        
'RELIGION '  :17,        
'STYLE'    :18,          
'WORLD NEWS'  :19,       
'TRAVEL'     :20,        
'TASTE'    :21,          
'ARTS'     :22,          
'FIFTY '      :23,       
'GOOD NEWS'  :24,        
'SCIENCE'     :25,       
'ARTS & CULTURE'   :26,  
'TECH'            :27,   
'COLLEGE'         :28,   
"LATINO VOICES"      :29,
"EDUCATION "      :30   
})

def preprocess_and_vector(text):
    doc=nlp(text)
    preprocessed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return wv.get_mean_vector([preprocessed_text])

df['vector']=df.headline.apply(lambda x: preprocess_and_vector(x))

X_train, X_test, y_train, y_test = train_test_split(df['vector'].values,
                                                     df['category_num'],
                                                       test_size=0.2,
                                                         random_state=42,
                                                         stratify=df['category_num'])
X_train_2d=np.stack(X_train)
X_test_2d=np.stack(X_test)
lr=LogisticRegression()
lr.fit(X_train_2d,y_train)
y_pred=lr.predict(X_test_2d)

print(classification_report(y_test, y_pred))

