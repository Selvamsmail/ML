
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import re

# @st.cache_data
# def collect():
#   nltk.download('stopwords')
#   stop_words = stopwords.words('english')
#   nltk.download('omw-1.4')
#   nltk.download('punkt')
#   nltk.download('wordnet')
#   nltk.download('averaged_perceptron_tagger')
#   return

# @st.cache_resource
# def getdata():

#   gdown.download(id = '1qFViB4VmxIrC_atqunNzq9pqUaz0YMk0')
#   gdown.download(id = '1YGzqZhxig_5szsufoDXYJdRiAvxKtz1F')
  
#   with open('/app/profinity_classification/tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#   model = load_model('/app/profinity_classification/model.h5')
  
#   return tokenizer, model


# def badword_prediction(input_data):
     
#     df = pd.DataFrame(columns = ['text'])
#     new_row= {'text': input_data}
#     new_row = pd.DataFrame(new_row, index=[0])
#     df = pd.concat([df, new_row], ignore_index=True)
    
#     df['text'] = [str(i).lower() for i in df['text']]
#     df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
    
#     df['text'] = df['text'].apply(lambda x: lemmatize_sentence(x))
    
#     texts = df['text'].to_list()

#     tokenizer.fit_on_texts(texts)
#     sequences = tokenizer.texts_to_sequences(texts)
    
#     max_sequence_length = 15 # based on pad_seq length sugestion
#     padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length)

#     predictions = model.predict(padded_sequence)[0][0]
#     threshold = 0.7
#     predictions = np.exp(predictions) / (np.exp(predictions) + 1)
#     confidence = predictions
#     predictions = ('it is a Bad word with a confidence of '+str(predictions)+' %' if predictions >= threshold else 'it is not a Bad word' )
       
#     return(predictions)

def main():
    
    st.write('Check the latest salary trends')
    st.title('Salary Insights Tool')
    
    
    model_filepath = 'GB.pkl'
    with open(model_filepath, 'rb') as file:
        model = pickle.load(file)
    
    minexp = st.text_input('Enter Minimum Experience:')
    maxexp = st.text_input('Enter Maximum Experience:')
    role = st.selectbox(
        "Select Role",
        options=['customer service/support', 'it infrastructure services',
       'software developer', 'data engineer',
       'devops engineer/consultant', 'recruiter',
       'bfsi trading and investments',
       'automation engineer/developer/architect', 'medical biller/coder',
       'field sales', 'tech support', 'business development',
       'human resources', 'technical/functional consultant',
       'digital/seo marketing', 'it security', 'design engineer',
       'relationship manager', 'graphic designer', 'technical architect',
       'tech lead', 'mobile/app developer', 'full stack developer',
       'back end developer', 'front end developer'],
    )
    
    # button
    if st.button('show salaries'):
        x_train = pd.read_csv('x_train.csv')
        for i in x_train.columns:
            x_train[i] = 0
        role = 'role_'+ role
        x_train[role] = 1
        x_train['experience_minimum'] = float(minexp)
        x_train['experience_maximum'] = float(maxexp)
        result = model.predict(x_train)
        avg = round(result[0,0] * 100000,1)
        min = round(result[0,1],1)
        max = round(result[0,2],1)
        
        data = {
            'Minimum_salary':str(min), 
            'Maximum_salary':str(max),
            'Average_salary':str(avg) 
        }
        st.write('your salaries would be: ',pd.DataFrame(data,index=[0]))
    st.write('this model was trained with an accuracy of 75%')
    st.link_button('My Github','https://github.com/Selvamsmail')
if __name__ == '__main__':
    main()