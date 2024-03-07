# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:33:58 2024

@author: rkpal
"""

# streamlit run d:\Projects\PubMedRead\gui.py

import os
import streamlit as st
import json
from Bio import Entrez
import time
from pubmedreadLLM import*



isDevt = 0

#-------------------------------------------------------------------
start_time = time.time()

if isDevt == 1:
   openai_api_key = os.environ["OPENAI_API_KEY"]
else:
   openai_api_key = st.secrets["OPENAI_API_KEY"]


st.set_page_config(
    page_title="Summarising Pubmed abstracts",
    layout="wide",  # Use "wide" layout for a wider page
)

st.title("Summarising PubMed Abstracts")

pane1, pane2 = st.columns(2)

def get_abstracts(query, retmax):
# Get abstracts from PubMed API (external)

    Entrez.email = "rk.palvannan@gmail.com"
    handle = Entrez.esearch(db='pubmed',term = query, retmax = retmax)
    studies = Entrez.read(handle)
    id_list = studies['IdList'] # list of strings of ids
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed', id = ids, retmode='xml')
    results = Entrez.read(handle)
    return(results)
    


def get_citation(papers, i):
# Extract only simple citations of article (except abstract); replace with empty str when not available in PubMed

    Title = papers['PubmedArticle'][i]['MedlineCitation']['Article']['ArticleTitle']
    Journal = papers['PubmedArticle'][i]['MedlineCitation']['Article']['Journal']['Title']
    
    try:
        Year = papers['PubmedArticle'][i]['MedlineCitation']['Article']['ArticleDate'][0]['Year']
    except:
        Year = ''
    try:
        authorLastName = papers['PubmedArticle'][i]['MedlineCitation']['Article']['AuthorList'][0]['LastName']
    except:
        authorLastName = ''
    try:
        authorForeName = papers['PubmedArticle'][i]['MedlineCitation']['Article']['AuthorList'][0]['ForeName']
    except:
        authorForeName = ''
    
    s = str(authorLastName + ', ' + authorForeName + '. ' + Title + ' ' + Journal + ' (' + Year + ')'  )
    return(s)


def summaryDownload(question, query, time, summary, abstracts, qa):
# File to download: all information i.e. summary, Q&A of each abstract, time log etc.

    s = f"Question: {question} \n\
Query: {query} \n\
Time taken min: {time} \n\n\
SUMMARY: \n\n\
{summary} \n\n"

    s = s + 'REFERENCES:\n\n'
    for i, abstract in enumerate(abstracts):
        s = s + str(i+1) + ') ' + abstract + '\n\n'
        s = s + 'Ans: ' + qa[i][0] + '\n\n\n'

    return(s)

#-----------------------------------------------------------------


with pane1:
    # Screen UI
        
    # Get user input for text string
    question = st.text_input("What is your question? e.g. Is universal screening for MRSA effective?")
    query = st.text_input("Enter PubMed query e.g. 'universal screening' [title] AND MRSA")
    
    
    # max num of abstracts to extracted
    radio_label_maxNumAbs = ['5', '20', '50']
    radio_maxNumAb = st.radio('Max num of abstracts - ', radio_label_maxNumAbs)
    retmax = int(radio_maxNumAb)
    
    
    isSubmit = st.button("Run query") 

# Display the entered text
if isSubmit:
    
    papers = get_abstracts(query, retmax)
    
    p = json.dumps(papers, indent = 4) # this is to viz the hierarchy (unused), p is a string
    
    # combine dict into simple list of abstracts (each abstract is a string)
    numAbstracts = len(papers['PubmedArticle'])
    titles = []
    abstracts = []
    for i in range(numAbstracts):

        print('Pubmed article - ', i+1)
        try:
            #
            txtElem = papers['PubmedArticle'][i]['MedlineCitation']['Article']['Abstract']['AbstractText']
            txt = ' '.join(txtElem)
            
            title = get_citation(papers, i) 
            titles.append(title)
            
            abstracts.append(title + '\n\n' + txt)

        except:
            print('Missing elements of abstract')
            
    numAbstracts = len(titles) # absracts with titles and abstract texts

    # Summary each article and then summarise the summaries (Use OpenAI LLM)
    x = call_llm(abstracts, question, query, openai_api_key, isDevt)

    summary = x[0]
    qa = x[1]

    stop_time = time.time()
    time_taken = str(round((stop_time - start_time)/60,2))


#   To download file    
    file_download = summaryDownload(question, query, time_taken, summary, abstracts, qa)
    
    with pane1:
        st.download_button('Download results', file_download, 'text/csv')
  
    with pane2:
    #   print to screen    
        st.write('Time taken (min): ', time_taken )
        st.write('Num of titles/abstracts: ', str(len(titles)), '/', str(len(abstracts)))
        st.write('SUMMARY: \n')
        st.write(summary)
            
        st.write("\n\n\nREFERENCES\n")
        for i in range(numAbstracts):
            st.write(str(i+1) + ') ' + titles[i] + '\n')
            s1 = qa[i][0]
            st.write(s1 + '\n\n')

    

