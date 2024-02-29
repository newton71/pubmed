# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:49:47 2023
@author: rkpal
extract information from a list of abstracts
"""


def call_llm(abstracts, question, query, openai_api_key, isDevt):
    
    # to convert text to numbers
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS # Facebook AI Similarity Search (dot product of 2 vectors)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key = openai_api_key) # download embeddings; every token is a vector


    # to split large text into manageable sections (chunks)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Q&A
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI
    
    # Chat (for summary)
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import(AIMessage, HumanMessage,SystemMessage)

    
#-----------------------------------------------------------
# 1. Q&A of each abstract (only 1 Q in this example)    

#    fname = 'D:/Projects/PubMedRead/' + topic + '_abstracts.txt'
#    with open(fname, 'r', encoding='utf-8') as f:
#        x = f.read()

    articles = abstracts 
   
#    articles = x.split('\n\n\n')
#    articles.pop() # debug later, redundant empty tail entry, to be removed
                           
    queries = []
    queries.append(question)
    
    num_queries = len(queries)
    num_articles = len(articles)
    
    results = [] # 2d list of [answers to each query X articles]
    title = [] # extracted and stored separately for printing
    for j in range(num_articles):
       
 #       print(j+1)

        title.append(articles[j].split('\n\n'))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 200,
            separators = ["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        
        texts = text_splitter.split_text(articles[j])
       
        # the chunks of text are converted to vectors and stored
        document_search = FAISS.from_texts(texts, embeddings)
        
  
        # Q&A chain
#        m = 'gpt-3.5-turbo-instruct'
        chain = load_qa_chain(OpenAI(temperature = 0.3, model = 'gpt-3.5-turbo-instruct', openai_api_key = openai_api_key), chain_type="stuff")
        
        # to store the chunk that is similar to the query
        docs = num_queries*[None] 
     
        row = [] # list of proper answer to each query  
        for n in range(num_queries):
             # matches the closest chunk with the query and stores in doc[n]
             docs[n] = document_search.similarity_search(queries[n])
             # calls the LLM model (QA chain and finds a suitable answer using the matched chunk)
             row.append(chain.run(input_documents=docs[n], question=queries[n]))
        results.append(row)
    # end of loop
    
        
# -----------------------------------------------

# 2. summarising the observations
    observations = []
    for j in range(len(results)):
        observations.append((results[j][0])) # index 0 means only 1 Q&A on each abstract
    
    # Concatenate the observations into a single string
    text_to_summarize = '\n'.join(observations)
#    print(text_to_summarize)
    



# Make the API call to OpenAI GPT-3 for summarization
    chat_messages = [
        SystemMessage(content = "You are an expert in the medical domain and are able to summarise concisely \
                      from multiple published abstracts into a single coherent summary."),
        HumanMessage( content = f"Please provide a single concise summary\
                     of the multiple answers from different abstracts in a single paragraph :\n TEXT = {text_to_summarize}. \
                        Keep summary within 1000 words")
    ]

                       
                     
#    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo')
    llm = ChatOpenAI(model_name = 'gpt-4', temperature = 0.3, openai_api_key = openai_api_key)

    summary = llm(chat_messages).content
   
    return(summary, results)


