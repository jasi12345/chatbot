from flask import Flask, render_template, request, jsonify
import numpy as np
import nltk
import string
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

f=open('chat.txt','r',errors='ignore')
raw_doc=f.read()


raw_doc=raw_doc.lower().strip()


nltk.download('punkt')
nltk.download('wordnet')
sent_tokens=nltk.sent_tokenize(raw_doc)
word_tokens=nltk.word_tokenize(raw_doc)

lemmer=nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict =dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREET_INPUTS=("hello","hi","greetings","sup","what's up","hey" )
GREET_RESPONSES=["Hi","hey","*nods*","hi there" , "hello"]
def greet(sentence):

  for word in sentence.split():
    if word.lower()in GREET_INPUTS:
      return random.choice(GREET_RESPONSES)

def get_subtopic_response(user_input):
    user_input = user_input.lower()

    if  "difference" in user_input and "machine learning" in user_input and "deep learning" in user_input and "generative ai" in user_input:
        start_index = raw_doc.find("difference between machine learning, deep learning, and generative ai:") + len("difference between machine learning, deep learning, and generative ai:")
        end_index = raw_doc.find("deep learning models:")

        if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
        else:
            return "Sorry, I couldn't find the definition of data science."
        
    elif "machine learning" in user_input and "deep learning" in user_input:
        start_index = raw_doc.find("benefits of deep learning over machine learning:") + len("benefits of deep learning over machine learning:")
        end_index = raw_doc.find("challenges of deep learning")

        if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
        else:
            return "Sorry, I couldn't find the definition of data science." 

    elif "data science" in user_input:
        
        
        if "process cycle" in user_input:
          
            start_index = raw_doc.find("data science process cycle:") + len("data science process cycle:")
            end_index = raw_doc.find("data science grown:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the data science process cycle."

        elif "growth" in user_input or "evolution" in user_input or "development" in user_input:
         
            start_index = raw_doc.find("data science grown:") + len("data science grown:")
            end_index = raw_doc.find("data scientists in data science:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the roles in data science."
            
        elif "careers" in user_input or "jobs" in user_input:
            start_index = raw_doc.find("careers and job titles in data science:") + len("careers and job titles in data science:")
            end_index = raw_doc.find("data science professionals tasks:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the data scientists."   

        elif "tasks" in user_input:
            start_index = raw_doc.find("data science professionals tasks:") + len("data science professionals tasks:")
            end_index = raw_doc.find("the future of data science:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the professionals tasks." 

        elif "future" in user_input:
            start_index = raw_doc.find("the future of data science:") + len("the future of data science:")
            end_index = raw_doc.find("uses of data science:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the future of data science." 

        elif "uses" in user_input:
            start_index = raw_doc.find("uses of data science:") + len("uses of data science:")
            end_index = raw_doc.find("benefits of data science:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the uses of data science." 
        
        elif "benefits" in user_input or "advantages" in user_input:
            start_index = raw_doc.find("benefits of data science:") + len("benefits of data science:")
            end_index = raw_doc.find("data science process:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the benefits of data science." 

        elif "data science process" in user_input:
            start_index = raw_doc.find("data science process:") + len("data science process:")
            end_index = raw_doc.find("data science techniques:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the data science process." 

        elif "techniques" in user_input:
            start_index = raw_doc.find("data science techniques:") + len("data science techniques:")
            end_index = raw_doc.find("the basic principle behind data science:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the data science techniques."   
        
        elif "principle" in user_input:
            start_index = raw_doc.find("the basic principle behind data science:") + len("the basic principle behind data science:")
            end_index = raw_doc.find("data science practitioners work with complex technologies:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the basic principle behind data science techniques."   
            
        elif "complex technologies" in user_input:
            start_index = raw_doc.find("data science practitioners work with complex technologies:") + len("data science practitioners work with complex technologies:")
            end_index = raw_doc.find("difference between data science and data analytics:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the data science practitioners work with complex technologies."   
            
        elif "data analytics" in user_input:
            start_index = raw_doc.find("difference between data science and data analytics:") + len("difference between data science and data analytics:")
            end_index = raw_doc.find("difference between data science and business analytics:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the difference between data science and data analytics."   

        elif "business analytics" in user_input:
            start_index = raw_doc.find("difference between data science and business analytics:") + len("difference between data science and business analytics:")
            end_index = raw_doc.find("difference between data science and data engineering:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the difference between data science and business analytics."   

        elif "data engineering" in user_input:
            start_index = raw_doc.find("difference between data science and data engineering:") + len("difference between data science and data engineering:")
            end_index = raw_doc.find("difference between data science and machine learning:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the difference between data science and data engineering."   
 
        elif "data engineering" in user_input:
            start_index = raw_doc.find("difference between data science and data engineering:") + len("difference between data science and data engineering:")
            end_index = raw_doc.find("difference between data science and machine learning:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the difference between data science and data engineering."   
 
        elif "machine learning" in user_input:
            start_index = raw_doc.find("difference between data science and machine learning:") + len("difference between data science and machine learning:")
            end_index = raw_doc.find("difference between data science and statistics:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the difference between data science and machine learning." 
    
        elif "statistics" in user_input:
            start_index = raw_doc.find("difference between data science and statistics:") + len("difference between data science and statistics:")
            end_index = raw_doc.find("different data science tools:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the difference between data science and statistics."
            
        elif "tools" in user_input:
            start_index = raw_doc.find("different data science tools:") + len("different data science tools:")
            end_index = raw_doc.find("machine learning:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the different data science tools."
        
        elif "definition" in user_input or "is data science" in user_input:
         
            start_index = raw_doc.find("data science definition:") + len("data science definition:")
            end_index = raw_doc.find("data science process cycle:")

            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the definition of data science."

    elif "data scientists" in user_input or "data scientist" in user_input:
            if "skils" in user_input or "qualifications" in user_input:
                start_index = raw_doc.find("data scientists skills and qualifications:") + len("data scientists skills and qualifications:")
                end_index = raw_doc.find("data scientists challenges:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the data scientists skills and qualifications."        
            
            elif "challenges" in user_input:
                start_index = raw_doc.find("data scientists challenges:") + len("data scientists challenges:")
                end_index = raw_doc.find("careers and job titles in data science:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the data scientists challenges."        
                
            elif "is" in user_input:
                start_index = raw_doc.find("data scientists:") + len("data scientists:")
                end_index = raw_doc.find("data scientists skills and qualifications:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the data scientists."

        
    elif "machine learning" in user_input:

        
        if "types" in user_input:
            start_index = raw_doc.find("different types of machine learning:") + len("different types of machine learning:")
            end_index = raw_doc.find("supervised machine learning structure:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the different types of machine learning."

        elif ("choose" in user_input or "build" in user_input) and "model" in user_input :
                    start_index = raw_doc.find("choose and build the right machine learning model:") + len("choose and build the right machine learning model:")
                    end_index = raw_doc.find("applications of machine learning:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the choose and build the right machine learning model."

        elif "applications" in user_input or "application" in user_input:
                    start_index = raw_doc.find("applications of machine learning:") + len("applications of machine learning:")
                    end_index = raw_doc.find("examples of machine learning:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the choose and build the right machine learning model."

        elif "examples" in user_input:
                    start_index = raw_doc.find("examples of machine learning:") + len("examples of machine learning:")
                    end_index = raw_doc.find("advantages and disadvantages of machine learning:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the examples of machine learning."
    
        elif "advantages and disadvantages" in user_input:
                    start_index = raw_doc.find("advantages and disadvantages of machine learning:") + len("advantages and disadvantages of machine learning:")
                    end_index = raw_doc.find("importance of human interpretable machine learning:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the advantages and disadvantages of machine learning."
                
        elif "transparency requirements" in user_input:
                    start_index = raw_doc.find("transparency requirements can dictate machine learning model choice:") + len("transparency requirements can dictate machine learning model choice:")
                    end_index = raw_doc.find("machine learning teams, roles and workflows:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the transparency requirements can dictate machine learning model choice." 
            
        elif "teams, roles and workflows" in user_input:
                    start_index = raw_doc.find("machine learning teams, roles and workflows:") + len("machine learning teams, roles and workflows:")
                    end_index = raw_doc.find("machine learning tools and platforms:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the machine learning teams, roles and workflows."
             
        elif "tools and platforms" in user_input:
                    start_index = raw_doc.find("machine learning tools and platforms:") + len("machine learning tools and platforms:")
                    end_index = raw_doc.find("frameworks and libraries:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the machine learning tools and platforms." 

        elif "frameworks" in user_input or "libraries" in user_input:
                    start_index = raw_doc.find("frameworks and libraries:") + len("frameworks and libraries:")
                    end_index = raw_doc.find("programming languages:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the transparency requirements can dictate machine learning model choice." 

        elif "programming languages" in user_input:
                    start_index = raw_doc.find("programming languages:") + len("programming languages:")
                    end_index = raw_doc.find("future of machine learning:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the programming languages."     
        
        elif "future" in user_input:
                    start_index = raw_doc.find("future of machine learning:") + len("future of machine learning:")
                    end_index = raw_doc.find("deep learning in ai:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the future of machine learning."    
                
        elif "importance of human interpretable" in user_input:
                    start_index = raw_doc.find("importance of human interpretable machine learning:") + len("importance of human interpretable machine learning:")
                    end_index = raw_doc.find("interpretable vs explainable ai:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the importance of human interpretable machine learning."
            
        elif "about" in user_input or "definiation" in user_input or "is" in user_input:
                    start_index = raw_doc.find("machine learning overview:") + len("machine learning overview:")
                    end_index = raw_doc.find("different types of machine learning:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the data science techniques."
    elif "supervised" in user_input:
            if "works" in user_input or "structure" in user_input:

                start_index = raw_doc.find("supervised learning structure:") + len("supervised learning structure:")
                end_index = raw_doc.find("supervised learning algorithms tasks:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the supervised machine learning structure."

            elif "tasks" in user_input:

                start_index = raw_doc.find("supervised learning algorithms tasks:") + len("supervised learning algorithms tasks:")
                end_index = raw_doc.find("semi supervised learning workflow:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the Supervised learning algorithms tasks."

    elif "semi" in user_input and "supervised" in user_input:
                if "workflow" in user_input:
                    start_index = raw_doc.find("semi supervised learning workflow:") + len("semi supervised learning workflow:")
                    end_index = raw_doc.find("semi supervised learning areas:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the semi supervised learning workflow."
                elif "areas" in user_input:
                    start_index = raw_doc.find("semi supervised learning areas:") + len("semi supervised learning areas:")
                    end_index = raw_doc.find("choose and build the right machine learning model:")
                    if start_index != -1 and end_index != -1:
                        return raw_doc[start_index:end_index].strip()
                    else:
                        return "Sorry, I couldn't find the semi supervised learning areas."
                    
    elif "interpretable vs explainable ai" in user_input or "different between interpretable explainable ai" in user_input:
            start_index = raw_doc.find("interpretable vs explainable ai:") + len("interpretable vs explainable ai:")
            end_index = raw_doc.find("transparency requirements can dictate machine learning model choice:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the interpretable vs explainable ai."


    elif "deep " in user_input and "learning" in user_input:
                
        if "generative" in user_input:
                start_index = raw_doc.find("deep generative learning:") + len("deep generative learning:")
                end_index = raw_doc.find("deep learning importantance:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the deep generative learning."
        
        elif "importance" in user_input:
                start_index = raw_doc.find("deep learning importance:") + len("deep learning importance:")
                end_index = raw_doc.find("deep learning use cases:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the deep learning importance."
        
        elif "application" in user_input:
                start_index = raw_doc.find("deep learning use cases:") + len("deep learning usecases:")
                end_index = raw_doc.find("computer vision:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the deep learning use cases."

        elif "workflow" in user_input:
                start_index = raw_doc.find("deep learning workflow:") + len("deep learning workflow:")
                end_index = raw_doc.find("the components of a deep neural network:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the deep learning workflow."

       
               
        elif "models" in user_input and "types" in user_input:
                start_index = raw_doc.find("types of deep learning models:") + len("types of deep learning models:")
                end_index = raw_doc.find("deep learning models uses:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the types of deep learning models."
        
        elif "models" in user_input and "uses" in user_input:
                start_index = raw_doc.find("deep learning models uses:") + len("deep learning models uses:")
                end_index = raw_doc.find("deep learning models work process:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the types of deep learning models."
        
        elif "models" in user_input and "process" in user_input:
            start_index = raw_doc.find("deep learning models work process:") + len("deep learning models work process:")
            end_index = raw_doc.find("deep learning models pros and cons:")
            if start_index != -1 and end_index != -1:
                return raw_doc[start_index:end_index].strip()
            else:
                return "Sorry, I couldn't find the deep learning models work process."
                
        elif "models" in user_input and "advantages and disadvantages" in user_input:
                start_index = raw_doc.find("deep learning models pros and cons:") + len("deep learning models pros and cons:")  
                end_index = raw_doc.find("neural network:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the pros and cons of using deep learning models."
        
        elif "models" in user_input and ("definition" in user_input or ("is" in user_input and "deep learning" in user_input)):

                start_index = raw_doc.find("deep learning models:") + len("deep learning models:")
                end_index = raw_doc.find("types of deep learning models:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the deep learning models."
        
        elif "definition" in user_input or "is" in user_input and "deep learning" in user_input and "models" not in user_input:
                start_index = raw_doc.find("deep learning in ai:") + len("deep learning in ai:")
                end_index = raw_doc.find("deep generative learning:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the deep learning in ai."
                
    elif "components" in user_input and "neural" in user_input:
                start_index = raw_doc.find("the components of a deep neural network:") + len("the components of a deep neural network:")
                end_index = raw_doc.find("difference between machine learning, deep learning, and generative ai:")
                if start_index != -1 and end_index != -1:
                    return raw_doc[start_index:end_index].strip()
                else:
                    return "Sorry, I couldn't find the components of a deep neural network."

def response(user_response, top_n=3):
    robo1_response = ''  
    
    subtopic_response = get_subtopic_response(user_response)
    if subtopic_response != "Sorry, I couldn't find information on that topic.":
        return subtopic_response
    
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][:-top_n-1:-1]  
    
    flat = vals.flatten()
    flat.sort()

    if flat[-2] == 0:  
        robo1_response = "I am sorry! I don't understand you."
    else:
        if flat[-2] > 0.5:  
            for idx in idx[1:]:  
                robo1_response += sent_tokens[idx] + " "
        else:
            robo1_response = sent_tokens[idx[1]]  
    
    sent_tokens.pop()
    return robo1_response.strip()  




@app.route("/")
def home():
    return render_template("new.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    if user_input.lower() in ["bye", "exit", "quit"]:
        return jsonify("Goodbye! Take care.")
    
    bot_response = greet(user_input)
    if bot_response is None:
        bot_response = response(user_input)
    
    return jsonify(f" {bot_response}")

if __name__ == "__main__":
    app.run(debug=True,port=5001)







