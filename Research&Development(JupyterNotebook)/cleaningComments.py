def number_to_text(data):
    temp_str = data.split()
    string = []
    for i in temp_str:

    # if the word is digit, converted to
    # word else the sequence continues

        if i.isdigit():
            import inflect
            temp = inflect.engine().number_to_words(i)
            string.append(temp)
        else:
            string.append(i)
    outputStr = " ".join(string)
    return outputStr

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')
lemma = WordNetLemmatizer()
def lemmatiz_text(data):
    from nltk import word_tokenize 
    tokens = word_tokenize(data)
    lemma_tokens = [lemma.lemmatize(word, pos='v') for word in tokens if word not in (stop_words)]
    return " ".join(lemma_tokens)


def cleantext(cleandata):
    import re
    import contractions
       
    cleandata = re.sub(r'[^\w\s]', " ", cleandata) # Remove punctuations
    
    cleandata = re.sub(r"https?:\/\/\S+", ",", cleandata) # Remove The Hyper Link
    
    cleandata = contractions.fix(cleandata) # remove contractions 
    
    cleandata = number_to_text(cleandata) # convert numbers to text   
    
    cleandata = cleandata.lower() # convert to lower case
        
    cleandata = lemmatiz_text(cleandata) # lemmatization
    
    return cleandata
