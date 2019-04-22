import sys, time
import pickle
pickle_in = open("grammar_with_prob","rb")
grammar = pickle.load(pickle_in)
from nltk.parse import ViterbiParser

sent = sys.argv[1]
print(sent)
#sent = "I am great!"

# Tokenize the sentence.
tokens = sent.split()

# Define a list of parsers.  We'll use all parsers.
parser = ViterbiParser(grammar)

print('Coverage of input words by a grammar:\n')
change_words = []
for i,ele in enumerate(tokens):
    try:
        grammar.check_coverage([ele])
    except:
        print("%s is not covered by the grammar. Replacing it with 'UNK'\n" % ele)
        change_words.append(tokens[i])
        tokens[i] = 'UNK'
        
    


trees = parser.parse_all(tokens)

for tree in trees:
    pass

UNK_str = tree.__str__()
answer= UNK_str

for i in change_words:
    answer = answer.replace("UNK",i,1)
print("\nTree is:\n\n")
print(answer)




