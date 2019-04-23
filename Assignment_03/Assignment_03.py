#Importing Libraries..
import nltk
nltk.download('treebank')
from nltk.corpus import treebank
from nltk import treetransforms
from nltk import induce_pcfg
from nltk.parse import pchart
from nltk import Nonterminal
from nltk import CFG
from nltk import tokenize
from nltk.parse import ViterbiParser
import pickle
import sys

#Chomsky Normal Form...
productions = []
for item in treebank.fileids()[:]:
    for tree in treebank.parsed_sents(item):
        # perform optional tree transformations, e.g.:
        tree.collapse_unary(collapsePOS = False)# Remove branches A-B-C into A-B+C
        tree.chomsky_normal_form(horzMarkov = 2)# Remove A->(B,C,D) into A->B,C+D->D
        productions += tree.productions()


lhs_prod = [p.lhs() for p in productions]
rhs_prod = [p.rhs() for p in productions]
set_prod = set(productions)

list_prod = list(set_prod)
len(list_prod)


#Making UNK tags for unsupervised words...
token_rule = []
for ele in list_prod:
    if ele.is_lexical():
        token_rule.append(ele)


set_token_rule = set(p.lhs() for p in token_rule)
list_token_rule = list(set_token_rule)
corr_list_token_rule = []
for word in list_token_rule:
    if str(word).isalpha():
        corr_list_token_rule.append(word)
        continue



a = []
for tok in corr_list_token_rule:
    lhs = 'UNK'
    rhs = [u'UNK']
    UNK_production = nltk.grammar.Production(lhs, rhs)   
    lhs2 = nltk.grammar.Nonterminal(str(tok))
    a.append(nltk.grammar.Production(lhs2, [lhs]))


token_rule.extend(a)
list_prod.extend(a)

S = Nonterminal('S')
grammar = induce_pcfg(S,list_prod)


#sent = sys.argv[1]
#print(sent)
#sent = "I am great!"

# Tokenize the sentence.
#tokens = sent.split()
tokens = sys.argv[1:]

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

UNK_str = trees[0].__str__()
answer= UNK_str

for i in change_words:
    answer = answer.replace("UNK",i,1)
print("\nTree is:\n\n")
print(answer)


