#Author: Omid Rohanian

#The program seeks to verify Zipf's law on a specific text corpus
#The language model is constructed from the script for the classic theatrical play 'Macbeth' 

import nltk, pylab, math
from scipy import stats

with open(r'./macbeth.txt', 'r') as file:
    text = file.read()
    file.close()

tokens = nltk.word_tokenize(text)
tokens = [token.lower() for token in tokens if len(token) > 1]
fdist = nltk.FreqDist(tokens)
words = fdist.most_common()

x = [math.log10(i[1]) for i in words]
y = [math.log10(i) for i in range(1, len(x))]
x.pop()


(m, b) = pylab.polyfit(x, y, 1)
yp = pylab.polyval([m, b], x)

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#r_value = -0.894028480118 so the square is 0.799286923262101


pylab.plot(x, yp)
pylab.scatter(x, y)
pylab.ylim([min(y), max(y)])
pylab.xlim([min(x), max(x)])
pylab.grid(True)
pylab.ylabel('Counts of words')
pylab.xlabel('Ranks of words')
pylab.figtext(.3, .5, "R^2 = 0.7992869")
##pylab.show()

##The dot plot is saved in a pdf file named "Zipf"
pylab.savefig(r'./Zipf.pdf')

