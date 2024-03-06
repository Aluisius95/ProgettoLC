import sys
import nltk
import codecs
import math
from nltk import ngrams

#liste per disambiguazione dei PoS
adjs = ["JJ", "JJR", "JJS"]
nouns = ["NN", "NNS", "NNP", "NNPS"]
advs = ["RB", "RBR", "RBS"]
symbol = [".",",",":",";","?","!","\"","'","*","(",")","[","]","{","}"]

#funzione per estrazione del PoS ed n-grammi
def estraiDati(sentences):
    corpusTot = []
    corpusPos = []
    listaPos = []
    for sentence in sentences:
        token = nltk.word_tokenize(sentence)
        tokenPos = nltk.pos_tag(token)
        corpusTot += token
        corpusPos += tokenPos
    #creo lista dei soli PoS
    for elem in corpusPos:
        listaPos.append(elem[1])
    bigrammi = list(ngrams(corpusPos, 2))
    return corpusTot, corpusPos, bigrammi, listaPos

#funzione di estrazione PoS generali o specifiche, divisa [monogrammi, bigrammi, trigrammi] e [aggettivi, avverbi] e calcolo frequenze
def estraiFreqPos(listPoS, n, m):
    #inizializzo un FreqDist generico che verrà elaborato in base all'elemento che viene spedito dalle funzioni principali
    DistrFreq = nltk.FreqDist(listPoS)
    #poiché 10 sono gli elementi richiesti di PoS divido con un if/elif in base alla parte dell'esercizio e poi analizzo in base alla lunghezza degli n-grammi
    if n == 10:
        getList = DistrFreq.most_common(n)
        i = 1
        if m == 1:
            for elem in getList:    
                print(i, " PoS singolo piu' frequente e' <", elem[0], "> con frequenza ", elem[1])
                i += 1
        elif m == 2:
            for elem in getList:
                print(i, " PoS bigramma piu' frequente e' <", elem[0][0], elem[0][1], "> con frequenza ", elem[1])
                i += 1
        elif m == 3:
            for elem in getList:
                print(i, " PoS trigramma piu' frequente e' <", elem[0][0], elem[0][1], elem[0][2], "> con frequenza ", elem[1])
                i += 1
    elif n == 20:
        #creo delle liste cumulatori per i token dei PoS richiesti
        adj = []
        adv = []
        getList=getList = DistrFreq.most_common(len(DistrFreq))
        #tramite il for filtro ed inserisco nei rispettivi array
        for elem in getList:
            if elem[0][1] in adjs:
                adj.append(elem)
            elif elem[0][1] in advs:
                adv.append(elem)
        adj = adj[0:n]
        adv = adv[0:n]
        print("I 20 aggettivi piu' frequenti sono:")
        for elem in adj:
            print(" - Token: ", elem[0][0], "\tFrequenza", elem[1])
        print("\nI 20 avverbi piu' frequenti sono:")
        for elem in adv:
            print(" - Token: ", elem[0][0], "\tFrequenza", elem[1])

#estrarre bigrammi Agg e Sost con freq tok > 3 e stampare FreqMAX, ProbCondMAX e LMIMax
def estraiBigAS(listPoS, bigrams, leng):
    listNoun = []
    listAdj = []
    listAdjNoun = []
    DistrFreq1 = nltk.FreqDist(listPoS)
    DistrFreq2 = nltk.FreqDist(bigrams)
    getList1 = DistrFreq1.most_common(len(DistrFreq1))
    getList2 = DistrFreq2.most_common(len(DistrFreq2))
    #filtro aggettivi, sostantivi e bigrammi <Agg, Sost>
    for elem in getList1:
            if elem[0][1] in adjs:
                listAdj.append(elem)
            if elem[0][1] in nouns:
                listNoun.append(elem)
    #se un bigramma ha una frequenza maggiore di tre, allora anche i token che lo compongono avranno Freq > 3
    for elem in getList2:
        if elem[0][0][1] in adjs and elem[0][1][1] in nouns and elem[1] > 3:
            listAdjNoun.append(elem)
    #stampo bigrammi in ordine di frequenza massima
    print("I 20 bigrammi con frequenza massima: ")
    for elem in listAdjNoun[0:20]:
        tok1 = elem[0][0][0]
        tok2 = elem[0][1][0]
        print(" - Bigramma: <", tok1, " ", tok2, "> \tFrequenza: ", elem[1])
    #creo dei dizionari per salvare i valori e, successivamente, ordinarli, listarli e selezionare i primi 20 per prob. Condizionata e LMI
    ProbBig = {}
    LMIMax = {}
    for elem in listAdjNoun:
        tok0 = elem[0][0][0]
        tok1 = elem[0][1][0]
        freq0 = 0
        freq1 = 0
        #estraggo per ogni bigramma, la relativa frequenza dei token che lo compongono
        for elemX in listAdj:
            if elem[0][0] == elemX[0]:
                freq0 = elemX[1]
        for elemX in listNoun:
            if elem[0][1] == elemX[0]:
                freq1 = elemX[1]
        #calcolo la probabilità condizionata (frequenza bigramma diviso la frequenza del primo elemento che lo compone)
        probCond = elem[1]/freq0
        #calcolo la LMI
        probMI = math.log((elem[1]*leng)/(freq0*freq1),2)
        lmi = probMI*elem[1]
        #ad ogni ciclo del for creo un nuovo elemento di dizionario dato dalla chiave composta dagli elementi del bigramma e valore pari a quella relativa del dizionario
        ProbBig[tok0 + ' ' + tok1] = probCond
        LMIMax[tok0 + ' ' + tok1] = lmi
    #ordino i due dizionari in ordine decrescente e, successivamente, stampo i primi 20 di ognuno
    ProbBigSort = list(sorted(ProbBig.items(), key = lambda x : x[1], reverse = True))
    LMIMaxSort = list(sorted(LMIMax.items(), key = lambda x : x[1], reverse = True))
    print("\nI 20 bigrammi con probabilita' condizionata massima in ordine:")
    for elem in ProbBigSort[0:20]:
        print(" - Bigramma: <", elem[0], ">\tProbabilita' condizionata: ", format(elem[1],'f'))
    print("\nI 20 bigrammi con Local Mutual Information massima in ordine:")
    for elem in LMIMaxSort[0:20]:
        print(" - Bigramma: <", elem[0], ">\tLMI: ", format(elem[1],'f'))

#funzione per calcolo freq media dei token per frase e calcolo di markov 2
def freqMediaMarkov2(sent, corpus):
    #estraggo lista dei bigrammi e trigrammi con frequenze assolute annesse
    bigrams = list(ngrams(corpus, 2))
    trigrams = list(ngrams(corpus, 3))
    freqBigrams = nltk.FreqDist(bigrams)
    getListBigrams = freqBigrams.most_common(len(freqBigrams))
    freqTrigrams = nltk.FreqDist(trigrams)
    getListTrigrams = freqTrigrams.most_common(len(freqTrigrams))

    #genero lista di hapax per filtrare successivamente le frasi che ne contengono (sono richieste frasi con freq>=2 per ogni token che le compongono)
    hapax = []
    DistrFreq = nltk.FreqDist(corpus)
    getList = DistrFreq.most_common(len(DistrFreq))
    for elem in getList:
        if elem[1] == 1:
            hapax.append(elem[0])

    #filtro le frasi per la lunghezza e l'assenza di hapax al loro interno
    sentFiltered = []
    for elem in sent:
        sentTokenized = []
        token = nltk.word_tokenize(elem)
        sentTokenized += token
        leng = len(sentTokenized)
        if leng >= 6 and leng < 25:
            for word in sentTokenized:
                if not word in hapax:
                    sentFiltered.append(sentTokenized)
                    #inserisco un break per evitare che il ciclo vada avanti
                    break

    #calcolo freq media massima e minima dei token nelle frasi, inizializzando dei valori di massimo e minimo ai valori opposti e delle stringhe per contenere le frasi
    max = 0
    min = math.inf
    sentMax = ''
    sentMin = ''
    #accedo alle frasi filtrate e inizio a verificare le frequenze medie richieste
    for sentence in sentFiltered:
        sentString = ''
        freqTot = 0
        for word in sentence:
            for wordC in getList:
                if word == wordC[0]:
                    freqTot += wordC[1]
            sentString += ' ' + word
        freq = freqTot/len(sentence)
        #alla fine di ogni ciclo, confronto i valori trovati con le varibili esterne al for per l'assegnamento dei valori max/min e relative frasi
        if  freq > max:
            max = freq
            sentMax = sentString
        if freq < min:
            min = freq
            sentMin = sentString
    print("La frase <", sentMax, "> ha la frequenza media di token maggiore pari a:", format(max,'f'))
    print("\nMentre la frase <", sentMin, "> ha la frequenza media di token minore pari a: ", format(min,'f'))

    #calcolo la probabilità maggiore con markov2, inizializzo una var contenitore per la probMax e una per la stringa relativa
    probMarkovMax = 0.0
    sentProbMax = ''
    #accedo alla lista di frasi filtrate e per ognuna di esse calcolo i trigrammi e bigrammi temporaneamente per il calcolo di Markov 2
    for elem in sentFiltered:
        sentProbtemp = ''
        trigramma = list(ngrams(elem,3))
        bigramma = list(ngrams(elem, 2))
        probMarkov = 1.0
        freqTrig = 1.0
        freqBig = 1.0
        freqPrimo = 0.0
        #temporaneamente salvo la stringa che sto analizzando
        for words in elem:
            sentProbtemp += ' ' + words
        #per la formula di Markov 2, inizio a trovare la probabilità del primo elemento di frase che inserisco nella variabile contatore di prob
        for word in getList:
            if word[0] == sentence[0]:
                freqPrimo = word[1]
                probMarkov = word[1]/len(corpus)
        #successivamente calcolo la probabilità del bigramma cercando, sempre nel corpus di riferimento, le frequenze del primo bigramma e lo divido per la frequenza del primo elemento
        #P(b|a)=(F(<a,b>)/(F(a)))
        for big in getListBigrams:
            if big[0] == bigramma[0]:
                probMarkov *= (big[1]/freqPrimo)
        #infine inizio a cercare tutte le frequenze dei trigrammi e, poiché il resto della formula di Markov 2 è la Produttoria di F(<a,b,c>)/F(<a,b>), moltiplico tutte le frequenze dei trigrammi richiesti
        for trig in trigramma:
            for freqT in getListTrigrams:
                if trig == freqT[0]:
                    freqTrig *= freqT[1]
        #moltiplico tutte le frequenze dei bigrammi richiesti, eccetto l'ultimo (se F<N, N, .>, a me serve F<N, N>)
        for bigr in bigramma:
            for freqB in getListBigrams:
                if bigr == bigramma[len(bigramma)-1]:
                    break
                #infatti verifico quando arrivo all'ultimo elemento della lista dei bigrammi di frase e interrompo il ciclo
                elif bigr == freqB[0]:
                    freqBig *= freqB[1]
        #Moltiplico la probabilità inizialmente trovata per la divisione del prodotto delle frequeneze e confronto con le variabili esterne
        probMarkov *= (freqTrig/freqBig)
        if probMarkov > probMarkovMax:
            probMarkovMax = probMarkov
            sentProbMax = sentProbtemp
    print("\nLa frase <", sentProbMax, "> ha probabilita' maggiore con Markov di ordine 2 pari a: ", format(probMarkovMax, 'f'))

#funzione che trova e restituisce i 15 nomi propri di persona più frequenti
def propernounNE(pos):
    #inizializzo l'albero delle Named Entity tramite il PoS del testo relativo e una lista per contenere gli elementi finali richiesti
    treeNE = nltk.ne_chunk(pos)
    properN = []
    for nodo in treeNE:
        #verifico, per ogni nodo, che esista un sotto-albero e che sia di tipo PERSON e non GPE o altro
        if hasattr(nodo, 'label'):
            if nodo.label() == 'PERSON':
                #inizializzo una variabile per la memorizzazione della stringa e con il ciclo le estraggo dal nodo, per poi appenderle come elementi di lista nell'array esterno
                NE = ''
                for partNE in nodo.leaves():
                    NE += partNE[0] + ' '
                properN.append(NE)
    #calcolo le frequenze dei nomi trovati e infine li stampo
    DistrFreq = nltk.FreqDist(properN)
    getList = DistrFreq.most_common(15)
    print("I 15 nomi propri di persona piu' frequenti nel testo sono:")
    for elem in getList:
        print(" - Nome: ", elem[0], "\tFrequenza: ", elem[1])

#raggruppo tutti i richiami alle funzioni di analisi in un'unica funzione per pulire il main
def startProg2(corpus, PoS, bigrams, listOnlyPoS, len, sent1):
    print("Primo punto\n")
    #estrazione e ordinamento decrescente delle 10 PoS più frequenti
    estraiFreqPos(listOnlyPoS, 10, 1)
    print()
    #estrazione e ordinamento decrescente dei 10 bigrammi PoS più frequenti
    estraiFreqPos(list(ngrams(listOnlyPoS, 2)), 10, 2)
    print()
    #estrazione e ordinamento decrescente dei 10 trigrammi PoS più frequenti
    estraiFreqPos(list(ngrams(listOnlyPoS, 3)), 10, 3)
    print()
    #estrazione e ordinamento decrescente dei 20 Aggettivi e 20 Avverbi più frequenti
    estraiFreqPos(PoS, 20, '')
    print()
    print("\n/--------------------------------------------------/\n")
    print("Secondo punto\n")
    #controllo solo i bigrammi <Agg, Sost> 
    estraiBigAS(PoS, bigrams, len)
    print()
    print("\n/--------------------------------------------------/\n")
    print("Terzo punto\n")
    #estrazione frequenza media in una frase 5 < len < 25 e calcolo di markov2
    freqMediaMarkov2(sent1, corpus)
    print()
    print("\n/--------------------------------------------------/\n")
    print("Quarto punto\n")
    #estrarre i 15 Nomi Propri di persona più frequenti
    propernounNE(PoS)

#definisco funzione main con estrazione delle parti utili allo svolgimento del secondo programma
def main(file1, file2):
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    input1 = codecs.open(file1, mode="r", encoding="utf-8")
    raws1 = input1.read()
    sent1 = sentence_tokenizer.tokenize(raws1)
    input2 = codecs.open(file2, mode="r", encoding="utf-8")
    raws2 = input2.read()
    sent2 = sentence_tokenizer.tokenize(raws2)
    print("\t\tProgramma numero 2!\n\n")
    #inizio dell'analisi dividendo il corpus intero in token, PoS, bigrammi e trigrammi
    corpus1, PoS1, bigrams1, listOnlyPoS1 = estraiDati(sent1)

    #chiamo funzione che avvierà tutte le analisi sui testi divisi per punti
    print("Testo in input numero 1 con |C| =", len(corpus1),"\n\n")
    startProg2(corpus1, PoS1, bigrams1, listOnlyPoS1, len(corpus1), sent1)
    print()
    print("\n/------------------------------------------------------------/\n")
    corpus2, PoS2, bigrams2, listOnlyPoS2 = estraiDati(sent2)
    print("Testo in input numero 2 con |C| =", len(corpus2),"\n\n")
    startProg2(corpus2, PoS2, bigrams2, listOnlyPoS2, len(corpus2), sent2)

main(sys.argv[1],sys.argv[2])