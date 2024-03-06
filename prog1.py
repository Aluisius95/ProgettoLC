import sys
import nltk
import codecs

#liste per disambiguazione PoS e raggruppammenti di insiemi
symbol = [".",",",":",";","?","!","\"","'","*","(",")","[","]","{","}"]
piene = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
funzionali = ["CC", "DT", "IN", "POS", "PRP", "PRP$", "TO", "WDT", "WP", "WP$", "WRB"]

#funzione per l'estrazione di corpus completo e relativa lunghezza, vocabolario e PoS Tagging
def estraiDati(sentences):
    corpusTot = []
    corpusF = []
    posList = []
    for sentence in sentences:
        token = nltk.word_tokenize(sentence)
        tokenPoS = nltk.pos_tag(token)
        #effettuo il filtraggio del corpus dai token di punteggiatura listati globalmente per errori riscontrati in precedenza con il pos_tag
        if not token in symbol:
            corpusF += token
        #inserisco i token e i bigrammi del PoS nelle rispettive liste
        corpusTot += token
        posList += tokenPoS
    return corpusTot, len(corpusTot), list(set(corpusTot)), posList, corpusF

#funzione per stampa dei confronti
def confronto(n, m, sentence):
    if n > m:
        print(" - Il testo numero 1 ha", sentence, "maggiore del testo numero 2 (", format(n, 'f'), " > ", format(m, 'f'), ")")
    else:
        print(" - Il testo numero 2 ha", sentence, "maggiore del testo numero 1 (", format(m, 'f'), " > ", format(n, 'f'), ")")

#funzione per calcolo delle lunghezze medie
def avgLen(length, Arr):
    lenTot = 0.0
    for elem in Arr:
        lenTot += len(elem)
    return lenTot/length

#funzione per calcolo degli hapax nei primi 1000 token di un testo
def lenHap(C):
    c1000 = C[0:1000]
    hapax = 0
    #creo una lista ordinata e raggruppata con tutti i token e loro frequenze per filtrare e poter contare solo gli hapax
    DistrFreq = nltk.FreqDist(c1000)
    getList = DistrFreq.most_common(len(DistrFreq))
    for elem in getList:
        if elem[1] == 1:
            hapax += 1
    return hapax

#funzione per calcolo di vocabolario incrementale e TTR relativo, più richiamo stampa confronto
def vocTTR(c1, c2, len1, len2):
    i = 500
    #inizializzo un while anziché un for per la comodità rispetto la lunghezza dei corpus
    while i < len1 and i < len2:
        #creo dei vocabolari parziali in base alla lunghezza incrementale di C e ne calcolo la TTR
        VPar1 = list(set(c1[0:i]))
        VPar2 = list(set(c2[0:i]))
        TTR1 = i/len(VPar1)
        TTR2 = i/len(VPar2)
        print("\nPer un corpus di", i, "token:")
        confronto(len(VPar1), len(VPar2), "un vocabolario")
        confronto(TTR1, TTR2, "un Token Type Ratio")
        i += 500
    #stampo infine il confronto di vocabolario e TTR per il corpus completo
    print("\nA corpus completi:")
    confronto(len(list(set(c1))), len(list(set(c2))), "un vocabolario")
    confronto(len(list(set(c1)))/len1, len(list(set(c2)))/len2, "un Token Type Ratio")

#funzione per distribuzione di frequenza percentuale per parole piene e parole funzionali
def frequenze(pos1, pos2):
    #inizializzo dei contatori per le parole piene e quelle funzionali per entrambi i testi
    piene1 = 0
    piene2 = 0
    funz1 = 0
    funz2 = 0
    #creo delle liste di tuple ordinate per i bigrammi <Token, PoS> e la loro frequenza assoluta
    DFPoS1 = nltk.FreqDist(pos1)
    getList1 = DFPoS1.most_common(len(DFPoS1))
    DFPoS2 = nltk.FreqDist(pos2)
    getList2 = DFPoS2.most_common(len(DFPoS2))
    #per ogni selezione in entrambi i cicli, confronto il PoS con le liste globali per selezionare gli elementi richiesti
    for elem in getList1:
        if elem[0][1] in piene:
            piene1 += elem[1]
        elif elem[0][1] in funzionali:
            funz1 += elem[1]
    for elem in getList2:
        if elem[0][1] in piene:
            piene2 += elem[1]
        elif elem[0][1] in funzionali:
            funz2 += elem[1]
    #ritorno i valori delle frequenze relative in percentuale
    return (piene1/len(pos1)*100), (piene2/len(pos2)*100), (funz1/len(pos1)*100), (funz2/len(pos2)*100)

#inizializzo la funzione main, dove i testi verranno tokenizzati in frasi e successivamente chiamate le funzioni per analisi e confronti 
def main(file1, file2):
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    input1 = codecs.open(file1, mode="r", encoding="utf-8")
    raws1 = input1.read()
    sent1 = sentence_tokenizer.tokenize(raws1)
    input2 = codecs.open(file2, mode="r", encoding="utf-8")
    raws2 = input2.read()
    sent2 = sentence_tokenizer.tokenize(raws2)

    #per ogni testo estraggo corpus e lunghezza, vocabolario, PoS Tag e corpus filtrati dalla punteggiatura
    corpus1, lunghezza1, vocabolario1, PoS1, corpus1F = estraiDati(sent1)
    lenSent1 = len(sent1)
    corpus2, lunghezza2, vocabolario2, PoS2, corpus2F = estraiDati(sent2)
    lenSent2 = len(sent2)

    #inizio del confronto
    print("I testi analizzati hanno le seguenti caratteristiche:")
    
    #stampo il confronto per numero di frasi e lunghezza del corpus
    confronto(lenSent1, lenSent2, "un numero di frasi")
    confronto(lunghezza1, lunghezza2, "la lunghezza del corpus")
    print()
    
    #estraggo le lunghezze medie delle frasi in termini in tok e quella dei token in char (esclusa punteggiatura)
    #la lunghezza media delle frasi è pari alla lunghezza del corpus fratto il numero di frasi di esso, dunque:
    avgF1 = lunghezza1/lenSent1
    avgF2 = lunghezza2/lenSent2
    confronto(avgF1, avgF2, "la lunghezza media delle frasi in tok")
    confronto(avgLen(len(corpus1F), corpus1F), avgLen(len(corpus2F), corpus2F), "la lunghezza media dei token in char")
    
    #confronto il numero di hapax sui primi 1000 token
    hapax1 = lenHap(corpus1)
    hapax2 = lenHap(corpus2)
    confronto(hapax1, hapax2, ", sui primi 1000 token, un numero di hapax")

    #confronto della grandezza del vocabolario ogni 500 token e della TTR
    print("\n/--------------------------------------------------/\n")
    print("Il confronto per la ricchezza lessicale e la grandezza del vocabolario, incrementando di 500 tok per volta:")
    vocTTR(corpus1, corpus2, lunghezza1, lunghezza2)

    #confronto delle percentuali di classi di frequenze
    print("\n/--------------------------------------------------/\n")
    print("Il confronto delle percentuali raggruppati per parole piene e parole funzionali:")
    P1, P2, F1, F2 = frequenze(PoS1, PoS2)
    confronto(P1, P2, "una percentuale di parole piene")
    confronto(F1, F2, "una percentuale di parole funzionali")

main(sys.argv[1],sys.argv[2])