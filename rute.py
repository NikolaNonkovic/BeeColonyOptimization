#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from numpy import genfromtxt
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt
import networkx as nx
import math
import random
import copy

# parsing arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('matrica_ruta', help='Ime fajla za matricu ruta u .tsv file formatu ')
parser.add_argument('matrica_povezanosti', help='Ime fajla za matricu povezanosti u .tsv file formatu ')

args = parser.parse_args()
matrica_ruta_file_name = args.matrica_ruta
matrica_povezanosti_file_name = args.matrica_povezanosti

####
matrica_ruta = genfromtxt(matrica_ruta_file_name, delimiter='\t')
matrica_povezanosti = genfromtxt(matrica_povezanosti_file_name, delimiter='\t')
######

matrica_povezanosti = genfromtxt('matrica_povezanosti_sa_papira.tsv', delimiter='\t')

matrica_ruta = np.matrix("0 1 1 0;1 0 1 1;1 1 0 1;0 1 1 0")
matrica_povezanosti = matrica_povezanosti.astype(np.int)
matrica_povezanosti = matrica_povezanosti[matrica_povezanosti[:,0].argsort()] # sortirana_po_poocetnom_cvoru

#### matrica_zahteva
dimenzija_kvadratne_matrice = len(matrica_ruta)

def napravi_matricu(donja_granica,gornja_granica, dimenzija_kvadratne_matrice, dijagola_nule =False):
    
    rows = dimenzija_kvadratne_matrice
    columns = dimenzija_kvadratne_matrice
    matrica = np.array([[random.randrange(donja_granica, gornja_granica+1) 
                        for x in range(columns)] for y in range(rows)])
    
    if dijagola_nule:
        np.fill_diagonal(matrica,0)
    return matrica


matrica_zahteva = napravi_matricu(0,5,dimenzija_kvadratne_matrice, dijagola_nule=True)


def napravi_matricu_povezanosti_od_matrice_zahteva(matrica_zahteva):
    matrica_povezanosti = [[i+1,position+1,number] for i,row in enumerate(matrica_zahteva) for position,number in enumerate(row) if position!=i]
    return matrica_povezanosti
            
napravi_matricu_povezanosti_od_matrice_zahteva(matrica_zahteva)



#crtanje grapha 

def Nacrtaj_graph(matrica_ruta):
    
    G = nx.from_numpy_matrix(matrica_ruta, create_using=nx.MultiDiGraph())
    pos = nx.circular_layout(G)
    nx.draw_circular(G)
    labels = {i : i +1 for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=15)
    plt.show()
    
    return G
########################
Graph = Nacrtaj_graph(matrica_ruta)


def all_paths(Graph, pocetni_cvor, terminalni_cvor):
    # vraca sve moguce rute od pocetnog do terminalnog cvora za dati graph
    generator_paths = nx.all_simple_paths(Graph, pocetni_cvor-1, terminalni_cvor-1)
    paths = [list(map(lambda x:x+1, path)) \
             for path in generator_paths] 
    #list(nx.all_simple_paths(Graph, pocetni_cvor-1, terminalni_cvor-1))
    
    return paths

paths = all_paths(Graph,3,4)



# da li treba da pravim matricu_zahteva
def Korak_1(broj_pcela, matrica_povezanosti): 
    
    #svaki red u matrici povezanosti je jenda ruta, sto znaci da je pcela matrica povezanosti -->samo suffle
    lista_pcela = []
    matrica_povezanosti_copy = np.copy(matrica_povezanosti)
    info_pcela = namedtuple("info_pcela", 'lista_pcela, broj_pcela')
    
    for i in range(broj_pcela):
        np.random.shuffle(matrica_povezanosti_copy) #permute rows 
        matrica_povezanosti_shuffled = np.copy(matrica_povezanosti_copy) 
        lista_pcela.append(matrica_povezanosti_shuffled) 
    info_pcela.broj_pcela = broj_pcela
    info_pcela.lista_pcela = lista_pcela
    
    return info_pcela


def Korak_2(info_pcela):
    
    lista_pcela = info_pcela.lista_pcela
    broj_pcela = info_pcela.broj_pcela
    lista_pcela_kopija = list(lista_pcela)
    lista_pcela_kopija = [pcela.tolist() for pcela in lista_pcela_kopija] 
    dict_pcela = {}
    lista_pcela_posle_brisanja_ruta = []
    nova_pcela = []
    
    for i in range(broj_pcela):
        key = "pcela_broj_{i}".format(i=i)
        dict_pcela[key] = [] 
    print(dict_pcela)
    
    for index, pcela in enumerate(lista_pcela_kopija):
        for broj_pcela, ruta in enumerate(pcela):
            if pcela[0][0]== ruta[0]:
                dict_pcela["pcela_broj_{i}".format(i=index)].append(ruta)
            else: 
                nova_pcela.append(ruta)
        lista_pcela_posle_brisanja_ruta.append(nova_pcela)
        nova_pcela = []
    
    return (dict_pcela, lista_pcela_posle_brisanja_ruta) # za ovaj deo cu takodje morati da proveravam u matrici ruta na koji sve nacin se moze doci iz pocetnog do kranjeg cvora
  
def Korak_2_1(info_pcela):
    
    lista_pcela = info_pcela.lista_pcela
    broj_pcela = info_pcela.broj_pcela
    lista_pcela_kopija = list(lista_pcela)
    lista_pcela_kopija = [pcela.tolist() for pcela in lista_pcela_kopija] 
    rute_za_preracuvanje = []
    lista_pcela_posle_brisanja_ruta = []
    nova_pcela = []
    
    for i in range(broj_pcela):
        rute_za_preracuvanje.append([i])
    
    for index, pcela in enumerate(lista_pcela_kopija):
        for broj_pcela, ruta in enumerate(pcela):
            if pcela[0][0]== ruta[0]:
                rute_za_preracuvanje[index].append(ruta)
            else: 
                nova_pcela.append(ruta)
        lista_pcela_posle_brisanja_ruta.append(nova_pcela)
        nova_pcela = []
    
    
    return (rute_za_preracuvanje, lista_pcela_posle_brisanja_ruta)

# korak 2.1 ce se rekurzivno zvati sa lista_pcela_posle_brisanja_ruta dok se ne obrise cela pcela 

Korak_2_1(info_pcela)  


def Korak_3(rute_za_preracuvanje, matrica_ruta):
    
    Graph = nx.from_numpy_matrix(matrica_ruta, create_using=nx.MultiDiGraph())
    broj_pcele_i_najduza_ruta_i_sve_njene_podrute_list = []
    for broj_pcele, rute_sa_istim_pocetkom in enumerate(rute_za_preracuvanje):
        najduza_ruta = 0
        mozda_najduza_ruta = []
        lista_potencijalnih_ruta_i_fs_vrednosti = []
        for ruta in rute_sa_istim_pocetkom: # pazi da ne uhvatis broj pcele koji je hardcoded u listi
            if isinstance(ruta,list):
               #print('ruta je ', ruta)
               pocetni_cvor, terminalni_cvor, fs = ruta 
               potencijalne_rute = all_paths(Graph,pocetni_cvor, terminalni_cvor)
               #print('potencijalne_rute su ', potencijalne_rute)
               # dodati fs na svaku rutu
               potencijalne_rute.append(("fs_vrednost",fs))
               #print('potencijalne_rute_i_fs_vrednost su ', potencijalne_rute_i_fs_vrednost)
               lista_potencijalnih_ruta_i_fs_vrednosti.append(potencijalne_rute)
               mozda_najduza_ruta == max(potencijalne_rute,key=len)
               if najduza_ruta == 0:
                   najduza_ruta = max(potencijalne_rute,key=len)
               
               if len(mozda_najduza_ruta) > len(najduza_ruta):
                   najduza_ruta = mozda_najduza_ruta
                   najduza_ruta.append(("fs_vrednost",fs))
        
        najduza_ruta_i_sve_njene_podrute = []
        for lista_potencijalnih_ruta in lista_potencijalnih_ruta_i_fs_vrednosti: # nije optimizovano je ce porediti i najduzu rutu sa najduzom i tek je tad ubaciti u  najduza_ruta_i_sve_njene_podrute
            for potencijlna_ruta in lista_potencijalnih_ruta:
                if isinstance(ruta,list):
                    counter = 0
                    len_za_sad_najduze_podrute = 0
                    for redni_broj, cvor in enumerate(potencijlna_ruta):
                        if najduza_ruta[redni_broj] == cvor and counter == redni_broj:
                            counter += 1
                    len_potencijlna_ruta =  len(potencijlna_ruta)
                    if len_potencijlna_ruta == counter:
                        if len_za_sad_najduze_podrute <len_potencijlna_ruta:
                            len_za_sad_najduze_podrute = len_potencijlna_ruta
                            najduza_podruta = potencijlna_ruta
                            najduza_podruta.append(lista_potencijalnih_ruta[-1]) # dodaj fs vrednost
                            
            
            najduza_ruta_i_sve_njene_podrute.append(najduza_podruta)
    
        najduza_ruta_i_sve_njene_podrute.sort(key=len, reverse = True)
        broj_pcele_i_najduza_ruta_i_sve_njene_podrute_list.append([("broj_pcele", broj_pcele),najduza_ruta_i_sve_njene_podrute])
        #print("broj_pcele_i_najduza_ruta_i_sve_njene_podrute  ", najduza_ruta_i_sve_njene_podrute)
        
        
        
    return broj_pcele_i_najduza_ruta_i_sve_njene_podrute_list


def Korak_4(broj_pcele_i_najduza_ruta_i_sve_njene_podrute, zastitni_opseg):
    broj_pcele_i_najduza_ruta_i_sve_njene_podrute_copy = copy.deepcopy(broj_pcele_i_najduza_ruta_i_sve_njene_podrute)
    broj_pcele, lista_ruta = broj_pcele_i_najduza_ruta_i_sve_njene_podrute_copy
    #Hr = (2*GB*x1-(x2-x3))+1 # izracunvanje verovatnoce
    # GB must be int 
    random.seed(123)
    lista_H = []
    lista_p = []
    sum_of_exp_H = 0
    GB = int(zastitni_opseg)
    x3 = len(lista_ruta[0])-1 # -1 je zbog tuple-a
    
    
    for ruta in lista_ruta[1:]:
        x2 = len(ruta) -1 
        x1 = x2-1
        H = (2*GB*x1-(x2-x3))+1
        sum_of_exp_H += math.exp(H)
        lista_H.append(H)
        
    
    len_lista_ruta = len(lista_ruta)
    #ista_ruta[0].append(("p_randomizovano", 1))
    for index,H in enumerate(lista_H):
        p = math.exp(H)/sum_of_exp_H
        #random_broj = random.random()
        #p_randomizovano = p*random_broj
        lista_p.append(p)
        if (index +1) < len_lista_ruta:
            lista_ruta[index+1].append(("p", p))
    
    #lista_ruta.sort(key=lambda x: x[][1]) # vidi kako samo da pozoves ovo da se sortira na osnovu p_randomizovano
    return lista_ruta
        
    



info_pcela = Korak_1(5,matrica_povezanosti)
lista_pcela = info_pcela.lista_pcela
broj_pcela = info_pcela.broj_pcela
print(broj_pcela, lista_pcela)

rute_za_preracuvanje, lista_pcela_posle_brisanja_ruta = Korak_2_1(info_pcela) 

broj_pcele_i_najduza_ruta_i_sve_njene_podrute_list = Korak_3(rute_za_preracuvanje, matrica_ruta)


broj_pcele_i_najduza_ruta_i_sve_njene_podrute = copy.deepcopy(list(broj_pcele_i_najduza_ruta_i_sve_njene_podrute_list[0]))

#broj_pcele_i_najduza_ruta_i_sve_njene_podrute = list(broj_pcele_i_najduza_ruta_i_sve_njene_podrute_list[0])

ulaz_za_korak_5 = Korak_4(broj_pcele_i_najduza_ruta_i_sve_njene_podrute, 1)



def Korak_5(ulaz_za_korak_5,matrica_zauzetosti,dimenzija_kvadratne_matrice=False):
    if dimenzija_kvadratne_matrice:
        matrica_zauzetosti = napravi_matricu(0,0,dimenzija_kvadratne_matrice)
    sorted_by_p_value = sorted(ulaz_za_korak_5,key=lambda x: x[-1][1], reverse=True)
    resctructured_matrix = []
    len_of_maximal_rute = len(sorted_by_p_value[0])-1
    for index, ruta in enumerate(sorted_by_p_value): 
        if index == 0:
            broj_cvorova = ruta[0:-1]
        else:
            broj_cvorova = ruta[0:-2]
        len_broj_cvorova= len(broj_cvorova)
        if len_broj_cvorova!=len_of_maximal_rute:
            broj_cvorova.extend([0] * (len_of_maximal_rute - len_broj_cvorova)) # ovde se produzuje sa nulama da budu svi iste duzine
            tezina = ruta[-2][1]
        else:
            tezina = ruta[-1][1]
        
        resctructured_matrix.append([broj_cvorova,tezina])
    for ruta in resctructured_matrix:
        first,second, *_ = ruta[0]
        
        #sada samo samo ides po dva cvora i dodajes tezine 


def walking(ruta):
    
    first,second, *_ = ruta[0][0]
    
    first,second, *_ = ruta[0]
    

a = Korak_1(5,matrica_povezanosti)
a, b = lista_pcela
[ruta for pcela in lista_pcela for ruta in pcela if pcela[0][0]== ruta[0]  ]
 
 


np.random.shuffle(matrica_povezanosti)


import collections

Count = collections.namedtuple('Count', 'letter, number')

def letter_count(letters, target):
    counter = collections.Counter(letters)
    return [Count(c, counter[c]) for c in target]


#
#class bee():
#    
#    def __init__(self,matrica_ruta, matrica_povezanosti):
#        self.matrica_ruta = matrica_ruta
#        self.matrica_povezanosti = matrica_povezanosti
#        
#    
matrica_povezanosti_shuffled = np.copy(matrica_povezanosti)
np.random.shuffle(matrica_povezanosti)

matrica_povezanosti_shuffled_1 = list(matrica_povezanosti)





arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
arr
np.delete(arr, 1, 0)



a = [[3, 4,], [3, 2, 1], [3, 1, 1]]

max(a, key=len)




baba = [('broj_pcele', 2),
  [[2, 4, 3, 1, ('fs_vrednost', 2)],
   [2, 4, 3, ('fs_vrednost', 2)],
   [2, 4, ('fs_vrednost', 1)]]]


random.random()