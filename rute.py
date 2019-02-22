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
from itertools import zip_longest


ZASTITNI_OPSEG = 1
KAPACITET_TRANSMITERA = 8
BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI = 3

def parsing_argumts():
    # parsing arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('matrica_linkova', help='Ime fajla za matricu ruta u .tsv file formatu ')
    parser.add_argument('matrica_povezanosti', help='Ime fajla za matricu povezanosti u .tsv file formatu ')

    args = parser.parse_args()
    matrica_linkova_file_name = args.matrica_linkova
    matrica_povezanosti_file_name = args.matrica_povezanosti

    return matrica_linkova_file_name, matrica_povezanosti_file_name 

def load_matrices_from_files_names(matrica_linkova_file_name, 
                                   matrica_povezanosti_file_name):
    
    ## hardcoded values for loading matrices
    matrica_linkova = np.matrix("0 1 1 0;1 0 1 1;1 1 0 1;0 1 1 0")
    matrica_povezanosti = genfromtxt('matrica_povezanosti_sa_papira.tsv', delimiter='\t')
#    matrica_linkova = genfromtxt(matrica_linkova_file_name, delimiter='\t')
#    matrica_povezanosti = genfromtxt(matrica_povezanosti_file_name, delimiter='\t')
    # convert to matrix values to int 
    matrica_povezanosti = matrica_povezanosti.astype(np.int)
    # sortiraj po poocetnom cvoru
    matrica_povezanosti = matrica_povezanosti[matrica_povezanosti[:,0].argsort()] 

    return matrica_linkova, matrica_povezanosti


matrica_linkova, matrica_povezanosti = load_matrices_from_files_names("dummy1", "dummy2")
#### matrica_zahteva
dimenzija_kvadratne_matrice = len(matrica_linkova)
 

# matrica povezanosti je matrica requstetova, cija je dimenzija 3 x broj konekcija koje treba da se ostavare 


# za slucaj da je potrebno samo dati maticu zahteva i matricu linkova
# zato postoje ove dve funcije napravi_matricu_zahteva napravi_matricu_povezanosti_od_matrice_zahteva
def napravi_matricu(donja_granica,
                    gornja_granica, 
                    dimenzija_kvadratne_matrice, 
                    dijagola_nule =False):
    rows = dimenzija_kvadratne_matrice
    columns = dimenzija_kvadratne_matrice
    matrica = np.array([[random.randrange(donja_granica, gornja_granica+1) 
                        for x in range(columns)] for y in range(rows)])
    
    if dijagola_nule:
        np.fill_diagonal(matrica,0)
    return matrica


# pravi matricu zahteva sa nasumicnim vrednostima
matrica_zahteva = napravi_matricu(0,5,dimenzija_kvadratne_matrice, dijagola_nule=True)


def napravi_matricu_povezanosti_od_matrice_zahteva(matrica_zahteva):
    matrica_povezanosti = [[i+1,position+1,number] for i,row in enumerate(matrica_zahteva) for position,number in enumerate(row) if position!=i]
    return matrica_povezanosti
            
napravi_matricu_povezanosti_od_matrice_zahteva(matrica_zahteva)



#crtanje grapha 

def Nacrtaj_graph(matrica_linkova):
    
    G = nx.from_numpy_matrix(matrica_linkova, create_using=nx.MultiDiGraph())
    pos = nx.circular_layout(G)
    nx.draw_circular(G)
    labels = {i : i +1 for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=15)
    plt.show()
    
    return G
########################
Graph = Nacrtaj_graph(matrica_linkova)

def all_paths(Graph, pocetni_cvor, terminalni_cvor):
    # vraca sve moguce rute od pocetnog do terminalnog cvora za dati graph
    generator_paths = nx.all_simple_paths(Graph, pocetni_cvor-1, terminalni_cvor-1)
    paths = [list(map(lambda x:x+1, path))
             for path in generator_paths] 
    #list(nx.all_simple_paths(Graph, pocetni_cvor-1, terminalni_cvor-1))
    
    return paths

paths = all_paths(Graph,3,4)


def Korak_1_pravljenje_pcele_sa_izmesanim_rutama(broj_pcela, matrica_povezanosti): 
    
    #svaki red u matrici povezanosti je jenda ruta, sto znaci da je pcela matrica povezanosti -->samo suffle
    lista_pcela = []
    matrica_povezanosti_copy = np.copy(matrica_povezanosti)
    info_pcela_tup = namedtuple("info_pcela", 'lista_pcela, broj_pcela')
    
    for i in range(broj_pcela):
        np.random.shuffle(matrica_povezanosti_copy) #permute rows 
        matrica_povezanosti_shuffled = np.copy(matrica_povezanosti_copy) 
        lista_pcela.append(matrica_povezanosti_shuffled) 
    info_pcela = info_pcela_tup(lista_pcela,broj_pcela)
    
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
    
    for index, lista_requestova_u_pceli in enumerate(lista_pcela_kopija):
        for request_num, request in enumerate(lista_requestova_u_pceli):
            if lista_requestova_u_pceli[0][0]== request[0]:
                # skida prvu rutu sa pcele i stavlja je u dict koje se zove
                # ruta_za_preracuvanje gde za svaku se skida po jedna ruta
                rute_za_preracuvanje[index].append(request)
            else: 
                nova_pcela.append(request)
        lista_pcela_posle_brisanja_ruta.append(nova_pcela)
        nova_pcela = []
    
    
    return (rute_za_preracuvanje, lista_pcela_posle_brisanja_ruta)


info_pcela = Korak_1_pravljenje_pcele_sa_izmesanim_rutama(broj_pcela = 5,
                                                          matrica_povezanosti = matrica_povezanosti)

pcela = info_pcela.lista_pcela[0]

def nadji_requestove_sa_istim_pocetnim_cvorom_u_pceli(pocetni_request,
                                                      ostatak_requestova):
    requestove_sa_istim_pocetnim_cvorom = [pocetni_request]
    indeksi_requstova_koji_su_ostali = []
    indeksi_requstova_koji_su_uzeti = []
    pocetni_cvor = pocetni_request[0]
    for index,ostk_request in enumerate(ostatak_requestova):
        if ostk_request[0] == pocetni_cvor:
            requestove_sa_istim_pocetnim_cvorom.append(ostk_request)
            indeksi_requstova_koji_su_uzeti.append(BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI + index)
        else:
            indeksi_requstova_koji_su_ostali.append(BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI + index)
            
    
    return requestove_sa_istim_pocetnim_cvorom, indeksi_requstova_koji_su_uzeti, indeksi_requstova_koji_su_ostali
    
def Racunaj_H(G,X,Y,Z,W):
    # X broj zajednickih linkova
    # Y duzina rute koje se razmatra sa grupisanje
    # Z duzina prve izabrane rute
    # W broj slotova te rute -fs
    # G zastitni opseg
    H = 2*G*X-(Y-Z)*W
    return H

    
def proveri_da_li_rute_sadrze_iste_cvorove(ruta_1, ruta_2):
    
    len_of_ruta_2 = len(ruta_2)
    index_na_kome_je_predhodni_match = -1
    # broj zajednckih linkova je -1 posto su potrebna dva uzastopna match-a jesu link
    broj_zajednickih_linkova = -1
    for i_1, cvor_1 in enumerate(ruta_1):
        if i_1 == len_of_ruta_2:
            break
        preklapaju_se = False
        for i_2,cvor_2 in enumerate(ruta_2):
            if cvor_1 == cvor_2:
                print(index_na_kome_je_predhodni_match,i_2)
                if i_2 > index_na_kome_je_predhodni_match:
                    preklapaju_se = True
                    if i_2 - index_na_kome_je_predhodni_match == 1 :
                        broj_zajednickih_linkova = broj_zajednickih_linkova + 1
                else:
                    return False
                index_na_kome_je_predhodni_match = i_2
                break
    
    return preklapaju_se, broj_zajednickih_linkova
    
    
def ako_se_praklapaju_racunaj_h(ruta_sa_kojom_se_poredi,
                                potencijalne_rute):
    
    *ruta_sa_kojom_se_poredi_bez_fs, fs = ruta_sa_kojom_se_poredi
    len_of_ruta_sa_kojom_se_poredi_bez_fs = len(ruta_sa_kojom_se_poredi_bez_fs)
    fs_potencijalne_rute = potencijalne_rute[-1][1]
    #uzmi prvu rutu i napravi dummy samo da bi moglo da se poredi
    predhodna_ruta_sa_H_vece_od_0 = (potencijalne_rute[0],0)
    for ruta in potencijalne_rute[:-1]:
        preklapaju_se, broj_zajednickih_linkova = proveri_da_li_rute_sadrze_iste_cvorove(ruta_sa_kojom_se_poredi_bez_fs,
                                                                                        ruta)
        if preklapaju_se is True:
            G = ZASTITNI_OPSEG # promeni ovo posle
            len_ruta = len(ruta)
            H = Racunaj_H(G,
                          broj_zajednickih_linkova,
                          len_ruta,
                          len_of_ruta_sa_kojom_se_poredi_bez_fs,
                          fs_potencijalne_rute)
            if H>0:
                if predhodna_ruta_sa_H_vece_od_0[1] < H:
                    predhodna_ruta_sa_H_vece_od_0 = (ruta, H)
                elif predhodna_ruta_sa_H_vece_od_0[1] == H:
                    # biraj kracu rutu
                    if len(predhodna_ruta_sa_H_vece_od_0[0])>len_ruta:
                        predhodna_ruta_sa_H_vece_od_0 = (ruta, H)
        
    ruta_sa_najvecim_H = predhodna_ruta_sa_H_vece_od_0
    
    return ruta_sa_najvecim_H
            
        
        

    # iF je h veca od nule soritaj od najmanje ka najvecoj 
    # sve sto su posle i uzmi prvih n tako da zbir
    # fs ne predje U, kapacitet tranzistora   \
    
    # Kada se popunjava matrica popunjenosti krace rute stavljati desno
    # na taj nacin ce one prve da se zavrse i tako nece biti praznih slotova
    

def trazi_sve_preklapajuce_rute_medju_requestovima_sa_istim_pocetnim_cvorom(indeksi_requstova_koji_su_ostali,
                                                                            indeksi_requstova_koji_su_uzeti,
                                                                            requestove_sa_istim_pocetnim_cvorom):
    # Moras i indexe ovde da prosledis posto ukoliko ne postoji ruta koja se preklapa 
    # ili ako je h < 0 da bi je vratio u listu indeksi_requstova_koji_su_ostali
    Graph = nx.from_numpy_matrix(matrica_linkova, create_using=nx.MultiDiGraph())
    broj_pcele_i_najduza_ruta_i_sve_njene_podrute_list = []
    kapacitet_transmitera = 0
    lista_odabranih_ruta_i_fs_vrednosti = []
    # request_za_koji_se_traze_preklapajuce_rute
    pocetni_cvor, terminalni_cvor, fs = requestove_sa_istim_pocetnim_cvorom[0]
    # uzmi najkracu rutu
    ruta_sa_kojom_se_poredi = min(all_paths(Graph,pocetni_cvor, terminalni_cvor),key = lambda x: len(x))
    ruta_sa_kojom_se_poredi.append(("fs_vrednost",fs))
    lista_odabranih_ruta_i_fs_vrednosti.append(ruta_sa_kojom_se_poredi)
    
    for index,ruta in enumerate(requestove_sa_istim_pocetnim_cvorom[1:]): 
        #print('ruta je ', ruta)
        pocetni_cvor, terminalni_cvor, fs = ruta 
        potencijalne_rute = all_paths(Graph,pocetni_cvor, terminalni_cvor)
        potencijalne_rute.append(("fs_vrednost",fs))
        ruta_sa_najvecim_H = ako_se_praklapaju_racunaj_h(ruta_sa_kojom_se_poredi,
                                                        potencijalne_rute)
       
        # ovde dodaj kapacitet provodnika U i nekako sredi indeksi_requstova_koji_su_ostali ako pretekne preko 

        H = ruta_sa_najvecim_H[1]
        if H > 0:
            #ako je vece od nule dodaj za transmisiju
            if kapacitet_transmitera + H < KAPACITET_TRANSMITERA:
                kapacitet_transmitera += H
                lista_odabranih_ruta_i_fs_vrednosti.append([ruta_sa_najvecim_H[0],("fs_vrednost",fs)])
            else:
                indeksi_requstova_koji_su_ostali.append(indeksi_requstova_koji_su_uzeti[index])
        else:
            indeksi_requstova_koji_su_ostali.append(indeksi_requstova_koji_su_uzeti[index])
               
       
    # menjanje rute sa kojim se poredi da bude lista sa dva elementa, negde sam zajebao pa je ovo q&d 
    lista_odabranih_ruta_i_fs_vrednosti[0] = [lista_odabranih_ruta_i_fs_vrednosti[0][:-1], lista_odabranih_ruta_i_fs_vrednosti[0][-1]]
    
    
    return lista_odabranih_ruta_i_fs_vrednosti, indeksi_requstova_koji_su_ostali
       




def uzmi_n_requsteova_sa_pocekta_pcele(pcela,matrica_slotova):
    n_broj_requestova = BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI
    # za svaki od 3 requesta pronadji redom koji requstovi iz ostatka pcele sadrze
    # zajednicki pocetni cvor 
    ostatak_requestova = pcela[n_broj_requestova:]
    for i in range(n_broj_requestova):
        pocetni_request = pcela[i]
        # nadji_requestove_sa_istim_pocetnim_cvorom_u_pceli
        requestove_sa_istim_pocetnim_cvorom, indeksi_requstova_koji_su_uzeti, indeksi_requstova_koji_su_ostali = nadji_requestove_sa_istim_pocetnim_cvorom_u_pceli(pocetni_request,
                                                                                                                                                                   ostatak_requestova)
        
        
        lista_odabranih_ruta_i_fs_vrednosti, indeksi_requstova_koji_su_ostali = trazi_sve_preklapajuce_rute_medju_requestovima_sa_istim_pocetnim_cvorom(indeksi_requstova_koji_su_ostali,
                                                                                                                                                        indeksi_requstova_koji_su_uzeti,
                                                                                                                                                        requestove_sa_istim_pocetnim_cvorom)
        ostatak_requestova = pcela[indeksi_requstova_koji_su_ostali]
        
        matrica_slotova_as_dict
        
        
        
        
        
        
        
        
        
        
        # nadji_sve_moguce_ruta_na_osnovu_kojih_se_moze_resiti_request
        paths = all_paths(Graph,3,4)
        
        
        # racunaj H ovde
        
    
def napravi_matricu_slotova(matrica_linkova):
    Graph = Nacrtaj_graph(matrica_linkova)
    matrica_slotova_as_dict = {"_".join([str(edge[0]+1),str(edge[1]+1)]):[0] for edge in Graph.edges}
    
    return matrica_slotova_as_dict


def racunaj_startu_poziciju_za_popunjavanje_matrice(rute, matrica_slotova_as_dict):
    startna_pozicija_za_popunjavanje_matrice_slotova = 0
    for ruta_i_fs in rute:
        ruta = ruta_i_fs[0]
        len_of_ruta = len(ruta)
        for i in range(len_of_ruta):
            if i != len_of_ruta-1:
                link_izmedju_cvorova = "_".join([str(cvor) for cvor in ruta[i:i+2]])
                broj_slobodnog_slota_na_linku = matrica_slotova_as_dict[link_izmedju_cvorova][-1] 
            if broj_slobodnog_slota_na_linku > startna_pozicija_za_popunjavanje_matrice_slotova:
                startna_pozicija_za_popunjavanje_matrice_slotova = broj_slobodnog_slota_na_linku
                
    return startna_pozicija_za_popunjavanje_matrice_slotova


def izbaci_rute_koji_imaju_zauzete_linkove_koje_ruta_za_porednjenje_ne_sadrzi(matrica_slotova_as_dict,
                                                                              ruta_za_porednjenje,
                                                                              rute):
    startna_pozicija_za_popunjavanje_matrice_slotova = racunaj_startu_poziciju_za_popunjavanje_matrice([ruta_za_porednjenje], matrica_slotova_as_dict)
    # posto me zanima broj slota bez zadnjeg granicnika ZASTITNI_OPSEG
    zbir_fs = ZASTITNI_OPSEG + startna_pozicija_za_popunjavanje_matrice_slotova + 1
    rute_koje_imaju_slobodne_slotove = []
    rute_koje_imaju_slobodne_slotove.append(ruta_za_porednjenje)
    for index,cvorovi_i_fs in enumerate(rute):
        zadrzi_rutu = True
        cvorovi, fs = cvorovi_i_fs
        len_cvorovi = len(cvorovi)
        for index_cvor,cvor in enumerate(cvorovi):
            if cvor not in cvorovi_rute_za_porednjenje:
                
                #vrati broj slota
                link_izmedju_cvorova_list = []
                if len_cvorovi == index_cvor:
                   link_izmedju_cvorova_list.append("_".join([str(cvorovi[index_cvor-1]),str(cvor)]))
                
                else:
                    # cvor koji ruta za porednjenje ne sadrzi nije na kraju 
                    # OVO MOZDA BUDE MORALO DA SE MENJA JER MOZE POSTOJIATI SLUCAJ kada svaki link izmedju dva cvora je slobodan ali je umetnuto vise linkova pa se svi moraju uzeti u obzir
                    link_izmedju_cvorova_list.append("_".join([str(cvorovi[index_cvor-1]),str(cvor)]),
                                                     "_".join([str(cvor),str(cvorovi[index_cvor+1])]))
                
                for link_izmedju_cvorova in link_izmedju_cvorova_list:
                    if matrica_slotova_as_dict[link_izmedju_cvorova][-1] > zbir_fs:
                        zadrzi_rutu = False
                    
        if zadrzi_rutu is True:
           rute_koje_imaju_slobodne_slotove.append(cvorovi_i_fs) 
           zbir_fs += fs[1]
    
    return rute_koje_imaju_slobodne_slotove
    

def Popunjavanje_matrice_slotova(lista_odabranih_ruta_i_fs_vrednosti,matrica_slotova):
    ruta_za_porednjenje = lista_odabranih_ruta_i_fs_vrednosti[0]
    len_prva_ruta = len(prva_ruta[0])
    rute_za_grupisanje = lista_odabranih_ruta_i_fs_vrednosti[1:]
    rute_za_grupisanje_sortirane_od_najvece_ka_najmanjoj = sorted(rute_za_grupisanje,key = lambda x: len(x[0]), reverse= True)
    
    rute = rute_za_grupisanje_sortirane_od_najvece_ka_najmanjoj
    
    rute_koje_imaju_slobodne_slotove = izbaci_rute_koji_imaju_zauzete_linkove_koje_ruta_za_porednjenje_ne_sadrzi(matrica_slotova_as_dict,
                                                                                                                 ruta_za_porednjenje,
                                                                                                                 rute_za_grupisanje_sortirane_od_najvece_ka_najmanjoj)
    
    
    startna_pozicija_za_popunjavanje_matrice_slotova = racunaj_startu_poziciju_za_popunjavanje_matrice(rute)
            
    
    # popuni matiricu bez pretpostavke umetnutih cvorova, ali radi sa rutama koje su duze po pocetne i sve cvorovi se preklapaju
    len_of_ruta = len(prva_ruta)
    rute_za_grupisanje_sortirane_od_najvece_ka_najmanjoj_copy = copy.deepcopy(rute_za_grupisanje_sortirane_od_najvece_ka_najmanjoj)
    
    
    zbir_svih_fs = startna_pozicija_za_popunjavanje_matrice_slotova + 1 + 2*ZASTITNI_OPSEG + sum(ruta[-1][1] for ruta in rute)
    
   # minus 2 je posto kad ostane jedan link na najduzoj ruti, tad se zavrsava
   for broj_koraka in range(len(max(rute,key=len))-2): 
       zbir_svih_fs_po_koraku = copy.deepcopy(zbir_svih_fs) 
       ############ VIDI KAKO DA KADA SE OBRISE POCETNA RUTA IPAK RACUNAS RUPU KOJA OSTANE LEVO ZBOG NJE
               #ostala je samo fs vrednost kao tuple u listi
               fs = ruta[0][1]
               zbir_svih_fs_po_koraku -= fs                 
               
            if len(ruta)==3:
                #skidaj oba cvora ako je ostao samo jedan link i fs vrednost
                prvi_cvor = ruta.pop(0)
                drugi_cvor = ruta.pop(0)
            
            if len(ruta)>3:
                prvi_cvor = ruta.pop(0)
                drugi_cvor = ruta[0]
                    
        link_izmedju_cvorova = "_".join([str(prvi_cvor),str(drugi_cvor)])
        matrica_slotova_as_dict[link_izmedju_cvorova].append(zbir_svih_fs_po_koraku) 
            
            
    
    
    
    
    
    



# korak 2.1 ce se rekurzivno zvati sa lista_pcela_posle_brisanja_ruta dok se ne obrise cela pcela 

Korak_2_1(info_pcela)  


def Korak_3(rute_za_preracuvanje, matrica_linkova):
    
    Graph = nx.from_numpy_matrix(matrica_linkova, create_using=nx.MultiDiGraph())
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
               mozda_najduza_ruta = max(potencijalne_rute,key=len)
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
        
    



info_pcela = Korak_1_pravljenje_pcele_sa_izmesanim_rutama(broj_pcela = 5,
                                                          matrica_povezanosti = matrica_povezanosti)
lista_pcela = info_pcela.lista_pcela
broj_pcela = info_pcela.broj_pcela
print(broj_pcela, lista_pcela)

rute_za_preracuvanje, lista_pcela_posle_brisanja_ruta = Korak_2_1(info_pcela) 

broj_pcele_i_najduza_ruta_i_sve_njene_podrute_list = Korak_3(rute_za_preracuvanje, matrica_linkova)


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
    

a = Korak_1_pravljenje_pcele_sa_izmesanim_rutama(5,matrica_povezanosti)
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
#    def __init__(self,matrica_linkova, matrica_povezanosti):
#        self.matrica_linkova = matrica_linkova
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