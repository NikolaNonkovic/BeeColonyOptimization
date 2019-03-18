#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from numpy import genfromtxt
from collections import namedtuple
import matplotlib.pyplot as plt
import networkx as nx
import math
import random
import copy


ZASTITNI_OPSEG = 1
KAPACITET_TRANSMITERA = 8
BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI = 3
BROJ_RUTA_K = 3

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


def napravi_matricu_povezanosti_od_matrice_zahteva(matrica_zahteva):
    matrica_povezanosti = [[i+1,position+1,number] for i,row in enumerate(matrica_zahteva) for position,number in enumerate(row) if position!=i]
    return matrica_povezanosti
            



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

def all_paths(Graph, pocetni_cvor, terminalni_cvor):
    # vraca sve moguce rute od pocetnog do terminalnog cvora za dati graph
    generator_paths = nx.all_simple_paths(Graph, pocetni_cvor-1, terminalni_cvor-1)
    paths = [list(map(lambda x:x+1, path))
             for path in generator_paths] 
    #list(nx.all_simple_paths(Graph, pocetni_cvor-1, terminalni_cvor-1))
    
    return paths


def pravljenje_liste_pcela_sa_izmesanim_rutama(broj_pcela, 
                                               matrica_povezanosti,
                                               Graph): 
    
    pcela_parcijalno_resenje_tup = namedtuple("pcela_parcijalno_resenje", "rute, trenutni_fs, matrica_slotova")

    #svaki red u matrici povezanosti je jenda ruta, sto znaci da je pcela matrica povezanosti -->samo suffle
    lista_pcela = []
    matrica_povezanosti_copy = np.copy(matrica_povezanosti)
    trenutni_fs = 0
    matrica_slotova = []
    matrica_slotova = napravi_matricu_slotova(Graph)
        
    for i in range(broj_pcela):
        np.random.shuffle(matrica_povezanosti_copy) #permute rows 
        matrica_povezanosti_shuffled = np.copy(matrica_povezanosti_copy) 
        pcela = pcela_parcijalno_resenje_tup(matrica_povezanosti_shuffled,trenutni_fs,matrica_slotova)
        lista_pcela.append(pcela) 
    
    return lista_pcela


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

    
def proveri_da_li_rute_sadrze_iste_cvorove(ruta_1, ruta_2, index, matrica_slotova):
    
    
    # [2,4,1,3] i [2,4,1,2] bice False, ovaj slucaj je izbacen jer ga je tesko isprogramirati
    len_ruta_1 = len(ruta_1)
    len_ruta_2 = len(ruta_2)
    ostatak_cvorova = None
    
    if len_ruta_2>len_ruta_1:
        broj_cvorova_koji_su_ostatak = len_ruta_2 -len_ruta_1
        # -broj_cvorova_koji_su_ostatak+1 znaci uzmi sve cvorove koje se sadrze u ruti 2 i
        # poslednji zajednicki cvor
        ostatak_cvorova = ruta_2[-(broj_cvorova_koji_su_ostatak+1):]
        preklapaju_se = True if ruta_1 == ruta_2[:-broj_cvorova_koji_su_ostatak] else False
        if preklapaju_se is True:
            ruta_sa_najmanjim_indeksom = racunaj_startu_poziciju_za_popunjavanje_matrice([ostatak_cvorova],matrica_slotova)
            preklapaju_se = False if ruta_sa_najmanjim_indeksom["index_u_matrici"]>index else True
        broj_zajednickih_linkova = len_ruta_1 - 1
            
    else:
        for par_cvor in zip(ruta_1,ruta_2):
            preklapaju_se = True if par_cvor[0] == par_cvor[1] else False
            
            if preklapaju_se is False:
                break
            
        broj_zajednickih_linkova = len_ruta_2 - 1
    
    return preklapaju_se, broj_zajednickih_linkova, ostatak_cvorova
    
    
    
def ako_se_praklapaju_racunaj_h(ruta_sa_kojom_se_poredi,
                                potencijalne_rute,
                                index_u_matrici_rute_koja_se_poredi,
                                matrica_slotova):
    
    ruta_sa_kojom_se_poredi_bez_fs, fs = ruta_sa_kojom_se_poredi
    len_of_ruta_sa_kojom_se_poredi_bez_fs = len(ruta_sa_kojom_se_poredi_bez_fs)
    fs_potencijalne_rute = potencijalne_rute[-1][1]
    #uzmi prvu rutu i napravi dummy samo da bi moglo da se poredi
    predhodna_ruta_sa_H_vece_od_0 = (potencijalne_rute[0],0)
    ruta_sa_najvecim_H_ostatak_cvorova = 0
    # do :-1 posto je fs zadnji tuple koji treba se preskoci
    for ruta in potencijalne_rute[:-1]:
        preklapaju_se, broj_zajednickih_linkova, ostatak_cvorova = proveri_da_li_rute_sadrze_iste_cvorove(ruta_sa_kojom_se_poredi_bez_fs,
                                                                                                          ruta,
                                                                                                          index_u_matrici_rute_koja_se_poredi,
                                                                                                          matrica_slotova)
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
                    ruta_sa_najvecim_H_ostatak_cvorova = ostatak_cvorova
                elif predhodna_ruta_sa_H_vece_od_0[1] == H:
                    # biraj kracu rutu
                    if len(predhodna_ruta_sa_H_vece_od_0[0])>len_ruta:
                        ruta_sa_najvecim_H_ostatak_cvorova = ostatak_cvorova
                        predhodna_ruta_sa_H_vece_od_0 = (ruta, H)
        
    ruta_sa_najvecim_H = predhodna_ruta_sa_H_vece_od_0
    
    return ruta_sa_najvecim_H, ruta_sa_najvecim_H_ostatak_cvorova
            
        

# [ostatak_cvorova] wrapovano je u listu posto prima rute a ne rutu

    

def trazi_sve_preklapajuce_rute_medju_requestovima_sa_istim_pocetnim_cvorom(indeksi_requstova_koji_su_ostali,
                                                                            indeksi_requstova_koji_su_uzeti,
                                                                            requestove_sa_istim_pocetnim_cvorom,
                                                                            Graph,
                                                                            matrica_slotova):
  
    kapacitet_transmitera = 0
    broj_ruta_sa_ostatkom_cvorova = 0
    lista_odabranih_ruta_i_meta_informacija = []
    # request_za_koji_se_traze_preklapajuce_rute
    pocetni_cvor, terminalni_cvor, fs = requestove_sa_istim_pocetnim_cvorom[0]
    # uzmi rutu koja ima najmanji index, nebitna je duzina u prvom koraku
    rute = sorted(all_paths(Graph,pocetni_cvor, terminalni_cvor),key = lambda x: len(x))[:BROJ_RUTA_K]
    ruta_sa_najmanjim_indeksom = racunaj_startu_poziciju_za_popunjavanje_matrice(rute, matrica_slotova)
    ruta_sa_kojom_se_poredi = [ruta_sa_najmanjim_indeksom["ruta"],("fs_vrednost",fs)]
    index_u_matrici_rute_koja_se_poredi = ruta_sa_najmanjim_indeksom["index_u_matrici"]
    ###   "index_u_matrici"
    lista_odabranih_ruta_i_meta_informacija.append(ruta_sa_kojom_se_poredi)
    
    
    if len(ruta_sa_kojom_se_poredi) != 2:
        # ako je jednako tri znaci da je ruta_sa_kojom_se_poredi duzine 2 i da ne postoji grupisanje
        # npr [4, 5, ('fs_vrednost', 1)]
    
        for index,ruta in enumerate(requestove_sa_istim_pocetnim_cvorom[1:]): 
            #print('ruta je ', ruta)
            pocetni_cvor, terminalni_cvor, fs = ruta 
            # PROMENI OVO nemoj sorted nego sort ili cak sve izbaci da path kad vraca, vraca samo najkrace, ako moze
            potencijalne_rute = sorted(all_paths(Graph,pocetni_cvor, terminalni_cvor),key = lambda x: len(x))[:BROJ_RUTA_K]
            # PROMENI OVO pravi samo K najkracih ruta u grafu, za pocetni i krajni cvor, manje ce biti komputaciono zahtevno
            
            potencijalne_rute.append(("fs_vrednost",fs))
            ruta_sa_najvecim_H, ruta_sa_najvecim_H_ostatak_cvorova = ako_se_praklapaju_racunaj_h(ruta_sa_kojom_se_poredi,
                                                               potencijalne_rute,
                                                               index_u_matrici_rute_koja_se_poredi,
                                                               matrica_slotova)
           
            # ovde dodaj kapacitet provodnika U i nekako sredi indeksi_requstova_koji_su_ostali ako pretekne preko 
        
            H = ruta_sa_najvecim_H[1]
            if H > 0:
                #ako je vece od nule dodaj za transmisiju
                if kapacitet_transmitera + H < KAPACITET_TRANSMITERA:
                    kapacitet_transmitera += H
                    lista_odabranih_ruta_i_meta_informacija.append([ruta_sa_najvecim_H[0],
                                                                ("fs_vrednost",fs),
                                                                ("ostatak_cvorova",ruta_sa_najvecim_H_ostatak_cvorova),
                                                                ("index_iter",index)])
                    if ruta_sa_najvecim_H_ostatak_cvorova is not None  :          
                        broj_ruta_sa_ostatkom_cvorova += 1
                else:
                    indeksi_requstova_koji_su_ostali.append(indeksi_requstova_koji_su_uzeti[index])
            else:
                indeksi_requstova_koji_su_ostali.append(indeksi_requstova_koji_su_uzeti[index])
               
                # dodati se logiku koja ako postoje veci broj grupisanih requestova koji su duzi od requesta rute_koje_se_poredi
                # kada sortiras te duze requestove vidi da li da se onaj nizi u soritranom nizu ima slobodan index na poziciji kada se oni visi zavrse
                # 
    
    lista_odabranih_ruta_i_fs_vrednosti = proveri_da_li_se_ostaci_grupisanim_ruta_preklapaju(lista_odabranih_ruta_i_meta_informacija)
    
    return lista_odabranih_ruta_i_fs_vrednosti, indeksi_requstova_koji_su_ostali
       

def proveri_da_li_se_ostaci_grupisanim_ruta_preklapaju(broj_ruta_sa_ostatkom_cvorova,
                                                       lista_odabranih_ruta_i_meta_informacija):
    
    lista_odabranih_ruta_i_fs_vrednosti = []
    
    lista_odabranih_ruta_i_meta_informacija.sort(key= lambda x: len(x[0]))
    
    if broj_ruta_sa_ostatkom_cvorova > 1:
        for ruta_i_meta_info in lista_odabranih_ruta_i_meta_informacija:
            
        
        
    return lista_odabranih_ruta_i_fs_vrednosti





def uzmi_n_requsteova_sa_pocekta_pcele(pcela,Graph):
    n_broj_requestova = BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI
    # za svaki od 3 requesta pronadji redom koji requstovi iz ostatka pcele sadrze
    # zajednicki pocetni cvor 
    ostatak_requestova = pcela.rute[n_broj_requestova:]
    for i in range(n_broj_requestova):
        pocetni_request = pcela.rute[i]
        # nadji_requestove_sa_istim_pocetnim_cvorom_u_pceli
        requestove_sa_istim_pocetnim_cvorom, indeksi_requstova_koji_su_uzeti, indeksi_requstova_koji_su_ostali = nadji_requestove_sa_istim_pocetnim_cvorom_u_pceli(pocetni_request,
                                                                                                                                                                   ostatak_requestova)
        
        
        lista_odabranih_ruta_i_fs_vrednosti, indeksi_requstova_koji_su_ostali = trazi_sve_preklapajuce_rute_medju_requestovima_sa_istim_pocetnim_cvorom(indeksi_requstova_koji_su_ostali,
                                                                                                                                                        indeksi_requstova_koji_su_uzeti,
                                                                                                                                                        requestove_sa_istim_pocetnim_cvorom,
                                                                                                                                                        Graph,
                                                                                                                                                        pcela.matrica_slotova)
        ostatak_requestova = pcela.rute[indeksi_requstova_koji_su_ostali]
        
        pcela.matrica_slotova = popunjavanje_matrice_slotova(lista_odabranih_ruta_i_fs_vrednosti,
                                                             pcela.matrica_slotova,
                                                             Graph)
        
        
    pcela.rute = ostatak_requestova
    pcela.trenutni_fs = racunaj_sumu_trenutnog_fs(pcela.matrica_slotova)
        
    return pcela
    

def racunaj_sumu_trenutnog_fs(matrica_slotova):
    # +1 zato sto ako je vrednost '4_5': [0, 2] znaci da su popunjenji slotovi 0,1,2
    return sum(value[-1]+1 for value in matrica_slotova.values() if value[-1] != 0)
        
    
def napravi_matricu_slotova(Graph):
    matrica_slotova_as_dict = {"_".join([str(edge[0]+1),str(edge[1]+1)]):[0] for edge in Graph.edges}
    
    return matrica_slotova_as_dict


def popunjavanje_matrice_slotova(lista_odabranih_ruta_i_fs_vrednosti,matrica_slotova,Graph):
    ruta_za_porednjenje = lista_odabranih_ruta_i_fs_vrednosti[0]
    if len(ruta_za_porednjenje[0]) != 2:
        rute_za_grupisanje = lista_odabranih_ruta_i_fs_vrednosti[1:]
        rute_za_grupisanje_sortirane_od_najvece_ka_najmanjoj = sorted(rute_za_grupisanje,key = lambda x: len(x[0]), 
                                                                      reverse= True)
        
        pocetni_cvor, terminalni_cvor =  ruta_za_porednjenje[0]
    
        rute_koje_imaju_slobodne_slotove = izbaci_rute_koji_imaju_zauzete_linkove_koje_ruta_za_porednjenje_ne_sadrzi(matrica_slotova,
                                                                                                                     ruta_za_porednjenje,
                                                                                                                     rute_za_grupisanje_sortirane_od_najvece_ka_najmanjoj)
        rute_koje_imaju_slobodne_slotove.sort(key = lambda x: len(x[0]),reverse= True)
      
        
        zbir_svih_fs = sum(ruta[-1][1] for ruta in rute_koje_imaju_slobodne_slotove)
        rute_koje_imaju_slobodne_slotove_sa_zero_pading = copy.deepcopy(rute_koje_imaju_slobodne_slotove)
        
        duzina_najduze_rute = len(rute_koje_imaju_slobodne_slotove[0][0])
        for ruta in rute_koje_imaju_slobodne_slotove_sa_zero_pading:
            zero_pading(ruta[0],duzina_najduze_rute)
        
        for broj_koraka in range(1,duzina_najduze_rute):
            for ruta_i_fs in rute_koje_imaju_slobodne_slotove_sa_zero_pading:
                
                if broj_koraka == 1:
                    fs = ruta_i_fs[-1][1]
                    ruta = ruta_i_fs[0]
                    prvi_cvor = ruta[0]
                    drugi_cvor = ruta[1]
                    link = "_".join([str(prvi_cvor),str(drugi_cvor)])
                    zbirni_fs_po_linku = zbir_svih_fs 
                    
                else:
                    fs = ruta_i_fs[-1][1]
                    ruta = ruta_i_fs[0]
                    prvi_cvor = ruta[broj_koraka-1]
                    drugi_cvor = ruta[broj_koraka]
                    print(ruta, prvi_cvor, drugi_cvor)
                    if prvi_cvor != 0 and drugi_cvor != 0:
                        link = "_".join([str(prvi_cvor),str(drugi_cvor)])
                        zbirni_fs_po_linku += fs
                    
            zbirni_fs_po_linku_i_zastigni_opseg = ZASTITNI_OPSEG + zbirni_fs_po_linku + ZASTITNI_OPSEG - 1 # posto je zero based array
            matrica_slotova[link].append(matrica_slotova[link][-1] + zbirni_fs_po_linku_i_zastigni_opseg) 
            zbirni_fs_po_linku = 0
        
    else:
         pass   
        
    
    return matrica_slotova
    

    
def zero_pading(list_for_transformation,len_of_list_after_pading):
    list_for_transformation.extend([0] * (len_of_list_after_pading - len(list_for_transformation)))
    return None    

def racunaj_startu_poziciju_za_popunjavanje_matrice(rute, matrica_slotova_as_dict):
    # rute moraju da dodju sortirane od najmanje ka najvecoj
    # ruta koja ima majmanji index se uzima, posto K ograniceno nece uzimati previse ruta
    
    startna_pozicija_za_popunjavanje_matrice_slotova = None
    ruta_sa_najmanjim_indeksom = {}
    print("rute", rute)
    for ruta in rute:
        
        len_of_ruta = len(ruta)
        print("ruta", ruta, "len_of_ruta", len_of_ruta)
        for i in range(len_of_ruta):
            print("ruta", ruta, "len_of_ruta", len_of_ruta, "i", i )
            
            if i != len_of_ruta-1:
                link_izmedju_cvorova = "_".join([str(cvor) for cvor in ruta[i:i+2]])
                broj_slobodnog_slota_na_linku = matrica_slotova_as_dict[link_izmedju_cvorova][-1]

        if startna_pozicija_za_popunjavanje_matrice_slotova is None:
            startna_pozicija_za_popunjavanje_matrice_slotova = broj_slobodnog_slota_na_linku
            
            ruta_sa_najmanjim_indeksom["ruta"] = ruta
            ruta_sa_najmanjim_indeksom["index_u_matrici"] = startna_pozicija_za_popunjavanje_matrice_slotova 
        
        elif broj_slobodnog_slota_na_linku < startna_pozicija_za_popunjavanje_matrice_slotova:
            startna_pozicija_za_popunjavanje_matrice_slotova = broj_slobodnog_slota_na_linku
            
            ruta_sa_najmanjim_indeksom["ruta"] = ruta
            ruta_sa_najmanjim_indeksom["index_u_matrici"] = startna_pozicija_za_popunjavanje_matrice_slotova
            
                
    return ruta_sa_najmanjim_indeksom


def izbaci_rute_koji_imaju_zauzete_linkove_koje_ruta_za_porednjenje_ne_sadrzi(matrica_slotova,
                                                                              ruta_za_porednjenje,
                                                                              rute):
    

    startna_pozicija_za_popunjavanje_matrice_slotova = racunaj_startu_poziciju_za_popunjavanje_matrice([ruta_za_porednjenje], matrica_slotova)
    print("POSLE OVOG AAAAAAA")
    # posto me zanima broj slota bez zadnjeg granicnika ZASTITNI_OPSEG
    zbir_fs = ZASTITNI_OPSEG + startna_pozicija_za_popunjavanje_matrice_slotova + 1
    rute_koje_imaju_slobodne_slotove = []
    rute_koje_imaju_slobodne_slotove.append(ruta_za_porednjenje)
    if rute != []:
        for index,cvorovi_i_fs in enumerate(rute):
            zadrzi_rutu = True
            cvorovi, fs = cvorovi_i_fs
            len_cvorovi = len(cvorovi)
            for index_cvor,cvor in enumerate(cvorovi):
                if cvor not in ruta_za_porednjenje[0]:
                    
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
                        if matrica_slotova[link_izmedju_cvorova][-1] > zbir_fs:
                            zadrzi_rutu = False
                        
            if zadrzi_rutu is True:
               rute_koje_imaju_slobodne_slotove.append(cvorovi_i_fs) 
               zbir_fs += fs[1]
    
    return rute_koje_imaju_slobodne_slotove


    
def racunaj_verovatnoce(pcela_parcijalno_resenje,
                          fs_min,
                          fs_max,
                          random_rud,
                          ob_sum_recruter):
    
    fb = pcela_parcijalno_resenje.trenutni_fs
    ob = (fs_max-fb)/(fs_max-fs_min)
    pb_loyal = 1-math.log10((1+(1-ob)))
    
    is_follower = True if pb_loyal <= random_rud else False

    if is_follower is False:
        ob_sum_recruter += ob
    
    return ob_sum_recruter, {"pcela_parcijalno_resenje":pcela_parcijalno_resenje,
                                "ob":ob,
                                "pb_loyal":pb_loyal,
                                "is_follower":is_follower}
    
def poredi_pcele(pcele_parcijalna_resenja):
    
    random_rud = random.uniform(0, 1)

    pcela_parcijalno_resenje_sa_verovatnocama = []
    ob_sum_recruter = 0
    fs_min = min(key = lambda pcela: pcela.trenutni_fs)
    fs_max = max(key = lambda pcela: pcela.trenutni_fs)
    
    for pcela_parcijalno_resenje in pcele_parcijalna_resenja:
        ob_sum_recruter, pcela_sa_verovatnocama  = racunaj_verovatnoce(pcela_parcijalno_resenje,
                                                 fs_min,
                                                 fs_max,
                                                 random_rud,
                                                 ob_sum_recruter)
        pcela_parcijalno_resenje_sa_verovatnocama.append(pcela_sa_verovatnocama)
        

    lista_pcela_posle_follow_recruter_faze = regrutacija_pcela(pcela_parcijalno_resenje_sa_verovatnocama,
                                                               ob_sum_recruter)
    
    

def regrutacija_pcela(pcela_parcijalno_resenje_sa_verovatnocama,ob_sum_recruter):
    
    verovatnoca_regruteri = [(index_rekrutera ,pcela_i_verovatnoce["ob"]/ob_sum_recruter) 
                                  for index_rekrutera, pcela_i_verovatnoce in enumerate(pcela_parcijalno_resenje_sa_verovatnocama) 
                                      if pcela_i_verovatnoce["is_follower"] is False].sort(key= lambda x: x[1])
    
    lista_pcela_posle_follow_recruter_faze = []
    
    for pcela_i_verovatnoce in pcela_parcijalno_resenje_sa_verovatnocama:
        if pcela_i_verovatnoce["is_follower"] is True:
            random_broj_follower = random.uniform(0, 1)
            for verovatnoca_regruter in verovatnoca_regruteri:
                if random_broj_follower <= verovatnoca_regruter[1]:
                   kopija_pcele = copy.deepcopy(pcela_parcijalno_resenje_sa_verovatnocama[verovatnoca_regruter[0]])
                   lista_pcela_posle_follow_recruter_faze.append(kopija_pcele["pcela_parcijalno_resenje"])
                   
        else:
            lista_pcela_posle_follow_recruter_faze.append(pcela_i_verovatnoce["pcela_parcijalno_resenje"])
            
                    
    return lista_pcela_posle_follow_recruter_faze

        

def main():


    broj_pcela = 5
    dimenzija_kvadratne_matrice = 5
    min_slot = 1
    max_slot = 4

    # matrica_linkova, matrica_povezanosti = load_matrices_from_files_names("dummy1", "dummy2")
    matrica_linkova = napravi_matricu(1, 1, dimenzija_kvadratne_matrice, dijagola_nule=True)
    matrica_zahteva = napravi_matricu(min_slot, max_slot, dimenzija_kvadratne_matrice, dijagola_nule=True)
    matrica_povezanosti = napravi_matricu_povezanosti_od_matrice_zahteva(matrica_zahteva)
    
    Graph = nx.from_numpy_matrix(matrica_linkova, create_using=nx.MultiDiGraph())

    lista_pcela = pravljenje_liste_pcela_sa_izmesanim_rutama(broj_pcela, 
                                                             matrica_povezanosti,
                                                             Graph)
    
    for pcela in lista_pcela:
        uzmi_n_requsteova_sa_pocekta_pcele(pcela,Graph)



if __name__ == '__main__':
    main()




# rute_koje_imaju_slobodne_slotove_cuvanje