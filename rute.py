#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from numpy import genfromtxt
# import matplotlib.pyplot as plt
import networkx as nx
import math
import random
import copy
import time
# from gooey import Gooey

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# nonbuffered_stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
# sys.stdout = nonbuffered_stdout

ZASTITNI_OPSEG = 1
KAPACITET_TRANSMITERA = 8
BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI = 3
BROJ_RUTA_K = 3
BROJ_ITERACIJA = 3


# @Gooey(program_name="BCO", image_dir=".\image")
def parsing_argumts():
    global ZASTITNI_OPSEG, KAPACITET_TRANSMITERA, BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI, BROJ_RUTA_K, BROJ_ITERACIJA
    ZASTITNI_OPSEG = 1
    KAPACITET_TRANSMITERA = 8
    BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI = 3
    BROJ_RUTA_K = 3
    BROJ_ITERACIJA = 3

    parser = argparse.ArgumentParser(
        description='Algoritam za izracunavanja popunjenosti mreze slotova baziran na heuristici - optimizacija kolonijom pcela (BCO)')

    parser.add_argument('-m', '--matrica_linkova',
                        help='Ime fajla za matricu linkova u .csv file formatu.', required=True, type=str)

    parser.add_argument('-b', '--broj_pcela',
                        help='Broj pcela za simulaciju', required=True, type=int)

    parser.add_argument('-min', '--min_slot',
                        help='minimalni broj slotova prilikog generisanja matrice povezanosti.', required=True,
                        type=int)

    parser.add_argument('-max', '--max_slot',
                        help='maksimalan broj slotova prilikog generisanja matrice povezanosti.', required=True,
                        type=int)

    parser.add_argument('-u', '--kapacitet_trasmitera',
                        help='Broj slotova koje mogu da zauzmu rute prilikom grupisannja.',
                        default=KAPACITET_TRANSMITERA, type=int)

    parser.add_argument('-g', '--zastitni_opseg',
                        help='Broj slotova koji se dodaje kao granicnik sa obe strane grupisanih ruta.',
                        default=ZASTITNI_OPSEG, type=int)

    parser.add_argument('-nr', '--broj_req',
                        help='Broj uzetih requestova za razmatranje prilikom grupisanja.',
                        default=BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI, type=int)

    parser.add_argument('-k', '--broj_ruta',
                        help='Broj uzetih ruta za razmatranje prilikom grupisanja.', default=BROJ_RUTA_K, type=int)

    parser.add_argument('-i', '--broj_iteracija',
                        help='Broj iteracija', default=BROJ_ITERACIJA, type=int)

    parser.add_argument('-o', '--output',
                        help='Ime file gde se sacuvati tabela sa rezultatima', default="rezultat.csv", type=str)

    parser.add_argument('-mz', '--matrica_zahteva',
                        help='Matrica zahteva', default=None, type=str)

    parser.add_argument('-s', '--sort_req',
                        help='Sortiraj requestove od najmanje ka najvecoj vrednosti FS',
                        action='store_true')

    parser.add_argument('-rs', '--sort_req_reverse',
                        help='Sortiraj requestove od najvece ka najmanjoj vrednosti FS',
                        action='store_true')

    parser.add_argument('-kf', '--kriterijumska_funkcija',
                        help='Tip kriterijumske funkcije koja se koristi',
                        default='min_fs', const='min_fs', nargs='?', choices=['min_fs', 'max_c', 'min_t', 'min_t_norm'])

    parser.add_argument('-a1', '--parametarizacija_broj_transmitera',
                        help='Parametarizacija prlikom racuanjanja broja transmitera', default=0.5, type=int)

    parser.add_argument('-a2', '--parametarizacija_procenta_iskoriscenosti',
                        help='Parametarizacija prlikom racuanjanja procentaiskoriscenosti transmitera', default=0.5,
                        type=int)

    args = parser.parse_args()
    matrica_linkova_file_name = args.matrica_linkova
    broj_pcela = args.broj_pcela
    min_slot = args.min_slot
    max_slot = args.max_slot
    output = args.output
    matrica_zahteva = ucitaj_matricu_zahteva(args.matrica_zahteva)
    sort_req = args.sort_req
    sort_req_reverse = args.sort_req_reverse
    kriterijumska_funkcija = args.kriterijumska_funkcija
    parametarizacija_broj_transmitera = args.parametarizacija_broj_transmitera
    parametarizacija_procenta_iskoriscenosti = args.parametarizacija_procenta_iskoriscenosti

    KAPACITET_TRANSMITERA = args.kapacitet_trasmitera
    ZASTITNI_OPSEG = args.zastitni_opseg
    BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI = args.broj_req
    BROJ_RUTA_K = args.broj_ruta
    BROJ_ITERACIJA = args.broj_iteracija

    return (matrica_linkova_file_name,
            broj_pcela,
            min_slot,
            max_slot, output,
            matrica_zahteva,
            sort_req,
            sort_req_reverse,
            kriterijumska_funkcija,
            parametarizacija_broj_transmitera,
            parametarizacija_procenta_iskoriscenosti)


def ucitaj_matricu_zahteva(matrica_zahteva):
    if matrica_zahteva is None:
        return matrica_zahteva
    else:
        return load_matrices_from_files_names(matrica_zahteva)


def load_matrices_from_files_names(matrica_linkova_file_name):
    matrica_linkova = genfromtxt(matrica_linkova_file_name, delimiter=',')
    # convert to matrix values to int 
    matrica_linkova = matrica_linkova.astype(np.int)

    return matrica_linkova


def napravi_matricu(donja_granica,
                    gornja_granica,
                    dimenzija_kvadratne_matrice,
                    dijagola_nule=False):
    rows = dimenzija_kvadratne_matrice
    columns = dimenzija_kvadratne_matrice
    matrica = np.array([[random.randrange(donja_granica, gornja_granica + 1)
                         for x in range(columns)] for y in range(rows)])

    if dijagola_nule:
        np.fill_diagonal(matrica, 0)
    return matrica


def napravi_matricu_povezanosti_od_matrice_zahteva(matrica_zahteva):
    matrica_povezanosti = [[i + 1, position + 1, number]
                           for i, row in enumerate(matrica_zahteva)
                           for position, number in enumerate(row) if position != i]
    return matrica_povezanosti


# crtanje grapha

# def Nacrtaj_graph(Graph):
#
#     pos = nx.circular_layout(Graph)
#     nx.draw_circular(Graph)
#     labels = {i : i +1 for i in Graph.nodes()}
#     nx.draw_networkx_labels(Graph, pos, labels, font_size=15)
#     plt.show()

########################

def all_paths(Graph, pocetni_cvor, terminalni_cvor):
    # vraca sve moguce rute od pocetnog do terminalnog cvora za dati graph
    generator_paths = nx.all_simple_paths(Graph, pocetni_cvor - 1, terminalni_cvor - 1)
    paths = [list(map(lambda x: x + 1, path))
             for path in generator_paths]
    # list(nx.all_simple_paths(Graph, pocetni_cvor-1, terminalni_cvor-1))

    return paths


class Pcela:
    __slots__ = ['_rute',
                 '_trenutni_fs',
                 '_matrica_slotova',
                 '_ukupna_usteda',
                 '_kapacitet_iskoricenosti',
                 '_broj_transmitera',
                 '_max_broj_transmitera',
                 '_kapacitet_iskoricenosti_normalizovan']

    def __init__(self, rute, trenutni_fs, matrica_slotova):
        self._rute = rute
        self._trenutni_fs = trenutni_fs
        self._matrica_slotova = matrica_slotova
        self._ukupna_usteda = np.int64(0)
        self._kapacitet_iskoricenosti = 0
        self._broj_transmitera = 0
        self._max_broj_transmitera = len(rute)
        self._kapacitet_iskoricenosti_normalizovan = 0

    @property
    def rute(self):
        return self._rute

    @rute.setter
    def rute(self, value):
        self._rute = value

    @rute.deleter
    def rute(self):
        del self._rute

    @property
    def trenutni_fs(self):
        return self._trenutni_fs

    @trenutni_fs.setter
    def trenutni_fs(self, value):
        self._trenutni_fs = value

    @trenutni_fs.deleter
    def trenutni_fs(self):
        del self._trenutni_fs

    @property
    def matrica_slotova(self):
        return self._matrica_slotova

    @matrica_slotova.setter
    def matrica_slotova(self, value):
        self._matrica_slotova = value

    @matrica_slotova.deleter
    def matrica_slotova(self):
        del self._matrica_slotova

    @property
    def ukupna_usteda(self):
        return self._ukupna_usteda

    @ukupna_usteda.setter
    def ukupna_usteda(self, value):
        self._ukupna_usteda = value

    @ukupna_usteda.deleter
    def ukupna_usteda(self):
        del self._ukupna_usteda

    @property
    def kapacitet_iskoricenosti(self):
        return self._kapacitet_iskoricenosti

    @kapacitet_iskoricenosti.setter
    def kapacitet_iskoricenosti(self, value):
        self._kapacitet_iskoricenosti = value

    @kapacitet_iskoricenosti.deleter
    def kapacitet_iskoricenosti(self):
        del self._kapacitet_iskoricenosti

    @property
    def broj_transmitera(self):
        return self._broj_transmitera

    @broj_transmitera.setter
    def broj_transmitera(self, value):
        self._broj_transmitera = value

    @broj_transmitera.deleter
    def broj_transmitera(self):
        del self._broj_transmitera

    @property
    def max_broj_transmitera(self):
        return self._max_broj_transmitera

    @max_broj_transmitera.setter
    def max_broj_transmitera(self, value):
        self._max_broj_transmitera = value

    @max_broj_transmitera.deleter
    def max_broj_transmitera(self):
        del self._max_broj_transmitera

    @property
    def kapacitet_iskoricenosti_normalizovan(self):
        return self._kapacitet_iskoricenosti_normalizovan

    @kapacitet_iskoricenosti_normalizovan.setter
    def kapacitet_iskoricenosti_normalizovan(self, value):
        self._kapacitet_iskoricenosti_normalizovan = value

    @kapacitet_iskoricenosti_normalizovan.deleter
    def kapacitet_iskoricenosti_normalizovan(self):
        del self._kapacitet_iskoricenosti_normalizovan

    def __repr__(self):
        return f"""rute: {self._rute}
                  trenutni_fs: {self._trenutni_fs}
                  matrica_slotova: {self._matrica_slotova},
                  ukupna_usteda: {self.ukupna_usteda},
                  kapacitet_iskoricenosti: {self.kapacitet_iskoricenosti},
                  kapacitet_iskoricenosti: {self.kapacitet_iskoricenosti_normalizovan},
                  broj_transmitera: {self.broj_transmitera},
                  max_broj_transmitera:{self.max_broj_transmitera}"""


def pravljenje_liste_pcela_sa_izmesanim_rutama(broj_pcela,
                                               matrica_povezanosti,
                                               Graph,
                                               ordering_of_requests):
    # svaki red u matrici povezanosti je jenda ruta, sto znaci da je pcela matrica povezanosti -->samo suffle
    lista_pcela = []
    matrica_povezanosti_copy = np.copy(matrica_povezanosti)
    trenutni_fs = 0
    matrica_slotova = []
    matrica_slotova = napravi_matricu_slotova(Graph)

    ordering_of_requests_switch = {"sorted": lambda matrica: matrica[matrica[:, -1].argsort()],
                                   "sorted_reversed": lambda matrica: matrica[matrica[:, -1].argsort()[::-1]],
                                   "random": lambda matrica: matrica}

    for i in range(broj_pcela):
        np.random.shuffle(matrica_povezanosti_copy)  # permute rows
        matrica_povezanosti_shuffled = np.copy(matrica_povezanosti_copy)
        matrica_povezanosti_shuffled = ordering_of_requests_switch[ordering_of_requests](matrica_povezanosti_copy)

        #        print("matrica_povezanosti_shuffled", matrica_povezanosti_shuffled)
        pcela = Pcela(matrica_povezanosti_shuffled, trenutni_fs, matrica_slotova)
        lista_pcela.append(pcela)

    return lista_pcela


def nadji_requestove_sa_istim_pocetnim_cvorom_u_pceli(pocetni_request,
                                                      ostatak_requestova):
    requestove_sa_istim_pocetnim_cvorom = [pocetni_request]
    indeksi_requstova_koji_su_ostali = []
    indeksi_requstova_koji_su_uzeti = []
    pocetni_cvor = pocetni_request[0]
    for index, ostk_request in enumerate(ostatak_requestova):
        if ostk_request[0] == pocetni_cvor:
            requestove_sa_istim_pocetnim_cvorom.append(ostk_request)
            indeksi_requstova_koji_su_uzeti.append(BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI + index)
        else:
            indeksi_requstova_koji_su_ostali.append(BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI + index)

    return requestove_sa_istim_pocetnim_cvorom, indeksi_requstova_koji_su_uzeti, indeksi_requstova_koji_su_ostali


def Racunaj_H(G, X, Y, Z, W):
    # X broj zajednickih linkova
    # Y duzina rute koje se razmatra sa grupisanje
    # Z duzina prve izabrane rute
    # W broj slotova te rute -fs
    # G zastitni opseg
    H = 2 * G * X - (Y - Z) * W
    return H


def proveri_da_li_rute_sadrze_iste_cvorove(ruta_1, ruta_2, index, matrica_slotova):
    # [2,4,1,3] i [2,4,1,2] bice False, ovaj slucaj je izbacen jer ga je tesko isprogramirati
    len_ruta_1 = len(ruta_1)
    len_ruta_2 = len(ruta_2)
    ostatak_cvorova = None

    if len_ruta_2 > len_ruta_1:
        broj_cvorova_koji_su_ostatak = len_ruta_2 - len_ruta_1
        # -broj_cvorova_koji_su_ostatak+1 znaci uzmi sve cvorove koje se sadrze u ruti 2 i
        # poslednji zajednicki cvor
        ostatak_cvorova = ruta_2[-(broj_cvorova_koji_su_ostatak + 1):]
        preklapaju_se = True if ruta_1 == ruta_2[:-broj_cvorova_koji_su_ostatak] else False
        broj_zajednickih_linkova = len_ruta_1 - 1

    else:
        for par_cvor in zip(ruta_1, ruta_2):
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
    # uzmi prvu rutu i napravi dummy samo da bi moglo da se poredi
    predhodna_ruta_sa_H_vece_od_0 = (potencijalne_rute[0], 0)
    ruta_sa_najvecim_H_ostatak_cvorova = 0
    # do :-1 posto je fs zadnji tuple koji treba se preskoci
    for ruta in potencijalne_rute[:-1]:
        preklapaju_se, broj_zajednickih_linkova, ostatak_cvorova = proveri_da_li_rute_sadrze_iste_cvorove(
            ruta_sa_kojom_se_poredi_bez_fs,
            ruta,
            index_u_matrici_rute_koja_se_poredi,
            matrica_slotova)
        if preklapaju_se is True:
            G = ZASTITNI_OPSEG  # promeni ovo posle
            len_ruta = len(ruta)
            H = Racunaj_H(G,
                          broj_zajednickih_linkova,
                          len_ruta,
                          len_of_ruta_sa_kojom_se_poredi_bez_fs,
                          fs_potencijalne_rute)
            if H > 0:
                if predhodna_ruta_sa_H_vece_od_0[1] < H:
                    predhodna_ruta_sa_H_vece_od_0 = (ruta, H)
                    ruta_sa_najvecim_H_ostatak_cvorova = ostatak_cvorova
                elif predhodna_ruta_sa_H_vece_od_0[1] == H:
                    # biraj kracu rutu
                    if len(predhodna_ruta_sa_H_vece_od_0[0]) > len_ruta:
                        ruta_sa_najvecim_H_ostatak_cvorova = ostatak_cvorova
                        predhodna_ruta_sa_H_vece_od_0 = (ruta, H)

    ruta_sa_najvecim_H = predhodna_ruta_sa_H_vece_od_0

    return ruta_sa_najvecim_H, ruta_sa_najvecim_H_ostatak_cvorova


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
    rute = sorted(all_paths(Graph, pocetni_cvor, terminalni_cvor), key=lambda x: len(x))[:BROJ_RUTA_K]
    ruta_sa_najmanjim_indeksom = racunaj_startu_poziciju_za_popunjavanje_matrice(rute, matrica_slotova)
    ruta_sa_kojom_se_poredi = [ruta_sa_najmanjim_indeksom["ruta"], ("fs_vrednost", fs)]
    index_u_matrici_rute_koja_se_poredi = ruta_sa_najmanjim_indeksom["index_u_matrici"]
    index_u_matrici_rute_koja_se_poredi += (ZASTITNI_OPSEG + fs)
    ###   "index_u_matrici"
    lista_odabranih_ruta_i_meta_informacija.append(ruta_sa_kojom_se_poredi)

    if len(ruta_sa_kojom_se_poredi[0]) != 2:
        # ako je jednako tri znaci da je ruta_sa_kojom_se_poredi duzine 2 i da ne postoji grupisanje
        # npr [4, 5, ('fs_vrednost', 1)]

        for index, ruta in enumerate(requestove_sa_istim_pocetnim_cvorom[1:]):
            # print('ruta je ', ruta)
            pocetni_cvor, terminalni_cvor, fs = ruta
            # PROMENI OVO nemoj sorted nego sort ili cak sve izbaci da path kad vraca, vraca samo najkrace, ako moze
            potencijalne_rute = sorted(all_paths(Graph, pocetni_cvor, terminalni_cvor), key=lambda x: len(x))[
                                :BROJ_RUTA_K]
            # PROMENI OVO pravi samo K najkracih ruta u grafu, za pocetni i krajni cvor, manje ce biti komputaciono zahtevno

            potencijalne_rute.append(("fs_vrednost", fs))
            ruta_sa_najvecim_H, ruta_sa_najvecim_H_ostatak_cvorova = ako_se_praklapaju_racunaj_h(
                ruta_sa_kojom_se_poredi,
                potencijalne_rute,
                index_u_matrici_rute_koja_se_poredi,
                matrica_slotova)

            # ovde dodaj kapacitet provodnika U i nekako sredi indeksi_requstova_koji_su_ostali ako pretekne preko 

            H = ruta_sa_najvecim_H[1]
            if H > 0:
                # ako je vece od nule dodaj za transmisiju
                if kapacitet_transmitera + fs < KAPACITET_TRANSMITERA:
                    kapacitet_transmitera += fs
                    lista_odabranih_ruta_i_meta_informacija.append([ruta_sa_najvecim_H[0],
                                                                    ("fs_vrednost", fs),
                                                                    ("ostatak_cvorova",
                                                                     ruta_sa_najvecim_H_ostatak_cvorova),
                                                                    ("index_iter", index)])
                    if ruta_sa_najvecim_H_ostatak_cvorova is not None:
                        broj_ruta_sa_ostatkom_cvorova += 1
                else:
                    indeksi_requstova_koji_su_ostali.append(indeksi_requstova_koji_su_uzeti[index])
            else:
                indeksi_requstova_koji_su_ostali.append(indeksi_requstova_koji_su_uzeti[index])

                # dodati se logiku koja ako postoje veci broj grupisanih requestova koji su duzi od requesta rute_koje_se_poredi
                # kada sortiras te duze requestove vidi da li da se onaj nizi u soritranom nizu ima slobodan index na poziciji kada se oni visi zavrse

        lista_odabranih_ruta_i_fs_vrednosti, indeksi_requstova_koji_su_ostali = proveri_da_li_se_ostaci_grupisanim_ruta_preklapaju(
            lista_odabranih_ruta_i_meta_informacija,
            index_u_matrici_rute_koja_se_poredi,
            indeksi_requstova_koji_su_ostali,
            indeksi_requstova_koji_su_uzeti,
            matrica_slotova,
        )

    else:
        # vrati na pocetak index bez fs i granicnika kako bi funkcija popunjavanje_matrice_slotova radila    
        index_u_matrici_rute_koja_se_poredi = ruta_sa_najmanjim_indeksom["index_u_matrici"]
        indeksi_requstova_koji_su_ostali.extend(indeksi_requstova_koji_su_uzeti)
        lista_odabranih_ruta_i_fs_vrednosti = lista_odabranih_ruta_i_meta_informacija
        fs = ruta_sa_kojom_se_poredi[1][1]
        kapacitet_transmitera += fs

    if kapacitet_transmitera == 0:
        # ovo znaci je za sve rute H<0 pa zato se ne jedna nije grupisala pa uzmi samo fs od ruta_sa_kojom_se_poredi
        fs = ruta_sa_kojom_se_poredi[1][1]
        kapacitet_transmitera += fs

    kapacitet_iskoricenosti = kapacitet_transmitera / KAPACITET_TRANSMITERA

    broj_zauzetih_slotova_za_otvoreni_transmiter = kapacitet_transmitera

    return lista_odabranih_ruta_i_fs_vrednosti, indeksi_requstova_koji_su_ostali, index_u_matrici_rute_koja_se_poredi, kapacitet_iskoricenosti, broj_zauzetih_slotova_za_otvoreni_transmiter


def proveri_da_li_se_ostaci_grupisanim_ruta_preklapaju(lista_odabranih_ruta_i_meta_informacija,
                                                       index_u_matrici_rute_koja_se_poredi,
                                                       indeksi_requstova_koji_su_ostali,
                                                       indeksi_requstova_koji_su_uzeti,
                                                       matrica_slotova):
    lista_odabranih_ruta_i_fs_vrednosti = []
    index_u_matrici_rute_koja_se_poredi_dopunjen = copy.deepcopy(index_u_matrici_rute_koja_se_poredi)
    # izbaci rutu sa kojom se grupise 
    ruta_za_grupisanje = lista_odabranih_ruta_i_meta_informacija[0][0]
    fs_tuple_grupisanje = lista_odabranih_ruta_i_meta_informacija[0][1]
    lista_odabranih_ruta_i_fs_vrednosti.append([ruta_za_grupisanje, fs_tuple_grupisanje])
    lista_odabranih_ruta_i_meta_informacija = lista_odabranih_ruta_i_meta_informacija[1:]
    lista_odabranih_ruta_i_meta_informacija.sort(key=lambda x: len(x[0]), reverse=True)

    for ruta_i_meta_info in lista_odabranih_ruta_i_meta_informacija:
        ruta = ruta_i_meta_info[0]
        fs_tuple = ruta_i_meta_info[1]
        ostatak_cvorova = ruta_i_meta_info[2][1]
        if ostatak_cvorova is not None:
            index_slobodnog_slota = racunaj_startu_poziciju_za_popunjavanje_matrice([ostatak_cvorova], matrica_slotova)
            if index_slobodnog_slota["index_u_matrici"] <= index_u_matrici_rute_koja_se_poredi_dopunjen:
                lista_odabranih_ruta_i_fs_vrednosti.append([ruta, fs_tuple])
                index_u_matrici_rute_koja_se_poredi_dopunjen += fs_tuple[1]
            else:
                index_iter = ruta_i_meta_info[3][1]
                indeksi_requstova_koji_su_ostali.append(indeksi_requstova_koji_su_uzeti[index_iter])
        else:
            lista_odabranih_ruta_i_fs_vrednosti.append([ruta, fs_tuple])

    return lista_odabranih_ruta_i_fs_vrednosti, indeksi_requstova_koji_su_ostali


def uzmi_n_requsteova_sa_pocekta_pcele(pcela, Graph):
    n_broj_requestova = BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI
    # za svaki od 3 requesta pronadji redom koji requstovi iz ostatka pcele sadrze
    # zajednicki pocetni cvor 
    len_pcela_rute = len(pcela.rute)
    if len_pcela_rute < n_broj_requestova:
        n_broj_requestova = len_pcela_rute
    ostatak_requestova = pcela.rute[n_broj_requestova:]
    kapacitet_iskoricenosti_pri_jednom_grupisanju = []
    broj_zauzetih_slotova_za_otvoreni_transmitere = []
    for i in range(n_broj_requestova):
        pocetni_request = pcela.rute[i]
        # nadji_requestove_sa_istim_pocetnim_cvorom_u_pceli
        requestove_sa_istim_pocetnim_cvorom, indeksi_requstova_koji_su_uzeti, indeksi_requstova_koji_su_ostali = nadji_requestove_sa_istim_pocetnim_cvorom_u_pceli(
            pocetni_request,
            ostatak_requestova)

        #        print("requestove_sa_istim_pocetnim_cvorom",requestove_sa_istim_pocetnim_cvorom)
        lista_odabranih_ruta_i_fs_vrednosti, indeksi_requstova_koji_su_ostali, index_u_matrici_rute_koja_se_poredi, kapacitet_iskoricenosti, broj_zauzetih_slotova_za_otvoreni_transmiter = trazi_sve_preklapajuce_rute_medju_requestovima_sa_istim_pocetnim_cvorom(
            indeksi_requstova_koji_su_ostali,
            indeksi_requstova_koji_su_uzeti,
            requestove_sa_istim_pocetnim_cvorom,
            Graph,
            pcela.matrica_slotova)
        ostatak_requestova = pcela.rute[indeksi_requstova_koji_su_ostali]

        pcela.ukupna_usteda = popunjavanje_matrice_slotova(lista_odabranih_ruta_i_fs_vrednosti,
                                                           pcela.matrica_slotova,
                                                           pcela.ukupna_usteda,
                                                           Graph,
                                                           index_u_matrici_rute_koja_se_poredi)

        kapacitet_iskoricenosti_pri_jednom_grupisanju.append(kapacitet_iskoricenosti)
        broj_zauzetih_slotova_za_otvoreni_transmitere.append(broj_zauzetih_slotova_za_otvoreni_transmiter)
        max_broj_transmitera = pcela.max_broj_transmitera

    pcela.rute = ostatak_requestova
    pcela.trenutni_fs = racunaj_sumu_trenutnog_fs(pcela.matrica_slotova)

    procenat_iskoriscenosti = sum(kapacitet_iskoricenosti_pri_jednom_grupisanju) / n_broj_requestova
    ukupan_broj_zauzetih_slotova_za_sve_otvorene_trasnimtere = sum(broj_zauzetih_slotova_za_otvoreni_transmitere)
    broj_transmitera = n_broj_requestova
    pcela.broj_transmitera += broj_transmitera
    pcela.kapacitet_iskoricenosti += racunaj_iskoriscenost_transmitera_min_t(broj_transmitera,
                                                                             procenat_iskoriscenosti)
    pcela.kapacitet_iskoricenosti_normalizovan += racunaj_iskoriscenost_transmitera_min_t_normalizovan(broj_transmitera,
                                                                                                       max_broj_transmitera,
                                                                                                       ukupan_broj_zauzetih_slotova_za_sve_otvorene_trasnimtere)

    return pcela


def racunaj_iskoriscenost_transmitera_min_t(broj_transmitera,
                                            procenat_iskoriscenosti):
    x_1 = broj_transmitera
    x_2 = procenat_iskoriscenosti
    a1 = parametarizacija_broj_transmitera
    a2 = parametarizacija_procenta_iskoriscenosti
    min_T = a1 * x_1 + (1 - a2) * (1 / x_2)
    # if min_T > broj_transmitera:
    #     print(f"broj_transmitera:{broj_transmitera}"
    #           f"procenat_iskoriscenosti :{procenat_iskoriscenosti}"
    #           f"min_T :{min_T}")

    return min_T


def racunaj_iskoriscenost_transmitera_min_t_normalizovan(broj_transmitera,
                                                         max_broj_transmitera,
                                                         ukupan_broj_zauzetih_slotova_za_sve_otvorene_trasnimtere):
    x_1 = broj_transmitera
    b1 = max_broj_transmitera
    ukupan_broj_slobodnih_slotova_za_sve_otvorene_trasnimtere = (broj_transmitera * KAPACITET_TRANSMITERA) - ukupan_broj_zauzetih_slotova_za_sve_otvorene_trasnimtere
    x_2 = ukupan_broj_slobodnih_slotova_za_sve_otvorene_trasnimtere
    b2 = broj_transmitera * KAPACITET_TRANSMITERA
    a1 = parametarizacija_broj_transmitera
    a2 = parametarizacija_procenta_iskoriscenosti
    min_T = a1 * (x_1 / b1) + a2 * (x_2 / b2)
    # if min_T > broj_transmitera:
    #     print(f"broj_transmitera:{broj_transmitera}"
    #           f"procenat_iskoriscenosti :{procenat_iskoriscenosti}"
    #           f"min_T :{min_T}")

    return min_T


def racunaj_sumu_trenutnog_fs(matrica_slotova):
    # +1 zato sto ako je vrednost '4_5': [0, 2] znaci da su popunjenji slotovi 0,1,2
    return sum(value[-1] + 1 for value in matrica_slotova.values() if value[-1] != 0)


def napravi_matricu_slotova(Graph):
    matrica_slotova_as_dict = {"_".join([str(edge[0] + 1), str(edge[1] + 1)]): [0] for edge in Graph.edges}

    return matrica_slotova_as_dict


def racunaj_ustede_usled_grupisanja(broj_grupisanih_requestova_po_ruti):
    return 2 * ZASTITNI_OPSEG * (broj_grupisanih_requestova_po_ruti - 1)


def popunjavanje_matrice_slotova(lista_odabranih_ruta_i_fs_vrednosti,
                                 matrica_slotova,
                                 ukupna_usteda,
                                 Graph,
                                 index_u_matrici_rute_koja_se_poredi):
    #    print("lista_odabranih_ruta_i_fs_vrednosti",lista_odabranih_ruta_i_fs_vrednosti)
    rute_za_grupisanje = lista_odabranih_ruta_i_fs_vrednosti

    rute_koje_imaju_slobodne_slotove_sa_zero_pading = copy.deepcopy(rute_za_grupisanje)

    len_ruta_za_poredjenje = len(rute_za_grupisanje[0][0])
    zbir_svih_fs = sum(ruta[-1][1] for ruta in rute_za_grupisanje)
    try:
        len_najduze_rute_za_grupisanje = len(rute_za_grupisanje[1][0])
        duzina_najduze_rute = max(len_ruta_za_poredjenje, len_najduze_rute_za_grupisanje)
    except IndexError:
        duzina_najduze_rute = len_ruta_za_poredjenje

    for ruta in rute_koje_imaju_slobodne_slotove_sa_zero_pading:
        zero_pading(ruta[0], duzina_najduze_rute)

    ukupna_usteda = copy.deepcopy(ukupna_usteda)
    for broj_koraka in range(1, duzina_najduze_rute):
        broj_grupisanih_ruta_korak_jedan = 0
        broj_grupisanih_ruta_ostali_koraci = 0
        usteda_po_koraku = 0
        for ruta_i_fs in rute_koje_imaju_slobodne_slotove_sa_zero_pading:

            if broj_koraka == 1:
                fs = ruta_i_fs[-1][1]
                ruta = ruta_i_fs[0]
                prvi_cvor = ruta[0]
                drugi_cvor = ruta[1]
                link = "_".join([str(prvi_cvor), str(drugi_cvor)])
                zbirni_fs_po_linku = zbir_svih_fs
                broj_grupisanih_ruta_korak_jedan += 1
            #                print(broj_koraka)
            #                print(ruta, prvi_cvor, drugi_cvor, zbirni_fs_po_linku)

            else:
                fs = ruta_i_fs[-1][1]
                ruta = ruta_i_fs[0]
                prvi_cvor = ruta[broj_koraka - 1]
                drugi_cvor = ruta[broj_koraka]
                if prvi_cvor != 0 and drugi_cvor != 0:
                    link = "_".join([str(prvi_cvor), str(drugi_cvor)])
                    zbirni_fs_po_linku += fs
                    broj_grupisanih_ruta_ostali_koraci += 1
        #                print(broj_koraka)
        #                print(ruta, prvi_cvor, drugi_cvor,zbirni_fs_po_linku)

        #        print("broj_grupisanih_ruta_korak_jedan",broj_grupisanih_ruta_korak_jedan)
        #        print("broj_grupisanih_ruta_ostali_koraci",broj_grupisanih_ruta_ostali_koraci)
        usteda_po_koraku = racunaj_ustede_usled_grupisanja(broj_grupisanih_ruta_korak_jedan
                                                           + broj_grupisanih_ruta_ostali_koraci)

        #        print("usteda_po_koraku",usteda_po_koraku)

        ukupna_usteda += usteda_po_koraku

        #        print("ukupna_usteda",ukupna_usteda)

        zbirni_fs_po_linku_i_zastigni_opseg = ZASTITNI_OPSEG + zbirni_fs_po_linku + ZASTITNI_OPSEG - 1  # posto je zero based array
        matrica_slotova[link].append(index_u_matrici_rute_koja_se_poredi + zbirni_fs_po_linku_i_zastigni_opseg)
        # ako bi cuvao i rupe onda bi ovde trebalo da ide index_u_matrici_rute_koja_se_poredi - matrica_slotova[link][-1] 
        zbirni_fs_po_linku = 0

    return ukupna_usteda


def zero_pading(list_for_transformation, len_of_list_after_pading):
    list_for_transformation.extend([0] * (len_of_list_after_pading - len(list_for_transformation)))
    return None


def racunaj_startu_poziciju_za_popunjavanje_matrice(rute, matrica_slotova_as_dict):
    # rute moraju da dodju sortirane od najmanje ka najvecoj
    # ruta koja ima majmanji index se uzima, posto K ograniceno nece uzimati previse ruta

    startna_pozicija_za_popunjavanje_matrice_slotova = None
    ruta_sa_najmanjim_indeksom = {}
    #    print("rute", rute)
    for ruta in rute:

        len_of_ruta = len(ruta)
        #        print("ruta", ruta, "len_of_ruta", len_of_ruta)
        for i in range(len_of_ruta):
            #            print("ruta", ruta, "len_of_ruta", len_of_ruta, "i", i )

            if i != len_of_ruta - 1:
                link_izmedju_cvorova = "_".join([str(cvor) for cvor in ruta[i:i + 2]])
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


def racunaj_verovatnoce_min_fs(pcela,
                               fs_min,
                               fs_max,
                               random_rud,
                               ob_sum_recruter):
    fb = pcela.trenutni_fs
    ob = (fs_max - fb) / (fs_max - fs_min)
    pb_loyal = 1 - math.log10((1 + (1 - ob)))

    is_follower = True if pb_loyal <= random_rud else False

    if is_follower is False:
        ob_sum_recruter += ob

    return ob_sum_recruter, {"pcela": pcela,
                             "ob": ob,
                             "pb_loyal": pb_loyal,
                             "is_follower": is_follower}


def vrati_random_broj():
    return random.uniform(0, 1)


def poredi_pcele_min_fs(lista_pcela):
    random_rud = vrati_random_broj()

    pcela_parcijalno_resenje_sa_verovatnocama = []
    ob_sum_recruter = 0
    fs_min = min(lista_pcela, key=lambda pcela: pcela.trenutni_fs).trenutni_fs
    fs_max = max(lista_pcela, key=lambda pcela: pcela.trenutni_fs).trenutni_fs
    if fs_min == fs_max:  # znaci da su sve pcele identicne
        lista_pcela = [lista_pcela[0]]
    # print ("fs_max",fs_max,"fs_min", fs_min)
    for pcela in lista_pcela:
        ob_sum_recruter, pcela_sa_verovatnocama = racunaj_verovatnoce_min_fs(pcela,
                                                                             fs_min,
                                                                             fs_max,
                                                                             random_rud,
                                                                             ob_sum_recruter)
        pcela_parcijalno_resenje_sa_verovatnocama.append(pcela_sa_verovatnocama)

    lista_pcela_posle_follow_recruter_faze = regrutacija_pcela(pcela_parcijalno_resenje_sa_verovatnocama,
                                                               ob_sum_recruter)

    return lista_pcela_posle_follow_recruter_faze


def racunaj_verovatnoce_max_c(pcela,
                              ukupna_usteda_min,
                              ukupna_usteda_max,
                              random_rud,
                              ob_sum_recruter):
    fb = pcela.ukupna_usteda
    ob = (abs(fb - ukupna_usteda_max)) / (ukupna_usteda_max - ukupna_usteda_min)
    pb_loyal = 1 - math.log10((1 + (1 - ob)))

    is_follower = True if pb_loyal <= random_rud else False

    if is_follower is False:
        ob_sum_recruter += ob

    return ob_sum_recruter, {"pcela": pcela,
                             "ob": ob,
                             "pb_loyal": pb_loyal,
                             "is_follower": is_follower}


def poredi_pcele_max_ukupna_usteda(lista_pcela):
    random_rud = vrati_random_broj()

    pcela_parcijalno_resenje_sa_verovatnocama = []
    ob_sum_recruter = 0
    ukupna_usteda_min = min(lista_pcela, key=lambda pcela: pcela.ukupna_usteda).ukupna_usteda
    ukupna_usteda_max = max(lista_pcela, key=lambda pcela: pcela.ukupna_usteda).ukupna_usteda
    if ukupna_usteda_min == ukupna_usteda_max:  # znaci da su sve pcele identicne
        lista_pcela = [lista_pcela[0]]
    # print ("ukupna_usteda_max",ukupna_usteda_max,"ukupna_usteda_min", ukupna_usteda_min)
    for pcela in lista_pcela:
        ob_sum_recruter, pcela_sa_verovatnocama = racunaj_verovatnoce_max_c(pcela,
                                                                            ukupna_usteda_min,
                                                                            ukupna_usteda_max,
                                                                            random_rud,
                                                                            ob_sum_recruter)
        pcela_parcijalno_resenje_sa_verovatnocama.append(pcela_sa_verovatnocama)

    # print ("pcela_parcijalno_resenje_sa_verovatnocama",pcela_parcijalno_resenje_sa_verovatnocama, "ob_sum_recruter",ob_sum_recruter)
    lista_pcela_posle_follow_recruter_faze = regrutacija_pcela(pcela_parcijalno_resenje_sa_verovatnocama,
                                                               ob_sum_recruter)

    return lista_pcela_posle_follow_recruter_faze


def racunaj_verovatnoce_min_t(pcela,
                              kapacitet_iskoricenosti_min,
                              kapacitet_iskoricenosti_max,
                              random_rud,
                              ob_sum_recruter):
    fb = pcela.kapacitet_iskoricenosti
    ob = (kapacitet_iskoricenosti_max - fb) / (kapacitet_iskoricenosti_max - kapacitet_iskoricenosti_min)
    pb_loyal = 1 - math.log10((1 + (1 - ob)))

    is_follower = True if pb_loyal <= random_rud else False

    if is_follower is False:
        ob_sum_recruter += ob

    return ob_sum_recruter, {"pcela": pcela,
                             "ob": ob,
                             "pb_loyal": pb_loyal,
                             "is_follower": is_follower}


def poredi_pcele_min_t(lista_pcela):
    random_rud = vrati_random_broj()

    pcela_parcijalno_resenje_sa_verovatnocama = []
    ob_sum_recruter = 0
    kapacitet_iskoricenosti_min = min(lista_pcela,
                                      key=lambda pcela: pcela.kapacitet_iskoricenosti).kapacitet_iskoricenosti
    kapacitet_iskoricenosti_max = max(lista_pcela,
                                      key=lambda pcela: pcela.kapacitet_iskoricenosti).kapacitet_iskoricenosti
    if kapacitet_iskoricenosti_min == kapacitet_iskoricenosti_max:  # znaci da su sve pcele identicne
        lista_pcela = [lista_pcela[0]]
    # print ("kapacitet_iskoricenosti_min",kapacitet_iskoricenosti_min,"kapacitet_iskoricenosti_max", kapacitet_iskoricenosti_max)
    for pcela in lista_pcela:
        ob_sum_recruter, pcela_sa_verovatnocama = racunaj_verovatnoce_min_t(pcela,
                                                                            kapacitet_iskoricenosti_min,
                                                                            kapacitet_iskoricenosti_max,
                                                                            random_rud,
                                                                            ob_sum_recruter)
        pcela_parcijalno_resenje_sa_verovatnocama.append(pcela_sa_verovatnocama)

    # print ("pcela_parcijalno_resenje_sa_verovatnocama",pcela_parcijalno_resenje_sa_verovatnocama, "ob_sum_recruter",ob_sum_recruter)
    lista_pcela_posle_follow_recruter_faze = regrutacija_pcela(pcela_parcijalno_resenje_sa_verovatnocama,
                                                               ob_sum_recruter)

    return lista_pcela_posle_follow_recruter_faze


def racunaj_verovatnoce_min_t_normalizovan(pcela,
                                           kapacitet_iskoricenosti_normalizovan_min,
                                           kapacitet_iskoricenosti_normalizovan_max,
                                           random_rud,
                                           ob_sum_recruter):
    fb = pcela.kapacitet_iskoricenosti_normalizovan
    ob = (kapacitet_iskoricenosti_normalizovan_max - fb) / (
            kapacitet_iskoricenosti_normalizovan_max - kapacitet_iskoricenosti_normalizovan_min)
    pb_loyal = 1 - math.log10((1 + (1 - ob)))

    is_follower = True if pb_loyal <= random_rud else False

    if is_follower is False:
        ob_sum_recruter += ob

    return ob_sum_recruter, {"pcela": pcela,
                             "ob": ob,
                             "pb_loyal": pb_loyal,
                             "is_follower": is_follower}


def poredi_pcele_min_t_normalizovan(lista_pcela):
    random_rud = vrati_random_broj()

    pcela_parcijalno_resenje_sa_verovatnocama = []
    ob_sum_recruter = 0
    kapacitet_iskoricenosti_normalizovan_min = min(lista_pcela,
                                                   key=lambda
                                                       pcela: pcela.kapacitet_iskoricenosti_normalizovan).kapacitet_iskoricenosti_normalizovan
    kapacitet_iskoricenosti_normalizovan_max = max(lista_pcela,
                                                   key=lambda
                                                       pcela: pcela.kapacitet_iskoricenosti_normalizovan).kapacitet_iskoricenosti_normalizovan
    if kapacitet_iskoricenosti_normalizovan_min == kapacitet_iskoricenosti_normalizovan_max:  # znaci da su sve pcele identicne
        lista_pcela = [lista_pcela[0]]
    # print ("kapacitet_iskoricenosti_min",kapacitet_iskoricenosti_min,"kapacitet_iskoricenosti_max", kapacitet_iskoricenosti_max)
    for pcela in lista_pcela:
        ob_sum_recruter, pcela_sa_verovatnocama = racunaj_verovatnoce_min_t_normalizovan(pcela,
                                                                                         kapacitet_iskoricenosti_normalizovan_min,
                                                                                         kapacitet_iskoricenosti_normalizovan_max,
                                                                                         random_rud,
                                                                                         ob_sum_recruter)
        pcela_parcijalno_resenje_sa_verovatnocama.append(pcela_sa_verovatnocama)

    # print ("pcela_parcijalno_resenje_sa_verovatnocama",pcela_parcijalno_resenje_sa_verovatnocama, "ob_sum_recruter",ob_sum_recruter)
    lista_pcela_posle_follow_recruter_faze = regrutacija_pcela(pcela_parcijalno_resenje_sa_verovatnocama,
                                                               ob_sum_recruter)

    return lista_pcela_posle_follow_recruter_faze


def regrutacija_pcela(pcela_parcijalno_resenje_sa_verovatnocama, ob_sum_recruter):
    verovatnoca_regruteri = [(index_rekrutera, pcela_i_verovatnoce["ob"] / ob_sum_recruter)
                             for index_rekrutera, pcela_i_verovatnoce in
                             enumerate(pcela_parcijalno_resenje_sa_verovatnocama)
                             if pcela_i_verovatnoce["is_follower"] is False]

    verovatnoca_regruteri.sort(key=lambda x: x[1])

    lista_pcela_posle_follow_recruter_faze = []

    for pcela_i_verovatnoce in pcela_parcijalno_resenje_sa_verovatnocama:
        if pcela_i_verovatnoce["is_follower"] is True:
            random_broj_follower = random.uniform(0, 1)
            for verovatnoca_regruter in verovatnoca_regruteri:
                if random_broj_follower <= verovatnoca_regruter[1]:
                    kopija_pcele = copy.deepcopy(pcela_parcijalno_resenje_sa_verovatnocama[verovatnoca_regruter[0]])
                    lista_pcela_posle_follow_recruter_faze.append(kopija_pcele["pcela"])

        else:
            lista_pcela_posle_follow_recruter_faze.append(pcela_i_verovatnoce["pcela"])

    return lista_pcela_posle_follow_recruter_faze


def inicijalizacija(broj_pcela,
                    min_slot,
                    max_slot,
                    matrica_linkova,
                    matrica_zahteva,
                    sort_req,
                    sort_req_reverse):
    # matrica_linkova, matrica_povezanosti = load_matrices_from_files_names("dummy1", "dummy2")
    dimenzija_kvadratne_matrice = matrica_linkova.shape[0]

    if matrica_zahteva is None:
        matrica_zahteva = napravi_matricu(min_slot, max_slot, dimenzija_kvadratne_matrice, dijagola_nule=True)

    ordering_of_requests = "random"
    if sort_req is True:
        ordering_of_requests = "sorted"
    if sort_req_reverse is True:
        ordering_of_requests = "sorted_reversed"

    matrica_povezanosti = napravi_matricu_povezanosti_od_matrice_zahteva(matrica_zahteva)

    Graph = nx.from_numpy_matrix(matrica_linkova, create_using=nx.MultiDiGraph())
    #    Nacrtaj_graph(Graph)
    lista_pcela = pravljenje_liste_pcela_sa_izmesanim_rutama(broj_pcela,
                                                             matrica_povezanosti,
                                                             Graph,
                                                             ordering_of_requests)

    return Graph, lista_pcela, matrica_zahteva


# proveri kako da uradis deep copy klase :) mozes damo da je iniciras, tako sto pokupis property-ije ali moze isto da pogledas __deepcopy__
def jedno_grupisanje(lista_pcela, Graph, kriterijumska_funkcija):
    for pcela in lista_pcela:
        uzmi_n_requsteova_sa_pocekta_pcele(pcela, Graph)

    if kriterijumska_funkcija == "min_fs":
        lista_pcela_posle_follow_recruter_faze = poredi_pcele_min_fs(lista_pcela)
    elif kriterijumska_funkcija == "max_c":
        lista_pcela_posle_follow_recruter_faze = poredi_pcele_max_ukupna_usteda(lista_pcela)
    elif kriterijumska_funkcija == "min_t":
        lista_pcela_posle_follow_recruter_faze = poredi_pcele_min_t(lista_pcela)
    elif kriterijumska_funkcija == "min_t_norm":
        lista_pcela_posle_follow_recruter_faze = poredi_pcele_min_t_normalizovan(lista_pcela)

    return lista_pcela_posle_follow_recruter_faze


def stop_kriterijum(lista_pcela):
    # print(*(pcela.rute.size for pcela in lista_pcela))
    return all(True if pcela.rute.size == 0 else False for pcela in lista_pcela)


def napravi_tabelu(sve_vrednosti, file_name):
    delimator = ","
    len_sve_vrednosti = len(sve_vrednosti)
    ukupan_fs_srednja_vrednost = sum(
        el["ukupan_fs"] for el in sve_vrednosti if el.get("ukupan_fs", False)) / len_sve_vrednosti
    ukupna_usteda_srednja_vrednost = sum(
        el["ukupna_usteda"] for el in sve_vrednosti if el.get("ukupna_usteda", False)) / len_sve_vrednosti
    ukupan_fs_bez_ustede_srednja_vrednost = sum(
        el["ukupan_fs_bez_ustede"] for el in sve_vrednosti if el.get("ukupan_fs_bez_ustede", False)) / len_sve_vrednosti
    vreme_egzekucije_srednja_vrednost = sum(
        el["vreme_egzekucije"] for el in sve_vrednosti if el.get("vreme_egzekucije", False)) / len_sve_vrednosti

    output_tabela = []
    header = delimator.join(list(sve_vrednosti[0].keys()))
    output_tabela.append(header)

    lista_redova = []
    for el in sve_vrednosti:
        red = delimator.join([str(e) for e in el.values()])
        lista_redova.append(red)
    output_tabela.extend(lista_redova)

    footer = delimator.join(["ukupan_fs_srednja_vrednost",
                             "ukupna_usteda_srednja_vrednost",
                             "ukupan_fs_bez_ustede_srednja_vrednost",
                             "vreme_egzekucije_srednja_vrednost"])
    footer_values = delimator.join((str(el) for el in [ukupan_fs_srednja_vrednost,
                                                       ukupna_usteda_srednja_vrednost,
                                                       ukupan_fs_bez_ustede_srednja_vrednost,
                                                       vreme_egzekucije_srednja_vrednost]))
    output_tabela.append(footer)
    output_tabela.append(footer_values)

    with open(file_name, "w") as f:
        print(*output_tabela, sep="\n", file=f)


def napravi_matrica_zahteva_file(matrica_zahteva,
                                 file_name):
    with open(file_name, "wt") as file:
        print(*(", ".join([str(el) for el in list(row)]) for row in matrica_zahteva),
              sep='\n', file=file)


def parsiraj_matrica_slotova(matrica_slotova):
    list_of_lines = []
    for key, value in matrica_slotova.items():
        list_of_lines.append(key.replace("_", "-") + "\t" + str(int(value[-1]) + 1))

    return "\n".join(list_of_lines)


def main():
    global parametarizacija_broj_transmitera, parametarizacija_procenta_iskoriscenosti

    matrica_linkova_file_name, broj_pcela, min_slot, max_slot, file_name, matrica_zahteva, sort_req, sort_req_reverse, kriterijumska_funkcija, parametarizacija_broj_transmitera, parametarizacija_procenta_iskoriscenosti = parsing_argumts()
    matrica_linkova = load_matrices_from_files_names(matrica_linkova_file_name)

    Graph, lista_pcela_pocenta, matrica_zahteva = inicijalizacija(broj_pcela,
                                                                  min_slot,
                                                                  max_slot,
                                                                  matrica_linkova,
                                                                  matrica_zahteva,
                                                                  sort_req,
                                                                  sort_req_reverse)

    print("PARAMETRI:",
          f"matrica linkova ime fajla: '{matrica_linkova_file_name}'",
          f"broj pcela: '{broj_pcela}'",
          f"min slot: '{min_slot}'",
          f"max slot: '{max_slot}'",
          f"zastitni opseg: '{ZASTITNI_OPSEG}'",
          f"kapacitet transmitera: '{KAPACITET_TRANSMITERA}'",
          f"broj requestova za razmatranje po iteraciji po pceli: '{BROJ_REQUESTOVA_ZA_RAZMATRANJE_PO_ITERACIJI_PO_PCELI}'",
          f"broj ruta k: '{BROJ_RUTA_K}'",
          f"broj iteracija: '{BROJ_ITERACIJA}'",
          f"ime output file: '{file_name}'",
          f"kriterijumska_funkcija: '{kriterijumska_funkcija}'",
          f"matrica zahteva: \n'{matrica_zahteva}'",
          f"parametarizacija_broj_transmitera : \n'{parametarizacija_broj_transmitera}'",
          f"parametarizacija_procenta_iskoriscenosti : \n'{parametarizacija_procenta_iskoriscenosti}'",
          sep="\n")

    print("\n")

    print("Initialisation is finished")
    print("\n")

    start_ukupno = time.time()
    sve_vrednosti = []
    for i in range(BROJ_ITERACIJA):
        start = time.time()
        m = 0
        print(f"The iteration {i+1}. started")
        lista_pcela = copy.deepcopy(lista_pcela_pocenta)
        while not stop_kriterijum(lista_pcela):
            lista_pcela = jedno_grupisanje(lista_pcela,
                                           Graph,
                                           kriterijumska_funkcija)
            m += 1
        end = time.time()

        vreme_egzekucije = end - start
        print(f"Time of execution per iteration:{vreme_egzekucije}")
        if kriterijumska_funkcija == "min_fs":
            najbolja_pcela = min(lista_pcela, key=lambda x: x.trenutni_fs)
        elif kriterijumska_funkcija == "max_c":
            najbolja_pcela = max(lista_pcela, key=lambda x: x.ukupna_usteda)
        elif kriterijumska_funkcija == "min_t":
            najbolja_pcela = min(lista_pcela, key=lambda x: x.kapacitet_iskoricenosti)
        elif kriterijumska_funkcija == "min_t_norm":
            najbolja_pcela = min(lista_pcela, key=lambda x: x.kapacitet_iskoricenosti_normalizovan)
        trenutni_fs = najbolja_pcela.trenutni_fs
        ukupna_usteda = najbolja_pcela.ukupna_usteda
        matrica_slotova = najbolja_pcela.matrica_slotova
        kapacitet_iskoricenosti = najbolja_pcela.kapacitet_iskoricenosti
        max_broj_transmitera = najbolja_pcela.max_broj_transmitera
        broj_transmitera = najbolja_pcela.broj_transmitera
        kapacitet_iskoricenosti_normalizovan = najbolja_pcela.kapacitet_iskoricenosti_normalizovan

        ukupan_fs_bez_ustede = trenutni_fs + ukupna_usteda
        sve_vrednosti.append({"ukupan_fs": trenutni_fs,
                              "ukupan_fs_bez_ustede": ukupan_fs_bez_ustede,
                              f"ukupna usteda : {ukupna_usteda}"
                              "kapacitet_iskoricenosti": kapacitet_iskoricenosti,
                              "kapacitet_iskoricenosti_normalizovan": kapacitet_iskoricenosti_normalizovan,
                              "broj_transmitera": broj_transmitera,
                              "max_broj_transmitera": max_broj_transmitera,
                              "vreme_egzekucije": vreme_egzekucije
                              })
        print(*[f"ukupan fs sa ustedom: {trenutni_fs}",
                f"ukupan fs bez ustede : {ukupan_fs_bez_ustede}",
                f"ukupna usteda : {ukupna_usteda}",
                f"broj transmitera: {broj_transmitera}",
                f"maksinalan broj transmitera: {max_broj_transmitera}",
                f"kapacitet iskoriscenosti transmitera: {kapacitet_iskoricenosti}",
                f"kapacitet iskoriscenosti transmitera normalizovan: {kapacitet_iskoricenosti_normalizovan}",
                parsiraj_matrica_slotova(matrica_slotova)], sep="\n")
        print(f"The iteration {i+1}. is finished")
        print("\n\n")
    end_ukupno = time.time()
    print(f"Ukupno izvrsavanja je:{end_ukupno - start_ukupno}")

    napravi_tabelu(sve_vrednosti, file_name)
    file_name = f"{min_slot}_{max_slot}_matrica_zahteva.csv"
    napravi_matrica_zahteva_file(matrica_zahteva, file_name)


if __name__ == '__main__':
    main()
