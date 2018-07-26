import wfdb
import math
import ntpath
import difflib
import os
from PIL import Image
import cv2
import numpy as np
from findtools.find_files import (find_files, Match)
import matplotlib.pyplot as plt
import sys

#originfiledir="mitdb/" 
#originfiledir="aftdb/" error 
originfiledir="ltafdb/" 
#originfiledir="edb/" 
#originfiledir="ltstdb/" #ST-T

txt_files_pattern = Match(filetype = 'f', name = '*.dat')
found_files = find_files(path=originfiledir, match=txt_files_pattern)

ptucount = 0

symbol_dict = dict()

printout = False

def dbprint(line):
    if printout:
        print(line)


def translate(symbol):
#    if symbol == 'p' or symbol == 'u' or symbol == 't':
#        ptucount += 1
    if symbol == '(NOD':
        dbprint("---------------Nodal (A-V junctional) rhythm")
    else:
        if '(N' in symbol:
            dbprint("---------------Normal sinus rhythm   {}".format(symbol))

    if symbol == 'N': 
        dbprint("---------------Normal beat")
    elif symbol == 'L':
        dbprint("---------------Left bundle branch block beat")
    elif symbol == 'R':
        dbprint("---------------Right bundle branch block beat")
    elif symbol == 'A':
        dbprint("---------------Atrial premature beat")
    elif symbol == 'a':
        dbprint("---------------Aberrated atrial premature beat")
    elif symbol == 'J':
        dbprint("---------------Nodal (junctional) premature beat ")
    elif symbol == 'S':
        dbprint("---------------Supraventricular premature or ectopic beat (atrial or nodal)")
    elif symbol == 'V':
        dbprint("---------------Premature ventricular contraction")
    elif symbol == 'F':
        dbprint("---------------Fusion of ventricular and normal beat")
    elif symbol == '!':
        dbprint("---------------Ventricular flutter wave") 
    elif symbol == 'e':
        dbprint("---------------Atrial escape beat")
    elif symbol == 'j':
        dbprint("---------------Nodal (junctional) escape beat")
    elif symbol == 'E':
        dbprint("---------------Ventricular escape beat")
    elif symbol == '/':
        dbprint("---------------Paced beat")
    elif symbol == 'f':
        dbprint("---------------Fusion of paced and normal beat")
    elif symbol == 'x':
        dbprint("---------------Non-conducted P-wave (blocked APB)")
    elif symbol == 'Q':
        dbprint("---------------Non-conducted P-wave (blocked APB)")
    elif symbol == '|':
        dbprint("---------------Isolated QRS-like artifact ")
 

    elif symbol == '(AB':
        dbprint("---------------Atrial bigeminy")
    elif symbol == '(AFIB':
        dbprint("---------------Atrial fibrillation")
    elif symbol == '(AFL':
        dbprint("---------------Atrial flutter")
    elif symbol == '(B':
        print("---------------Ventricular bigeminy")
    elif symbol == 'n':
        dbprint("---------------Supraventricular escape beat (atrial or nodal)")


    elif symbol == '"':
        print("---------------see comment")
    elif symbol == '(PREX':
        dbprint("---------------Pre-excitation (WPW)")
    elif symbol == '(BII':
        dbprint("---------------2° heart block")
    elif symbol == '(SBR':
        dbprint("---------------Sinus bradycardia")
    elif symbol == 'T':
        dbprint("---------------T-wave change") 
    elif symbol == '(T':
        dbprint("---------------Ventricular trigeminy")
    elif symbol == '(VT':
        dbprint("---------------Ventricular tachycardia")
    elif symbol == 't':
        dbprint("---------------Peak of t-wave!")
    elif symbol == 's':
        dbprint("---------------ST segment change")
    elif symbol == 'NOTE-at':
        print("---------------T deviation resulting from axis shift")
    elif symbol == 'NOTE-TS':
        print("---------------Tape slippage")
    elif symbol == 'NOTE-ast':
        print("---------------ST deviation resulting from axis shift")
    else:
        print("***********************************************Add more: {}************".format(symbol))
   
    if symbol in symbol_dict:
        symbol_dict[symbol] += 1
    else:
        symbol_dict[symbol] = 1 


def peakfinder(sig):
    max = sig[0]
    maxloc = -1
    for i in range(len(sig)):
        if sig[i] > max:
            max = sig[i]
            maxloc = i
            
    print("peak value: {}, peak loc: {}".format(max, maxloc))
    return maxloc


def savefig(readdir, loc, sym):
    print("checkpoint")
    halfoffset = 100
    start = max(loc - halfoffset, 0)
    end = loc + halfoffset
    record = wfdb.rdsamp(readdir,  sampfrom = start, sampto = end)
    peakloc = peakfinder(record.p_signals[:,0])
    loc = peakloc - halfoffset + loc
    start = max(loc - halfoffset, 0)
    end = loc + halfoffset
    record = wfdb.rdsamp(readdir,  sampfrom = start, sampto = end)
                 

    plt.plot(record.p_signals[:,0])
    plt.axis('off')
    plt.ylim(-2, 2.5) 
    if not os.path.isdir(databasedir):
        print(os.mkdir(databasedir))
    if not os.path.isdir(os.path.join(databasedir, sym)):
        print (os.mkdir(os.path.join(databasedir, sym)))
              
    figuresavepath = '{}{}{}{}{}{}{}'.format(databasedir,"/", sym, "/", recordname,  loc, ".png") 
    plt.savefig(figuresavepath)
    plt.clf() 
    #plotcount += 1



recordcount = 0
plotcount = 0
#endofsample = 100
#endofsample = 'end'
databasedir = "ecgdatabasedir"

i = 0
for found_file in found_files:
    head, tail = ntpath.split(found_file)
    recordname = tail.split('.')[0]
    #print (tail)
    print (recordname)
##############################
    recordcount += 1
    #if recordcount != target:
    #    continue
        
    readdir = head + '/' + recordname

    record = wfdb.rdsamp(readdir)
    #annotation = wfdb.rdann(readdir, 'atr', sampfrom = 0, sampto = endofsample)
    annotation = wfdb.rdann(readdir, 'atr')
    num = len(annotation.annsamp)
    print("total sym count: {}".format(i))
    i = 0
    lastloc = -1
    while i < num:
        loc = annotation.annsamp[i]
        sym = annotation.anntype[i]
        aux = annotation.aux[i]
        i += 1

        if '+' in sym:
            if '(N' not in aux and '(NOD' in aux:
                print("location: {},  symbol: {}, aux: {}".format(loc, sym, aux))
            elif '(VT' in aux:
                translate('(VT')
            elif '(AFIB' in aux:
                translate('(AFIB')
            elif '(PREX' in aux:
                translate('(PREX')
            elif '(SBR' in aux:
                translate('(SBR')
            elif '(BII' in aux:
                translate('(BII')
            elif aux == '(B':
                savefig(readdir, loc, 'aux-B')

                translate(aux)
            else:
                translate(aux)
            #else:
                #print("symbol check: {} == (N".format(aux))
             #   translate(aux)
        elif sym == '"':
            if 'at' in aux:
                translate('Note-at')
            elif 'TS' in aux:
                translate('Note-TS')
            elif 'ast' in aux:
                translate('Note-ast')
                print("break:  sym: {}, aux: {}".format(sym, aux))
        elif  sym == 's':
            translate(sym)
            print("sym: {}  aux: {}".format(sym, aux))
            if 'AST' in aux:
                translate('s-AST')
        elif sym == 'T':
            print("sym: {}  aux: {}".format(sym, aux))
            if 'AT' in aux:
                translate('T-AT')

        elif sym == '~':
            print("change in signal quality")
        else:
####################
            if sym == 'B':
                print("checkpoint")
                halfoffset = 100
                start = max(loc - halfoffset, 0)
                end = loc + halfoffset
                record = wfdb.rdsamp(readdir,  sampfrom = start, sampto = end)
                peakloc = peakfinder(record.p_signals[:,0])
                #print(peakloc)
                loc = peakloc - halfoffset + loc
                start = max(loc - halfoffset, 0)
                end = loc + halfoffset
                record = wfdb.rdsamp(readdir,  sampfrom = start, sampto = end)
                 
               
                plt.plot(record.p_signals[:,0])
                plt.axis('off')
                plt.ylim(-2, 2.5) 
                if not os.path.isdir(databasedir):
                    print(os.mkdir(databasedir))
                if not os.path.isdir(os.path.join(databasedir, sym)):
                    print (os.mkdir(os.path.join(databasedir, sym)))
              
                figuresavepath = '{}{}{}{}{}{}{}'.format(databasedir,"/", sym, "/", recordname,  loc, ".png") 
                plt.savefig(figuresavepath)
                plt.clf() 
                plotcount += 1
  
            translate(sym)
            continue
            if sym == 'N' and loc > 100:
                #print("location: {},  symbol: {}, aux: {}".format(loc, sym, aux))
                #no saving
                #continue

                halfoffset = 40
                start = max(loc - halfoffset, 0)
                end = loc + halfoffset
                record = wfdb.rdsamp(readdir,  sampfrom = start, sampto = end)
                peakloc = peakfinder(record.p_signals[:,0])
                #print(peakloc)
                loc = peakloc - halfoffset + loc
                start = max(loc - halfoffset, 0)
                end = loc + halfoffset
                record = wfdb.rdsamp(readdir,  sampfrom = start, sampto = end)
                 

                plt.plot(record.p_signals[:,0])
                plt.axis('off')
                plt.ylim(-2, 2.5) 
                if not os.path.isdir(databasedir):
                    print(os.mkdir(databasedir))
                if not os.path.isdir(os.path.join(databasedir, sym)):
                    print (os.mkdir(os.path.join(databasedir, sym)))
              
                figuresavepath = '{}{}{}{}{}{}{}'.format(databasedir,"/", sym, "/", recordname,  loc, ".png") 
                plt.savefig(figuresavepath)
                plt.clf() 
                plotcount += 1
                greyimg = Image.open(figuresavepath).convert('L')
                greyimg.save(figuresavepath)
                #plot only one
               
                figurenewpath = '{}{}{}{}{}{}{}'.format(databasedir,"/", sym, "/", recordname, loc, "_mask.png")
                #print(ori.__dict__)
                orimask=np.zeros([480,640], dtype=np.uint8) 
                orimask[0:480, 240:400] = 255
                ori_mask = Image.fromarray(orimask)
                ori_mask.save(figurenewpath) 
                               

                if lastloc > 0:
                    gap = loc - lastloc 
                    if gap < 300 and gap > 100:
                        newloc = lastloc + math.ceil(gap / 2)
                        start = max(newloc - halfoffset, 0)
                        end = newloc + halfoffset
                        record = wfdb.rdsamp(readdir,  sampfrom = start, sampto = end)
                        plt.plot(record.p_signals[:,0])
                        plt.axis('off')
                        plt.ylim(-2, 2.5) 
                        negsym = "NEG"
                        if not os.path.isdir(databasedir):
                            print(os.mkdir(databasedir))
                        if not os.path.isdir(os.path.join(databasedir, negsym)):
                            print (os.mkdir(os.path.join(databasedir, negsym)))
              
                        figuresavepath = '{}{}{}{}{}{}{}'.format(databasedir,"/", negsym, "/neg_", recordname, loc, ".png") 
                        negmasksavepath = '{}{}{}{}{}{}{}'.format(databasedir,"/", negsym, "/neg_", recordname, loc, "_mask.png") 
                        plt.savefig(figuresavepath)
                        plt.clf()
                        greyimg = Image.open(figuresavepath).convert('L')
                        greyimg.save(figuresavepath)
 


                        negmask = np.zeros([480,640], dtype=np.uint8) 
                        neg_mask_img = Image.fromarray(negmask)
                        neg_mask_img.save(negmasksavepath)

                lastloc = loc
            elif sym == 's':
                print("location: {},  symbol: {}, aux: {}".format(loc, sym, aux))
                translate(aux)
            else:
                #print("location: {},  symbol: {}".format(loc, sym))
                translate(sym)
                if '+' in sym:
                    print("check sym == + clause") 


#print(annotation.__dict__)
print("{}: recordcount: {}, plotcount: {}, ptucount: {}".format(originfiledir, recordcount, plotcount, ptucount))


for keys ,values in symbol_dict.items():
    print("{} \t {}".format(keys, values))
