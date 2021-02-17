'''
author: bg
goal: 
type: util @ data source specific content parsing 
how: 
ref: 
refactors: 
'''

import os, glob 
import numpy as np 
from skimage import io 

import utilz 

STARE_FUNDUS_CONTENT_FPATH = "/mnt/externz/zRepoz/003_school/01_ocular/stare_fundus.txt"

## TODO: refactor + generalize 
## Pipeline = folder --> subsets --> file --> AnImage --> eq --> colormaps --> vessels --> save --> plot/report     
def load_stare_fundus_dir(  durl="../../data/stare_fundus", ext='*.ppm',
                            dcodes_txt="disease_codes.txt",
                            data_labelz_txt="all-mg-codes.txt",
                            outfile_sep='\t',
                            outputfile_fpath=STARE_FUNDUS_CONTENT_FPATH):

    disease_codez = {} #'7' :[ ['Background Diabetic Retinopathy', 'BDR-NPDR'], ]
    per_file_codez = {} #im0001 :['13 14', 'Choroidal Neovascularization AND Histoplasmosis'] 

    ## 0. disease code infor 
    print("Setting up disease codez map")
    def load_txt(fname, outds):
        with open(f"{durl}/{fname}", 'r') as fd:
            for f in fd.readlines():
                rec = f.split("\t")  
                outds[rec[0]] = [x.strip() for x in rec[1:] if len(x) > 0]
                #yield rec 


    load_txt(dcodes_txt,disease_codez)

    print("Setting up disease codez one hot encoding map")
    disease_one_hot = np.eye(len(disease_codez) )
    #print("one-hot shape: ", disease_one_hot.shape)

    #print(disease_codez)
    print("Setting up disease codez per image entry")
    load_txt(data_labelz_txt,per_file_codez)

    def get_code(fname):
        rec = per_file_codez[fname]
        dcodez = [r for r in rec[0].strip().split(' ') if len(r) >0 and r.isnumeric() ]
        #print(rec)#,"\n",dcodez,"\n")
        dcodez_str = [disease_codez[i] for i in dcodez ] 
        #print(fname, dcodez,"===" ,dcodez_str)

        dcode_one_hot = np.zeros( len(disease_codez) )
        for d in dcodez:
            if dcode_one_hot[int(d)] == 0.:
                dcode_one_hot += disease_one_hot[int(d)]

        # print(dcodez, "---", list(dcode_one_hot))

        return( '++'.join([c for c in dcodez]), 
                '++'.join([c[1] for c in dcodez_str] ),
                '++'.join([c[0] for c in dcodez_str] ),
                '++'.join(rec[-1:]),
                list(dcode_one_hot) )


    ## 1. glob dir @ image files 
    print("Creating record per image entry")
    storage = []
    for f in sorted(glob.glob( f"{durl}/{ext}") ):
        fname = os.path.splitext( os.path.basename(f) )[0]
        img = io.imread( f )
        ishape = img.shape 
        imin, imax, imean, istd = img.min(), img.max(), img.mean(), img.std()

        dicode, dscode, ddcode, dnotes, dcode_one_hot = get_code(fname) 

        storage.append([fname, f, ishape, imin, imax, imean, istd] + dcode_one_hot + [dicode, dscode, ddcode, dnotes] )


    headerz = ['fname', 'fpath', 'ishape', 'imin', 'imax', 'imean', 'istd']        
    fheaderz = ['dcodez_id', 'dcodez_short', 'dcodez_desc', 'dnotes']
    dcode_one_hot_headerz = [r[1] for r in disease_codez.values() ]

    ## 2. dump to file 
    print("Saving records to file") 
    
    headers = headerz + dcode_one_hot_headerz + fheaderz

    with open(outputfile_fpath, 'w') as fd:
        fd.writelines( outfile_sep.join( headers) ) 
        fd.write('\n')
        for r in storage: 
            fd.writelines( outfile_sep.join( [ str(i) for i in r]) ) 
            fd.write('\n')
    
    print("Done")




if __name__ == "__main__":
    fundus_dir = "/mnt/externz/zRepoz/datasets/fundus/stare" 

    print("=== 1. Creating content 'flat' file === ")
    load_stare_fundus_dir(durl=fundus_dir) 

    print("=== 2. Loading content 'flat' file === ")
    dcontent = utilz.FileIO.file_content(STARE_FUNDUS_CONTENT_FPATH) 
    print( next( dcontent ) )
    for i in range(3):
        print( next( dcontent ) )

    


