import pandas as pd 
import numpy as np 

import os, time, glob #TODO: pathlib 
import pickle 

#import skimage as skimage 
from skimage import io, color 
import utilz 


from sklearn.model_selection import train_test_split

## === TODO: relevance + displayable 
class ZSerializableMixin: 
    def load(self, fpath):
        with open(fpath, 'rb') as fd:
            #self.data = pickle.load(fd)
            self.__dict__.update(pickle.load(fd)) ## TODO: 
            

    def dump(self, fpath):
        with open(fpath, 'wb') as fd:
            pickle.dump( self.__dict__, fd ) 

## === Collection handler << TODO: sklearn transformer/pipeline on it + Make useful 
class ZMultiModalRecord(ZSerializableMixin, list): 
    def __init__(self, itemz):
        super().__init__(itemz) 

## === Modality data types 
class ZModal(ZSerializableMixin):
    TYPE_GEN = 0
    TYPE_TEXT = 1
    TYPE_IMAGE = 2
    TYPE_TXT_FILE = 3 
    TYPE_PICKLE_FILE = 4 

    def __init__(self, label, data=None, mod_type=TYPE_GEN):
        self.label = label 
        self.data = data 
        self.mod_type = mod_type 

    @property
    def size(self):
        return 0 if self.data is None else len(self.data ) \
                    if not isinstance(self.data, (np.ndarray, np.generic)) else self.data.shape 

    @property
    def stats(self): #TODO: fix error on some dtype i don't remember 
        headerz = ['label', 'size', 'min', 'max', 'mean']
        statz = []
        if (isinstance( self.data ,(np.ndarray, np.generic, list, tuple)) ):
            dmean = round(np.mean(self.data),3)
            dmin = round(np.min(self.data),3)
            dmax = round(np.max(self.data),3)
            dsize = self.data.shape if isinstance( self.data ,(np.ndarray, np.generic)) else len(self.data)
        else:
            dmean = 'N/A'
            dmin = min(self.data)
            dmax = max(self.data)
            dsize = len(self.data)
        statz = [self.label, dsize, dmin, dmax, dmean]
        return statz, headerz 

    def __repr__(self):
        s, h = self.stats
        return f"{self.__class__}: n={self.size}, data.dtype={type(self.data)}\n\t{h} \n\t{s}"


## === Image Objects
class ZImage(ZModal):    
    def __init__(self, label, fpath, cleanit=True):
        super().__init__(label, mod_type=ZModal.TYPE_IMAGE) 
        self.fpath = fpath
        self.cleanit = cleanit 
        self.data = io.imread(self.fpath )
        if self.cleanit:
            self.data = utilz.Image.basic_preproc_img(self.data) 
    
    # @property ##TODO: allow tuning cleaning paramz 
    # def clean_data(self):
    #     return utilz.Image.basic_preproc_img(self.origi_data)  if self.cleanit else self.origi_data 

    @property        
    def gray(self): 
        return color.rgb2gray( self.data )  ## TODO: to clean or not to clean 

    @property
    def red(self):
        return utilz.Image.get_channel(self.data, 0)
    @property
    def green(self):
        return utilz.Image.get_channel(self.data, 1)
    @property
    def blue(self):
        return utilz.Image.get_channel(self.data, 2)  
       
## === Fundus Image Remapped 
class RemappedFundusImage(ZImage):

    def __init__(self, label, fpath, cleanit=True): 
        super().__init__(label, fpath, cleanit=cleanit) 
    
    ## --- TODO: arch/struct these three well + decouple @clean_data Vs data
    def green_channel_update(self, img):
        return img.copy() 

    def red_channel_update(self, img, thresh=0.97):
        o = img.copy()
        rrange = o.max() - o.min() 
        o[ (o - o.min()/rrange) < thresh ] = 0
        return o 

    def blue_channel_update(self, img, thresh=1): 
        o = img.copy()  
        t = 1 if (thresh == 1 and o.max()==255) else (1/255) ##TODO: change o.max to dtype check + else case blue is lost;recompute thresh
        o[ o != t] = 0
        return o

    def vessels_channel(self, img, mtype=0):
        o = utilz.Image.edgez(img, mtype) 
        return o


    @property ##TODO: on dd compute Vs store another data object ++ save to fmap file
    def remapped_data(self): 
        outiez = []
        # 1. resize, equalize, rescale-float :@: using self.clean_data 
        img = utilz.Image.hist_eq(self.gray) 
        # 2. vessels
        outiez.append( self.vessels_channel( utilz.Image.hist_eq(self.green) ) )  
        # 3. color channelz
        outiez.append( self.green_channel_update(utilz.Image.hist_eq(self.green) )  ) 
        outiez.append( self.red_channel_update(utilz.Image.hist_eq(self.red) )  )
        outiez.append( self.blue_channel_update(utilz.Image.hist_eq(self.blue) ) )  
        # 4. combine color channelz
        o = np.dstack(outiez) 
        return o 
    

## === Stats, Visualize and Generator @ listing I/O
class PdDataStats:
    TYPE_IN_MEM_ARRAY = 0 ## pass pair of data and headerz 
    TYPE_DIR = 1
    TYPE_TXT_LINES_FILE = 2
    TYPE_JSON_FILE  = 3

    DATA_DICT_RECORDZ_KEY = 'recordz'
    DATA_DICT_HEADERZ_KEY = 'headerz'
    ##TODO: with some defaults or remove for not 
    loaderz = [ (None, {}),
                (utilz.FileIO.folder_content,{}), 
                (utilz.FileIO.file_content, {}),
                (None,{})
            ]
    ## data = dict of {'recordz': (array|fpath ), 'headerz':(list), } Minimum 
    def __init__(self, data_dict, ftype=TYPE_IN_MEM_ARRAY):
        self.data = data_dict 
        self.ftype = ftype 
        self.dframe = None 
        self.load()  
    
    @property
    def size(self):
        return len(self.dframe) if self.dframe is not None else 0 
    
    ## TODO: lighten 
    def load(self, **kwargz):
            # has_header_row=False, rec_parser=None, sep='\t', 
            # ext="*.*", additional_info_func=None, fname_parser=None, sep='-'): 
        loader, default_kargz = self.loaderz[self.ftype] 
        recz = self.data.get(PdDataStats.DATA_DICT_RECORDZ_KEY, None)
        headerz = self.data.get(PdDataStats.DATA_DICT_HEADERZ_KEY, None)

        if loader is not None: ## assume in mem otherwise 
            fkwargz = self.data.copy() 
            # print(fkwargz)
            fkwargz.pop(PdDataStats.DATA_DICT_RECORDZ_KEY, None)
            fkwargz.pop(PdDataStats.DATA_DICT_HEADERZ_KEY, None)
            fpath = recz
            recz = []
            for r in loader(fpath, **{**kwargz, **fkwargz }):  ##generator auch!!
                # print(r)
                # headerz = None
                recz.append(r) 

        if headerz is None:
            n = len(recz[0])
            headerz = [f'col_{i}' for i in range(n) ] 

        self.dframe = pd.DataFrame.from_records(recz, columns=headerz) 

    ## TODO: menu of commons ELSE below guys are pointless since can access dframe directly 
    ### ==== Stats and Visuals ====
    def select_colz_by_name(self, colz=None):
        return self.dframe.loc[:,colz] if colz is not None else self.dframe
    
    def summarize(self, colz=None, include='all'):
        return self.select_colz_by_name(colz).describe(include=include)
    
    ## TODO: seaborn etc 
    # kind = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html 
    def visualize(self, colz=None, countz=False, plt_type='bar', **kwargz):
        if countz: ## TODO: refactor, generalize(dframe Vs dseries), Common charts menu 
            return self.dframe[colz].value_counts().plot(kind=plt_type, **kwargz)
        else:
            return self.select_colz_by_name(colz).plot(kind=plt_type, **kwargz) 




## === Training Dataset - split etc
class ZPdDataset(PdDataStats):
    def __init__(self, data_dict, ftype=PdDataStats.TYPE_IN_MEM_ARRAY):
        super().__init__(data_dict, ftype )
        ## TODO: have these as masks and not dframes from sklearn 
        self.train_mask = None
        self.test_mask = None
        self.validation_mask = None 
    
    def train_test_validate_split(self, test_perc=0.3, validate_perc=0, shuffle=True):
        self.train_set, self.test_set = train_test_split(self.dframe, 
                                                    test_size=test_perc,
                                                    shuffle=shuffle,
                                                    random_state=999)
        print(f"Done splitting {test_perc}% test with shuffle = {shuffle}")



if __name__ == "__main__":
    print("CWD: ", os.getcwd() )
    f = "/mnt/externz/zRepoz/003_school/notebooks/01_ocular/notebooks/"
    f = "logit_res_0.625.png"
    x = RemappedFundusImage('tester',f)
    print(x)
    z = x.remapped_data
    print( type(z), len(z), z.shape ) 

    utilz.Image.plot_images_list([x.data, z[:,:,:3]])


    multimod = ZMultiModalRecord([x, ZModal('desc', 'The quick brown fox jumped over the lazy dogs')])
    _ = [ print(f"{r.label}: {r.stats}") for r in multimod] 

    pa = [[1,2,3,4],[10, 20, 30, 40]]
    pah = ['a', 'b', 'c', 'd']
    pdstats = PdDataStats(
                    {PdDataStats.DATA_DICT_RECORDZ_KEY:pa,
                    PdDataStats.DATA_DICT_HEADERZ_KEY:pah,                    
                    },
                     ftype=PdDataStats.TYPE_IN_MEM_ARRAY ) 
    
    pdstats.dframe

    print("\n----\n\n") 

    pdstats = PdDataStats(
                    {PdDataStats.DATA_DICT_RECORDZ_KEY:'/mnt/externz/zRepoz/003_school/01_ocular/notebooks/disease_codes.txt',
                    PdDataStats.DATA_DICT_HEADERZ_KEY:['DCODE', 'DFullName', 'DShortCode'],
                    'rec_parser': utilz.FileIO.row_parser                     
                    },
                     ftype=PdDataStats.TYPE_TXT_LINES_FILE ) 
    
    pdstats.dframe 

    print("\n----\n\n") 

    pdstats = ZPdDataset(
                    {PdDataStats.DATA_DICT_RECORDZ_KEY:'/mnt/externz/zRepoz/datasets/fundus/stare',
                    'fname_parser': utilz.FileIO.row_parser,
                    'additional_info_func':utilz.FileIO.image_file_parser   ,
                    'ext':"*.ppm"                  
                    },
                     ftype=PdDataStats.TYPE_DIR ) 
    
    pdstats.dframe 

    print("\n----\n\n") 