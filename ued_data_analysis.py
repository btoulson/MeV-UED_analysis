import os, re, glob, h5py
from pathlib import Path

import numpy as np
from PIL import Image

from scipy import optimize
import cv2 

from collections import OrderedDict
from collections import Counter



class file_to_h5:

    def __init__(self, path_data='/cds/home/b/bwt/gued_test_data', folder_key='images-ANDOR1'):
        self.path_data =  path_data
        self.folder_key = folder_key
        self.fs = [o.name for o in os.scandir(Path(self.path_data) / self.folder_key) if o.is_file()]
        self.res = sorted(file_to_h5._get_files(self.path_data, self.fs, extensions='tif'))         
        print(f"folder: {self.folder_key}")
        print(f"path: {self.path_data}")
        print(f"found {len(self.res)} files")
            
    def _get_files(p, fs, extensions=None):
        p = Path(p)
        res = [f for f in fs if not f.startswith('.')
               and ((not extensions) or f.split(".")[-1].lower() in extensions)]
        return res
          

    def load_and_save_h5(self, activate_save=False, name_h5="ued_test_2"):
        """Load files and write to h5"""
        path_h5 = os.path.join(self.path_data, name_h5 + ".h5")
        numfiles = len(self.res)
        
        if activate_save:
            with h5py.File(path_h5, 'w') as h5:

                im_num = []
                im_pos = []

                for idx, f in enumerate(self.res):
                    im_info = re.split(r'[-_]+', f)[2:4]
                    im_num.append(im_info[0])
                    im_pos.append(im_info[1])
                    im =  np.array(Image.open(Path(self.path_data) / self.folder_key / f))
                    if (idx < 5 or idx > numfiles-5): print(im_info, f, np.amax(im))
                    #364 x 2.1 mb tif should be 764 MB -> h5 is 1.5 GB for float32, 763 MB for float16
                    dataset = h5.create_dataset("/" + self.folder_key + "/" + im_info[0] + "_" + im_info[1], shape=np.shape(im),  dtype=np.uint16)
                    dataset[:] = im
                
                
    def get_info_h5(self, file_path_h5):
        """Parse metainfo from h5 file necessary to load UED_Exp class""" 
        with h5py.File(file_path_h5, 'r') as h5:
            delays_keys = [key for key in h5[self.folder_key].keys()]
            delays_values = [float(delays_key.split("_")[-1]) for delays_key in delays_keys]
            delays_i = ord_dict(delays_keys, delays_values)
            example_img = np.array(h5[self.folder_key + '/' + delays_keys[0]])

        num_delays = len(delays_i())
        print("Shape of example img", example_img.shape)
        CCD_height = np.shape(example_img)[1]
        CCD_length = np.shape(example_img)[0]

        metainfos = {'delays_i': delays_i, 'num_delays': num_delays, 
                     'CCD_height': CCD_height, 'CCD_length': CCD_length,
                     'folder_key': self.folder_key, 'file_path_h5': file_path_h5
                    }
        print('Number of delays: ' + str(num_delays))
        print(delays_i.keys())

        return metainfos  
                

class ord_dict():
    # this is a custom class that is used to store the attributes associated with
    # an experiment. For each attribute type (ex. delay, c, pump-on/pump-off)
    # we define methods for getting:
        # keys: all the possibile attribute values as a string 
        # values: all the possible attribute values as a number
        # index: all the possible attribute values as an index starting from zero
    # we define dictionaries for getting:
        # from the key, the value -> dict_val
        # from the key, the index -> dict_idx
    # we define a call property:
        # when the attribute is called, it returns a zipped matrix of all the keys, values and index
    # we define a getitem property:
        # when we get an item from the attribute:
            # if the item is an index, it returns the key associated with the index
            # it the item is a string, it returns the key if the string is a key
    
    def __init__(self, keys, values):
        self._keys = []
        self._values = []
        self._index = []
        self._zipped_value_key = sorted(zip(values, keys))
        index_start = 0
        for couple_value_key in self._zipped_value_key:
            self._keys.append(couple_value_key[1])
            self._values.append(couple_value_key[0])
            self._index.append(index_start)
            index_start += 1
        self._zipped_key_value_index = list(zip(self._keys, self._values, self._index))
        self._zipped_key_value = list(zip(self._keys, self._values))
        self._zipped_key_index = list(zip(self._keys, self._index))

        self.dict_val = OrderedDict(self._zipped_key_value)
        self.dict_ind = OrderedDict(self._zipped_key_index)

    def keys(self, *args):
        if len(args) == 1:
            return self._keys[args[0]]
        elif len(args) == 0:
            return self._keys
        else:
            print('only one ore zero inputs allowed')

    def values(self, *args):
        if len(args) == 1:
            return self._values[args[0]]
        elif len(args) == 0:
            return self._values
        else:
            print('only one ore zero inputs allowed')

    def index(self, *args):
        if len(args) == 1:
            return self._index[args[0]]
        elif len(args) == 0:
            return self._index
        else:
            print('only one ore zero inputs allowed')

    def __getitem__(self, i):
        if type(i) is int:
            return self._keys[i]
        elif type(i) is str:
            if i in self._keys:
                return i
            else:
                print("this is not an allowed value")

    def __call__(self):
        return self._zipped_key_value_index
    


class UED_Exp:

    def __init__(self, info_h5):
        #info_h5 = get_info_h5(file_path_h5)
        self.file_path_h5 = info_h5['file_path_h5']
        self.delays_i = info_h5['delays_i']
        self.delays_unique = np.unique(self.delays_i.values())
        self.delays_dict = {}
        self.delays_unique_cnt = self.delays_dict.values() 
        self.num_delays = info_h5['num_delays']
        self.folder_key = info_h5['folder_key']
        self.CCD_length = info_h5['CCD_length']
        self.CCD_height = info_h5['CCD_height']
        print("CCD dims:",  self.CCD_length, self.CCD_height)

        self.img_m = np.empty((self.num_delays, self.CCD_length, self.CCD_height), dtype=np.uint16)
        self.imgN = np.empty((self.num_delays, self.CCD_length, self.CCD_height), dtype=np.uint16)
        self.img_mean = np.empty((len(self.delays_unique), self.CCD_length, self.CCD_height), dtype=np.float32)
        self.imgN_mean = np.empty((len(self.delays_unique), self.CCD_length, self.CCD_height), dtype=np.float32)
            
        self.center = np.array([466.27, 458.95]) 
        self.maxRadius = 450 # important! Change with caution, this controls clipping of data!
        self.roi=(180,450) # Very sensitive to ROI, center of detector might have distortions / nonuninform detection efficiency from damage?
        self.useQuadrants = (0,1,2,3) # 0 is trashed from the beam dump and nonuninform detection efficiency?
        
        self.quads = np.empty((self.num_delays, self.maxRadius, len(self.useQuadrants) ), dtype=np.uint16) # seem to be having some problems with Rscaling if not int16 
        self.meanquads = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))
        self.stdquads = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))
        self.normRegion = (100,200)

        self.quadsN = np.empty((self.num_delays, self.maxRadius, len(self.useQuadrants) ))
        self.meanquadsN = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))
        self.stdquadsN = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))

        self.load_h5(self.file_path_h5)
        self.duplicates(info_h5['delays_i'].values())
        self.quadrants()
        self.quadrants_stats()
        self.mean_image_each_delay()

    def load_h5(self, file_path_h5):
        with h5py.File(file_path_h5, 'r') as h5:
            for delay_key, delay_value, delay_idx in self.delays_i():
                #for img, img_value, img_idx in self.img_i():
                self.img_m[delay_idx, :, :] = np.array(h5[self.folder_key + '/' + delay_key])
    
    def duplicates(self, n): #n="123123123"
        counter = Counter(n) #{'1': 3, '3': 3, '2': 3}
        dups=[i for i in counter if counter[i]!=1] #['1','3','2']
        result = {}
        for item in dups:
            result[item] = [i for i,j in enumerate(n) if j==item]
        print(result)
        self.delays_dict = result
        
    def quadrants(self):
        print(f"Using center: {self.center}")
        for delay_key, delay_value, delay_idx in self.delays_i():
            center = find_center(self, delay_idx)
            self.quads[delay_idx, :, :] = np.array(center.cart_to_polar_quad(self.center, Rscaling=False))
            intensity_in_region = np.mean(self.quads[delay_idx, self.normRegion[0]:self.normRegion[1]],axis=0)
            self.quadsN[delay_idx] = self.quads[delay_idx] / intensity_in_region
            
            self.imgN[delay_idx] = self.img_m[delay_idx] / np.mean(intensity_in_region)
            print("Delay_key:", self.delays_i[delay_idx], "   Delay_idx:", delay_idx, f"\tIntensity in normRegion: {np.mean(intensity_in_region):.2f}" )

    def quadrants_stats(self):
        """axis=0 is taking mean over data with same delay step. Example dataset has 364 images but only 31 unique delay steps"""
        for idx, delay_unique in enumerate(self.delays_unique):
            self.meanquads[idx] = np.mean(self.quads[self.delays_dict[delay_unique]],axis=0)
            self.stdquads[idx] = np.std(self.quads[self.delays_dict[delay_unique]],axis=0)
            self.meanquadsN[idx] = np.mean(self.quadsN[self.delays_dict[delay_unique]],axis=0)
            self.stdquadsN[idx] = np.std(self.quadsN[self.delays_dict[delay_unique]],axis=0)

    def mean_image_each_delay(self):
        """Can precompute mean image as it doesn't require center. Needs imgN to be calculated by quadrants() first."""
        for idx, delay_unique in enumerate(self.delays_unique):
            #print(idx, delay_unique)
            self.img_mean[idx] = np.mean(self.img_m[self.delays_dict[delay_unique]],axis=0)
            self.imgN_mean[idx] = np.mean(self.imgN[self.delays_dict[delay_unique]],axis=0)
            #Subtract min of that particular image, hopefully normalizes to same background
            self.img_mean[idx] -= np.min(self.img_mean[idx])
            self.imgN_mean[idx] -= np.min(self.imgN_mean[idx])
 
        # Now subract first time point (negative delay) to highlight change w.r.t. time, the first delay has zero intensity by this definition
        self.img_mean -= self.img_mean[0]
        self.imgN_mean -= self.imgN_mean[0]
        
        
    #def mean_images(self):
        # mean images at each delay, and normalized images
        #for delay_key, delay_value, delay_idx in self.delays_i():
        #    cart = np.array(self.img_m[delay_idx] - np.min(self.img_m[delay_idx]), dtype = np.uint16 ) 
        #
        #    intensity_in_region = np.mean(self.quads[delay_idx,self.normRegion[0]:self.normRegion[1]],axis=0)
            

       
    
class find_center:
    def __init__(self, Exp, idx):
        self.cart = np.array(Exp.img_m[idx] - np.min(Exp.img_m[idx]), dtype = np.uint16) # crucial to subtract detector background offset! 
        self.center_i = Exp.center #np.array([465, 457], dtype=np.int16) # Could deliberately pick a terrible guess to test convergence
        self.maxRadius = 450 # important! Change with caution, this controls clipping of data!
        self.roi=(180,450) # Very sensitive to ROI, center of detector might have distortions / nonuninform detection efficiency from damage?
        self.useQuadrants = Exp.useQuadrants # For all quadrants use: (0,1,2,3)

    def cart_to_polar_quad(self, center,  Rscaling=False):
        """note x,y convention is swapped in cv2"""
        polar = cv2.linearPolar(self.cart, (center[1],center[0]), self.maxRadius, cv2.INTER_LINEAR)
        polar2 = cv2.resize(polar, (self.maxRadius, 360)) 
        if Rscaling:
            R = np.arange(self.maxRadius).astype(polar2.dtype)
            Rweight = np.tile(R, (360,1))
            polar2 *= Rweight

        polar_quad = find_center.polar_quadrants(polar2)
        return polar_quad[:,self.useQuadrants]

    def polar_quadrants(polar2):
        ''' find profile for each quadrant'''
        polar_quad = [np.mean(s, axis=0) for s in np.split(polar2, 4)]
        return np.transpose(polar_quad)

    def err_fn(self, center):
        """Manual root mean square function as we need to use an optimizer that can cope with coarse steps, or we get no gradient!! 
           RScaling is great for weighting the fit to the structure seen at higher radii"""
        polar_quad = find_center.cart_to_polar_quad(self, center, Rscaling=True)[self.roi[0]:self.roi[1]]
        return np.sqrt(np.mean(np.square(np.std(polar_quad, axis=1))))


