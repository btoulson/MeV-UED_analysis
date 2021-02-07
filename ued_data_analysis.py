import os, re, glob, h5py
from pathlib import Path

import numpy as np
from PIL import Image

from scipy import optimize
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter, rotate

import cv2 

from collections import OrderedDict
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle

class file_to_h5:

    def __init__(self, path_data='/cds/home/b/bwt/gued_test_data', folder_key='images-ANDOR1'):
        self.path_data =  os.path.join(path_data + "/scan001/") # should point to folder of UED tif files on SLAC psana
        self.folder_key = folder_key
        print(f"folder: {self.folder_key}")
        print(f"path: {self.path_data}")
        self.fs = [o.name for o in os.scandir(Path(self.path_data) / self.folder_key) if o.is_file()]
        self.res = sorted(self._get_files(self.path_data, self.fs, extensions='tif'))         
        print(f"found {len(self.res)} files")
            
            
    def _get_files(self, p, fs, extensions=None):
        p = Path(p)
        res = [f for f in fs if not f.startswith('.')
               and ((not extensions) or f.split(".")[-1].lower() in extensions)]
        return res
          

    def load_and_save_h5(self, activate_save=False, name_h5="ued_test_2"):
        """Load files and write to h5"""
        path_h5 = os.path.join(os.getcwd() + "/" + name_h5 + ".h5") 
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
                
                
class parse_h5:
    """Separated parser into separate class. Can read h5 standalone, does not need to scan directories for files"""
    def get_info_h5(file_path_h5, folder_key='images-ANDOR1'):
        """Parse metainfo from h5 file necessary to load UED_Exp class""" 
        with h5py.File(file_path_h5, 'r') as h5:
            for item in h5.keys():
                print(item + ":", h5[item])
            delays_keys = [key for key in h5[folder_key].keys()]
            delays_values = [float(delays_key.split("_")[-1]) for delays_key in delays_keys]
            delays_i = ord_dict(delays_keys, delays_values)
            example_img = np.array(h5[folder_key + '/' + delays_keys[0]])

        num_delays = len(delays_i())
        print("Shape of example img", example_img.shape)
        CCD_height = np.shape(example_img)[1]
        CCD_length = np.shape(example_img)[0]

        metainfos = {'delays_i': delays_i, 'num_delays': num_delays, 
                     'CCD_height': CCD_height, 'CCD_length': CCD_length,
                     'folder_key': folder_key, 'file_path_h5': file_path_h5
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
        #self.delays_unique_cnt = self.delays_dict.values() 
        self.num_delays = info_h5['num_delays']
        self.folder_key = info_h5['folder_key']
        self.CCD_length = info_h5['CCD_length']
        self.CCD_height = info_h5['CCD_height']
        print("CCD dims:",  self.CCD_length, self.CCD_height)
        
        self.delays_cnt  = np.array([int(re.split(r'[-_]+', d)[0]) for d in self.delays_i])  # may want to find counter / know the order images were acquired
        self.delays_mask = np.empty(self.num_delays, dtype=np.float32)
        self.drop = np.empty(0)
        self.img_integral = np.empty(self.num_delays, dtype=np.float32)

        self.img_m = np.empty((self.num_delays, self.CCD_length, self.CCD_height), dtype=np.int16)
        self.imgN = np.empty((self.num_delays, self.CCD_length, self.CCD_height), dtype=np.int16)
        self.img_mean = np.empty((len(self.delays_unique), self.CCD_length, self.CCD_height), dtype=np.float32)
        self.imgN_mean = np.empty((len(self.delays_unique), self.CCD_length, self.CCD_height), dtype=np.float32)
            
        self.center = np.array([501, 548]) 
        self.maxRadius = 450 # important! Change with caution, this controls clipping of data!
        self.roi=(130,400) # Use pixels > 120! Center is sensitive to ROI, center of detector might have distortions / nonuninform detection efficiency from damage?
        self.useQuadrants = (0,1,2,3) # 0 is trashed from the beam dump and nonuninform detection efficiency?
        
        self.sigma_outliers = 1.8
        
        self.quads = np.empty((self.num_delays, self.maxRadius, len(self.useQuadrants) ), dtype=np.int16) # seem to be having some problems with Rscaling if not int16 
        self.meanquads = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))
        self.stdquads = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))
        self.normRegion = (180,400)

        self.quadsN = np.empty((self.num_delays, self.maxRadius, len(self.useQuadrants) ))
        self.quadsNClean = {}
        self.quadsNfilt = []
        self.meanquadsN = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))
        self.stdquadsN = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))

        self.load_h5(self.file_path_h5)
        self.duplicates() # group all acq. at same time delay
        self.mask_outliers() # by image intensity
        self.approx_center() # start with a crude center from sampling a few images. We can find the exact center of each image later.
        self.quadrants()
        self.quadrants_stats()
        # self.mean_image_each_delay() # skip while debugging. Requires quadrants(imgNcalc=True)
        
        

        
    def load_h5(self, file_path_h5):
        smooth = True
        sigma = 2.0  # use a very weak gaussian filter (e.g. sigma=1-3 pixels) to remove outlier pixels. Must be an odd number.
        if smooth: 
            print(f"Starting smoothing of images. Using blur to find outlier pixels. Only outliers by sigma={sigma} are replaced with blur.  Better than blurring whole image!")

        with h5py.File(file_path_h5, 'r') as h5:
            for delay_key, delay_value, delay_idx in self.delays_i():
                if delay_idx%100 == 0:
                    print(delay_idx, delay_key, delay_value)
                d = np.array(h5[self.folder_key + '/' + delay_key])
                if smooth: 
                    #d = gaussian_filter(d, sigma) # slow
                    #d = cv2.GaussianBlur(d, (0,0), sigma) #  cv2 5x faster than scipy. Can leave kernel as (0,0), cv2 will figure it out from sigma
                    #d = cv2.medianBlur(d, sigma)
                    d = self.find_hot_pixels(d, sigma=sigma)
                self.img_m[delay_idx, :, :] = d
                
                
    def find_hot_pixels(self, data, sigma=2.0, verbose=False):
        """Somewhat smart way to use blurring to find the hot pixels, then replace only hot pixels with blur. Better than blurring whole image but slow!"""
        
        # need to make sure data is not unsigned integer else we have problems subracting / normalizing
        d = np.array(data, dtype=np.float32)
        blurred = cv2.medianBlur(d, ksize=5) # blur kernel size should be larger (~5 pixels. 7 is computationally expensive) to suffiently blur or it won't catch hot pixels! 
        difference = d - blurred
        
        # Threshold is the std dev over ALL pixels of the dfference [between raw and blurred] (so is normalized by looking for a difference)
        threshold = sigma*np.std(difference) # If 2.1, catch zero hot pixels. If 2.0, very slow as too many points included. 
        
        #find the hot pixels, but ignore the edges
        hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
        hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column
             
        fixed_image = np.copy(d) #This is the image with the hot pixels removed
        for y,x in zip(hot_pixels[0],hot_pixels[1]):
            fixed_image[y,x]=blurred[y,x] 
        
        if verbose:
            print("FUNCTION find_hot_pixels() called")
            print(f"Threshold: +/- {int(threshold)}, Hot pixels that will be replaced with blur: {hot_pixels.shape[1]}")
            plt.hist(np.ravel(difference), 300, label="difference")
            plt.legend()
            plt.show()
                
        return fixed_image#, blurred, difference, hot_pixels
            
    
    def duplicates(self): #n="123123123"
        n = self.delays_i.values()
        counter = Counter(n) #{'1': 3, '3': 3, '2': 3}
        dups=[i for i in counter if counter[i]!=1] #['1','3','2']
        result = {}
        for item in dups:
            result[item] = [i for i,j in enumerate(n) if j==item]
        #print(result)
        self.delays_dict = result
        
        
    def mask_outliers(self):
        print(f"FUNCTION mask_outliers() called")
        print(f"Starting outlier analysis using image integral using sigma = {self.sigma_outliers}")
        img_integral = np.array([n.sum() for n in self.img_m])
        self.img_integral = img_integral
        delay_idx = np.array([delay_idx for delay_key, delay_value, delay_idx in self.delays_i()])
        droplow  = delay_idx[img_integral < np.mean(img_integral) - self.sigma_outliers * np.std(img_integral)] # list of hits to discard
        drophigh = delay_idx[img_integral > np.mean(img_integral) + self.sigma_outliers * np.std(img_integral)] # list of hits to discard
        print(f"{len(droplow)} Low to discard: {droplow}, \n {len(drophigh)} High to discard: {drophigh}")
        drop = sorted(np.append(droplow,drophigh))
        self.drop = drop
        print(f"Sorted final drop list {drop}")
        self.delays_mask = np.delete(delay_idx,drop)
        print(f"Delays: {self.num_delays}, Mask kept #: {len(self.delays_mask)}") 
        
        plt.plot(self.img_integral, label='all') # sorted by acq. number?
        masked_integral = self.img_integral[self.delays_mask]
        plt.plot(self.delays_mask, masked_integral, label='masked') # plot against correct acq. index
        plt.xlabel("Acq. #")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()
        
        
    def approx_center(self):
        print("FUNCTION approx_center() called")
        # Need to build in a guess of the center. At the moment if a bad center is in the sample, it skews the mean...

        centers = []
        verbose = False
        vmin1 = 2000
        vmax1 = 10000
        center_guess = self.center 
        samples = 100
        print(f"Using {samples} to estimate center position. Minimize difference between quadrants in ROI {self.roi} pixels: ")
        
        for delay_key, delay_value, delay_idx in self.delays_i():
            if delay_idx < samples: # test with a small number first

                center = find_center(self, delay_idx) # note that center uses roi (we want to avoid center pixels as ebeam block slightly off-center)
                center.center_i = center_guess # override center with guess
                optf_grad = optimize.minimize(center.err_fn, center.center_i, method='Nelder-Mead')  
                print(f"Delay_key: {self.delays_i[delay_idx]} \t Delay_idx: {delay_idx} \t  guess: {center.center_i}, rms: {center.err_fn(center.center_i):.2f}, \t fit center: {optf_grad.x[0]:.2f},  {optf_grad.x[1]:.2f}, rms: {center.err_fn(optf_grad.x):.2f}")
                centers.append(optf_grad.x)

                if (verbose == True) or (delay_idx < 4):
                    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14,4))
                    ax1.plot(np.log10(center.cart_to_polar_quad(center.center_i)))
                    ax1.set_title('Guess' + str(center.center_i)), ax1.set_xlabel('radius (pixel)'), ax1.set_ylabel('Log10 Intensity')

                    ax2.plot(np.log10(center.cart_to_polar_quad(optf_grad.x) ))
                    ax2.set_title('Fit' + str(optf_grad.x)), ax2.set_xlabel('radius (pixel)'), ax2.set_ylabel('Log10 Intensity')

                    cartim = ax3.imshow(center.cart, vmin=vmin1, vmax=vmax1, cmap='bwr', origin = 'lower')
                    ax3.set_title('Cartesian input'), ax3.set_xlabel('x (pixel)'), ax3.set_ylabel('y (pixel)')
                    cart_pix = 300
                    ax3.set_xlim(optf_grad.x[0]+cart_pix,optf_grad.x[0]-cart_pix)
                    ax3.set_ylim(optf_grad.x[1]+cart_pix,optf_grad.x[1]-cart_pix)
                    ax3.add_patch(Circle(optf_grad.x[::-1], radius=30, color='black'))
                    plt.colorbar(cartim, ax=ax3)
                    plt.show()

        c = np.array(centers)
        print(c.shape)
        center_opt = np.mean(c, axis=0)
        print(f"Fit found center from {samples} samples, mean: {center_opt}, sigma: {np.std(c, axis=0)}")
        self.center = center_opt # overwrite with new center
        
        
    def quadrants(self, imgNcalc = False):
        print("FUNCTION quadrants() called")
        # imgNcalc = False to save img memory (if True: doubles memory usage as every image is duplicated and normalized)
        print(f"Calculating quadrants... using center: {self.center}")
        for delay_key, delay_value, delay_idx in self.delays_i():
            # Try to only look at good images
            #if delay_idx in self.delays_mask:
            center = find_center(self, delay_idx)
            self.quads[delay_idx, :, :] = np.array(center.cart_to_polar_quad(self.center, Rscaling=False))
            intensity_in_region = np.mean(self.quads[delay_idx, self.normRegion[0]:self.normRegion[1]],axis=0)
            self.quadsN[delay_idx] = self.quads[delay_idx] / intensity_in_region
            if imgNcalc:
                self.imgN[delay_idx] = self.img_m[delay_idx] / np.mean(intensity_in_region)
            #print("Delay_key:", self.delays_i[delay_idx], "   Delay_idx:", delay_idx, f"\tIntensity in normRegion: {np.mean(intensity_in_region):.2f}" )
            
    
    def quadOutliers(self, sigma = 2.0):   
        print(f"FUNCTION quadOutliers() called to remove Sigma = {sigma} outliers")
        for idx, delay_unique in enumerate(self.delays_unique):
            acq_idxs = self.delays_dict[delay_unique] # for position 161.273105 returns [0,1,2,3]  etc. 
            hit_idxs = np.intersect1d(acq_idxs, self.delays_mask) # only use hits
            d = self.quadsN[hit_idxs ,:,:]
            #meanQuad = np.mean(d, axis=-1)  # mean over quadrants
            
            CleanList = []
            for quad in np.arange(d.shape[-1]):
                d2 = d[:,:,quad]
                rms = np.array([np.sqrt(np.mean(np.square(obs-np.mean(d2,axis=0) ))) for obs in d2]) #  calculate rms of 450 pixel spectrum
                Clean = d2[rms < np.mean(rms) + sigma*np.std(rms)] # keep those with an rms < 1 sigma deviation
                print(f"{delay_unique} \t{d2.shape}, {Clean.shape}, \t {(d2.shape[0] - Clean.shape[0])/d2.shape[0] *100 :.0f}% dropped")
                CleanList.append(Clean)
            
            self.quadsNClean[delay_unique] = np.array(CleanList)
    
            
    def reject_outliers(data, m=2.):
        d = np.abs(data - np.nanmedian(data))
        mdev = np.nanmedian(d)
        s = d / (mdev if mdev else 1.)
        return data[s < m]
    
    
    def removeOutliers(x, outlierConstant):
        a = np.array(x)
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        result = a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]
        return result.tolist()

    
    def quadrants_stats(self):
        """Function calculates mean properties. Note that it looks at self.delays_mask to discard bad hits by their idx
        axis=0 is taking mean over data with same delay step. Example dataset has 364 images but only 31 unique delay steps"""
        print("FUNCTION quadrants_stats() called")
        
        for idx, delay_unique in enumerate(self.delays_unique):
            acq_idxs = self.delays_dict[delay_unique] # for position 161.273105 returns [0,1,2,3]  etc. 
            #hit_idxs = np.setdiff1d(acq_idxs, self.delays_mask) # wrong, need intersection
            hit_idxs = np.intersect1d(acq_idxs, self.delays_mask)  
            self.meanquads[idx] = np.nanmean(self.quads[hit_idxs],axis=0)
            self.stdquads[idx]  = np.nanstd (self.quads[hit_idxs],axis=0)
            print(f"{delay_unique}, \t{[self.quadsN[hit_idxs][:,:,quad].shape for quad in [0,1,2,3]]}")
            self.meanquadsN[idx] = np.nanmean(self.quadsN[hit_idxs],axis=0)
            self.stdquadsN[idx]  = np.nanstd (self.quadsN[hit_idxs],axis=0)
            if idx > 100: return
        

    def mean_image_each_delay(self):
        """Can precompute mean image as it doesn't require center. Requires imgNcalc=True to be calculated by quadrants() first."""
        for idx, delay_unique in enumerate(self.delays_unique):
            acq_idxs = self.delays_dict[delay_unique] # for position 161.273105 returns [0,1,2,3]  etc. 
            hit_idxs = np.intersect1d(acq_idxs, self.delays_mask) 
            self.img_mean[idx] = np.mean(self.img_m[hit_idxs],axis=0)
            self.imgN_mean[idx] = np.mean(self.imgN[hit_idxs],axis=0)
            #Subtract min of that particular image, hopefully normalizes to same background
            self.img_mean[idx] -= np.min(self.img_mean[idx])
            self.imgN_mean[idx] -= np.min(self.imgN_mean[idx])
        # Now subract first time point (negative delay) to highlight change w.r.t. time, the first delay has zero intensity by this definition
        self.img_mean -= self.img_mean[0]
        self.imgN_mean -= self.imgN_mean[0]
    

       
    
class find_center:
    """Calculate radial profiles for a particular center. Finds center using lsq error function to find minimum difference between quadrants. Assumes images are symmetric..."""
    def __init__(self, Exp, idx):
        self.cart = np.array(Exp.img_m[idx] - int(np.mean(Exp.img_m[idx,0:30,0:30])), dtype = np.int16) # crucial to subtract detector background offset! Trying mean in corner. Be careful to have a datayupe that allows negative values if we are subtracting background, noisy pixels should be symmetrically distributed near zero. int16 or float32?
        #self.center_i = Exp.center
        self.center_i = np.array([524, 531], dtype=np.int16) # Could deliberately pick a terrible guess to test convergence. 522, 523 work well for Nov16,  524, 531 for Nov17
        self.maxRadius = 450 # important! Change with caution, this controls clipping of data!
        self.roi= Exp.roi #(180,450) # Very sensitive to ROI, center of detector might have distortions / nonuninform detection efficiency from damage?
        #print(f"fit will cut error fn between roi {self.roi}")
        self.useQuadrants = Exp.useQuadrants # For all quadrants use: (0,1,2,3)
        self.temp = np.empty((self.roi[1]-self.roi[0]), dtype=np.float32)
        
        
    def cart_to_polar_quad(self, center,  Rscaling=False):
        """note x,y convention is swapped in cv2"""
        #rotated = rotate(self.cart, 45) # wrong, need a rotation were we define center. Do we expect anisotropy or should we just move on?
        #polar = cv2.linearPolar(rotated, (center[1],center[0]), self.maxRadius, cv2.INTER_LINEAR)
        polar = cv2.linearPolar(self.cart, (center[1],center[0]), self.maxRadius, cv2.INTER_LINEAR)
        polar2 = cv2.resize(polar, (self.maxRadius, 360)) 
        if Rscaling:
            R = np.arange(self.maxRadius).astype(polar2.dtype)
            Rweight = np.tile(R, (360,1))
            polar2 *= Rweight
        polar_quad = self.polar_quadrants(polar2)
        return polar_quad[:,self.useQuadrants]

    
    def polar_quadrants(self, polar2):
        ''' find profile for each quadrant'''
        polar_quad = [np.mean(s, axis=0) for s in np.split(polar2, 4)]
        return np.transpose(polar_quad)

    
    def err_fn(self, center):
        """R scaling in the error function seems to break center finding??
           Manual root mean square function as we need to use an optimizer that can cope with coarse steps, or we get no gradient!! 
           RScaling is great for weighting the fit to the structure seen at higher radii"""
        polar_quad = find_center.cart_to_polar_quad(self, center, Rscaling=False)[self.roi[0]:self.roi[1]]
        self.temp = np.std(polar_quad, axis=1)
        return np.sqrt(np.mean(np.square(np.std(polar_quad, axis=1))))


class quads_class(object):

    def __init__(self):

        self.delays_dict = {}
        self.delays_unique = []
        self.delays_mask = []
        self.quadsN = []
        self.stage_t0 = []
        
        
    def save_quads(self, Exp, name_h5="quads_test_2b"):
        """Load files and write to h5"""
        path_h5 = os.path.join(os.getcwd() + "/" + name_h5 + ".h5") 

        with h5py.File(path_h5, 'w') as h5:

            #h5.create_dataset("/delays_dict", data=Exp.delays_dict)
            for k, v in Exp.delays_dict.items():
                #print(k,v)
                h5.create_dataset("delays_dict/" + str(k), data=np.array(v))

            h5.create_dataset("delays_unique/", data=Exp.delays_unique)
            h5.create_dataset("delays_mask/",   data=Exp.delays_mask)
            h5.create_dataset("stage_t0",       data=[161.423000])              # need some way to read info from eLog, the log & dat files are empty.

            h5.create_dataset("quadsN_0/",   data=Exp.quadsN[:,:,0])
            h5.create_dataset("quadsN_1/",   data=Exp.quadsN[:,:,1])
            h5.create_dataset("quadsN_2/",   data=Exp.quadsN[:,:,2])
            h5.create_dataset("quadsN_3/",   data=Exp.quadsN[:,:,3])
            
            
    def load_quads(self, fileName):
        self.filename = fileName
        print(self.filename)
        #load_quads(self, fileName)
        
        path_h5 = os.path.join(os.getcwd() + "/" + fileName)

        with h5py.File(path_h5, 'r') as h5:

            print("Keys: %s" % h5.keys())
            #print(dir(f))
            #for item in f.attrs.keys():
            #    print(item + ":", f.attrs[item])
            for item in h5.keys():
                print(item + ":", h5[item])

            #first_group_key = list(h5.keys())[0]
            #delays = list(h5[first_group_key])
            #print(first_group_key, delays)

            delays = list(h5["delays_dict"]) # returns a list like '161.12321'...
            result = {}
            for k in delays:
                result[float(k)] = list(h5["delays_dict"][k][:])
            #print(result) # note that the dict now must be searched like '161.12321' not 161.12321. Fixed using float()

            delays_unique = np.array(h5["delays_unique"][:])
            print("delays_unique", delays_unique)

            delays_mask = np.array(h5["delays_mask"][:])
            print("delays_mask", delays_mask)

            quadsN = np.array([h5["quadsN_0"][:], h5["quadsN_1"][:], h5["quadsN_2"][:], h5["quadsN_3"][:]])
            quadsN = np.swapaxes(quadsN,0,2) # shuffle the quads to the back
            quadsN = np.swapaxes(quadsN,0,1) # swap the index and radius
            print("quadsN", quadsN.shape)

            try:
                stage_t0 = np.array(h5["stage_t0"][:])
                print("stage_t0", stage_t0)
            except:
                stage_t0 = np.array([161.423])
                print("WARNING: stage_t0 not found in h5 file, will assume default value:", stage_t0)

            #quads_dict = {}
            self.delays_dict   = result
            self.delays_unique = delays_unique
            self.delays_mask   = delays_mask
            self.quadsN        = quadsN
            self.stage_t0      = stage_t0
            
        self.set_params()
        #return quads_dict

        
    def set_params(self):
        self.maxRadius = self.quadsN.shape[1]
        self.useQuadrants = (0,1,2,3)
        self.meanquadsN = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))
        self.stdquadsN  = np.empty(( len(self.delays_unique), self.maxRadius, len(self.useQuadrants)  ))
        self.quadrants_stats()

                
    def quadrants_stats(self):
        """axis=0 is taking mean over data with same delay step. Example dataset has 364 images but only 31 unique delay steps"""
        print("FUNCTION quadrants_stats() called")
        
        for idx, delay_unique in enumerate(self.delays_unique):
            #print(idx, delay_unique)
            acq_idxs = self.delays_dict[delay_unique] # for position 161.273105 returns [0,1,2,3]  etc. 
            hit_idxs = np.intersect1d(acq_idxs, self.delays_mask)

            print(f"{delay_unique}, \t{[self.quadsN[hit_idxs][:,:,quad].shape for quad in [0,1,2,3]]}")

            self.meanquadsN[idx] = np.nanmean(self.quadsN[hit_idxs],axis=0)
            self.stdquadsN[idx]  = np.nanstd (self.quadsN[hit_idxs],axis=0)



