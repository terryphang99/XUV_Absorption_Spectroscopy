import holoviews as hv
from holoviews import opts, dim, streams
import numpy as np
import panel as pn
import param
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
import xarray as xr
from PIL import Image

from utils.math import wl2e,d2t

hv.extension('bokeh')


class data_proc(param.Parameterized):
    
    select_scans = param.Range(default = (1,50))
    bkg_pump_off_scans = param.Range(default = (1,50))#,bounds = (1,2000))
    bkg_pump_on_scans = param.Range(default = (1,50))
    abs_scans = param.Range(default = (50,100))#,bounds = (1,2000))
    pump_off_first = param.Selector(objects =['yes','no'])
    
    def __init__(self,folder,**params):
        super().__init__(**params)
        self.folder = folder

    @param.depends('select_scans',
                   'bkg_pump_off_scans',
                   'bkg_pump_on_scans',
                   'abs_scans',
                   'pump_off_first'
                  )
    
    def single(self,n):
        """
        Returns nth scan in a data folder
        """
        scan = self.folder+"\Scan"+str(n)+".bin"
        scan = np.fromfile(scan,dtype = 'uint32')
        return scan
        

    def avg_data(self,a,b):
        """
        Takes the average of all spectra between the a-th and the b-th scan
        """
        data = []
        for i in range(int(a),int(b)+1):
            path = self.folder+"\Scan"+str(i)+".bin"
            contents = np.fromfile(path,dtype = 'uint32')
            data.append(contents)   
        avg_data = np.mean(data, axis = 0)
        return avg_data
    
    def avg_bkg(self,a,b,c,d):
        """
        Calculates the average background spectrum.
        - background off: select background spectra between (a,b)
        - background on: select background spectra between (c,d)
        """
        off = self.avg_data(a,b)
        on = self.avg_data(c,d)
        return off,on
    
#     def timedelay(self,a,b):
        
        

class energy_calib(data_proc):
    def __init__(self,folder,**params):
        super().__init__(folder,**params)
        
    def plot_cc(self,a,b):
        """
        Plots energy calibration curve in holoviews for interactive peak identification.
        Enter range (a,b) of calibration spectra for given element.
        """
        spec1 = a
        spec2 = b
        cc = hv.Curve(self.avg_data(spec1,spec2))
        cc = cc.opts(height = 300,
                    width = 400,
                    xlabel = 'pixel #',
                    ylabel = 'Counts')
        return cc
    
    def calibrate(self,arrpix1,arrwl1,arrpix2,arrwl2):
        """
        Performs energy calibration given four arrays.
        - arrpix1: array of peak pixels for 1st element
        - arrpix2: array of peak pixels for 2nd element
        - arrpix1: array of peak energies for 1st element
        - arrpix2: array of peak energies for 2nd element
        """
        c1pix = arrpix1
        c1E = arrwl1
        c2pix = arrpix2
        c2E = arrwl2
        pix_rng = np.arange(0,1342,1)
        pixels = np.concatenate((c1pix,c2pix))
        energy = np.concatenate((c1E,c2E))
        slope, intercept, r_value, p_value, std_err = stats.linregress(pixels,energy)
        def energy_calib(x):
            return slope*x+intercept
        fit = slope*pix_rng+intercept
        energy_data = wl2e(energy_calib(pix_rng))
        return pixels,energy,fit,energy_data
    
class raster_scan(data_proc):
    def __init__(self,folder,**params):
        super().__init__(folder,**params)
        
    def raster(self,a,b,c):
        """
        Plots raster scan given range of raster scan files (a,b), 
        and c, the number of pixels in a row of the sample.
        """
        arr = []
        ras = []
        ras_map = []
        for i in range(a,b):
            scan = self.folder+"\Scan"+str(i)+".bin"
            scan = np.fromfile(scan,dtype = 'uint32')
            arr.append(scan)
        
        for i in range(0,len(arr)):
            point = np.sum(arr[i])
            ras.append(point)
        
        ras_map = [ras[i:i+c] for i in range(0, len(ras)-1, c)]
        ras_map = np.asarray(ras_map)
        return ras_map

    
class characterize(data_proc):
    def __init__(self,folder,**params):
        super().__init__(folder,**params)
    
    def auger(self):
        """
        Plots Auger Electron Spectrum.
        No inputs except data file needed.
        """
        data = np.loadtxt(self.folder,skiprows = 1)
        KE,AES = np.transpose(data)
        return KE,AES
    
    def raman(self):
        """
        Plots Raman Spectrum.
        No inputs except data file needed.
        """
        data = np.loadtxt(self.folder,skiprows = 0)
        WN,RS = np.transpose(data)
        return WN,RS
    
    def atomicforce(self):
        """
        Plots Atomic Force Microscopy results
        No inputs except data file needed.
        """
        data = Image.open(self.folder)
        AFM = np.array(data)
        return AFM
    
class static_abs(data_proc):
    def __init__(self,folder,**params):
        super().__init__(folder,**params)

#     @param.depends('data_slider','abs_scans','pump_off_first')
    def static(self,a,b,c,d,e,f,off_first = 'yes'):
        """
        Plots static absorption spectrum given the following:
        - (a,b): range of spectrum data files for sas
        - (c,d): range of background off data files
        - (e,f): range of background on data files
        - off_first: enter 'yes' if file sequence starts with
          pump off.
        """
        scans = np.arange(a,b+1)
        odd = []
        even = []
        for i,scans in enumerate(scans,start = 1):
            path = self.folder+"\Scan"+str(i)+".bin"
            plot = np.fromfile(path,dtype = 'uint32')
            if i%2 == 0:
                even.append(plot)
            else:
                odd.append(plot)
        
        bkg_off,bkg_on = self.avg_bkg(c,d,e,f)
                
        if off_first == 'yes':
            avg_I0 = np.mean(odd, axis = 0)-bkg_off
            avg_I = np.mean(even, axis = 0)-bkg_on
            trans = avg_I0/avg_I
        else:
            avg_I0 = np.mean(even, axis = 0)-bkg_off
            avg_I = np.mean(odd, axis = 0)-bkg_on
            trans = avg_I0/avg_I
        
        abs_spec = np.log10(trans)
        return abs_spec
    
class trans_abs(data_proc):
    def __init__(self,folder,**params):
        super().__init__(folder,**params)

#     @param.depends('data_slider','abs_scans','pump_off_first')
    def transient(self,a,b,c,d,e,f,gr,step,totalscans,heat,off_first = 'yes'):
        """
        Plots transient absorption spectrum colormap given the following
        parameters:
        - (a,b): range of spectrum data files for tas
        - (c,d): range of background off data files
        - (e,f): range of background on data files
        - gr: array of gas reference tuples
        - step: number of scan files in one scan step, including gas ref
        - totalscans = total number of scans in set
        - heat: number of scans near t0 to average for subtracting heat signal
        - off_first: enter 'yes' if file sequence starts with
          pump off.
        """
        gr_low = [gr[i][0] for i in range(len(gr))]
        gr_high = [gr[i][1] for i in range(len(gr))]
#         group = []
#         tally = 0
#         gas_ref = []
#         store = 0
#         for i in range(0,int((b-a)/step)):
#             for j in range(0,step+1):
#                 path = self.folder+"\Scan"+str(j+(step+1)*i+a)+".bin"
#                 plot = np.fromfile(path,dtype = 'uint32')
#                 group.append(plot)
#         group = np.asarray(group)
#         for i in range(0,len(gr_low)):
#             low = gr_low[tally]
#             high = gr_high[tally]
#             gas_ref.append(group[low-store:high-store+1])
#             group = np.delete(group,np.s_[low-store:high-store],axis = 0)
#             store += len(gas_ref[i])          
#             tally += 1
        
        
#         newgroup = np.reshape(group,(totalscans+1,step+1,1342))
        
        gr_tally = 0
        group = []
        gas_ref = []
        for i in range(0,totalscans+1):
            j = 0
            while j in range(0,step+1):
                s = int(j+(step+1)*i+a)
                if s not in range(gr_low[gr_tally],gr_high[gr_tally],1):
                    path = self.folder+"\Scan"+str(s)+".bin"
                    plot = np.fromfile(path,dtype = 'uint32')
                    group.append(plot)
                    j+=1
                elif s in gr_low:
                    for k in range(gr_low[gr_tally],gr_high[gr_tally]+1):
                        path = self.folder+"\Scan"+str(k)+".bin"
                        plot = np.fromfile(path,dtype = 'uint32')
                        gas_ref.append(plot)
                    j = gr_high[gr_tally]+1
                    gr_tally+=1

#         newgroup = np.reshape(group,(totalscans+1,step+1,1342))
        
#         # For each scan, establish a pump on intensity and a pump off 
#         # intensity (even/odd) for each of the 50 time points
#         master_odd = []
#         master_even = []
#         for i in range(0,len(newgroup)):
#             scans = np.arange(0,len(newgroup[i]))
#             odd = []
#             even = []
#             for j,scans in enumerate(scans, start = 0):
#                 if j%2 == 0:
#                     even.append(newgroup[i][j])
#                 else:
#                     odd.append(newgroup[i][j])
#             master_odd.append(odd)
#             master_even.append(even)
            
#         # Define and subtract background 
#         # Note the change in odds and evens as compared to the static case. This is by design
#         bkg_off,bkg_on = self.avg_bkg(c,d,e,f)

#         if off_first == 'yes':
#             master_odd = [master_odd[i][j]-bkg_on for i in range(len(master_odd)) for j in range(len(master_odd[i]))]
#             master_even = [master_even[i][j]-bkg_off for i in range(len(master_even)) for j in range(len(master_even[i]))]
#             master_odd = np.reshape(master_odd,(totalscans+1,len(odd),1342))
#             master_even = np.reshape(master_even,(totalscans+1,len(even),1342))
#             I0 = np.mean(master_even,axis = 0)
#             I = np.mean(master_odd,axis = 0)
#             trans = np.asarray(I0/I)
#         else:    
#             master_odd = [master_odd[i][j]-bkg_off for i in range(len(master_odd)) for j in range(len(master_odd[i]))]
#             master_even = [master_even[i][j]-bkg_on for i in range(len(master_even)) for j in range(len(master_even[i]))]
#             master_odd = np.reshape(master_odd,(totalscans+1,len(odd),1342))
#             master_even = np.reshape(master_even,(totalscans+1,len(even),1342))
#             I0 = np.mean(master_odd,axis = 0)
#             I = np.mean(master_even,axis = 0)
#             trans = np.asarray(I0/I) 
        
#         # Subtract heat signal
#         abs_spec = np.log10(trans)
#         heatsignal = np.mean(abs_spec[0:heat],axis = 0)
#         t_abs_spec = abs_spec-heatsignal
#         t_abs_spec[np.isnan(t_abs_spec)] = 0
        
        return group,gas_ref,gr_tally,gr_low,gr_high#t_abs_spec,master_odd,master_even,gas_ref
    
    def tas_xarr(self,energy,timedelay,transient):
        title = 'Transient Absorption Spectrum'
        coords = {'E':xr.DataArray(data = energy, name = 'Energy',dims = ['E'], attrs = {'units':'eV'}),
                  'td':xr.DataArray(data = timedelay, name = 'Time Delay',dims = ['td'], attrs = {'units':'fs/ps'})}

        ds_dims = ['td','E']
        
        t_abs_spec = xr.Dataset({'cts' : xr.DataArray(transient, dims=ds_dims, attrs={'long_name':'Counts'})},
                                coords=coords,
                                attrs = {'name':title})
        return t_abs_spec

    def edgeref(self,specOn,specOff,edge):
        if len(specOff.shape)>2:
            dOD = -np.log(np.mean(specOn/specOff,axis=0))
            dODCalib = -np.log(specOff.reshape((specOff.shape[0]*specOff.shape[1],-1))[2::2,:]/specOff.reshape((specOff.shape[0]*specOff.shape[1],-1))[1::2,:])
        else:
            dOD = np.log((specOn/specOff))
            dODCalib = np.log(specOff[2::2,:]/specOff[1::2,:])

        dODEdgeCalib=dODCalib[:,edge]
        dODEdgeCalib[np.isnan(dODEdgeCalib)] = 0
        dODCalib[np.isnan(dODCalib)] = 0
        B = np.linalg.lstsq((dODEdgeCalib.T@dODEdgeCalib),(dODEdgeCalib.T@dODCalib))
        dODRef = dOD - dOD[:,edge]@B[0]
        return dODRef
