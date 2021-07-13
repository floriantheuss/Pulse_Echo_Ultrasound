import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from copy import copy
import os




class AnalyzeData:


    def __init__ (self, filename, sweep_type, Bmin=-np.inf, Bmax=np.inf, Tmin=0, Tmax=np.inf):
        """
        filename: path of the data file
        sweep_type: string, can be "temperature" or "field"
        """
        self.filename = filename

        self.sweep_type = sweep_type
        self.text = None # text for plots etc. giving the average field, temp (for temp, field sweep respecitvely)

        # initialize upper and lower bounds for magnetic field and temperature
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.Tmin = Tmin
        self.Tmax = Tmax

        self.good_index_list = None

        self.alpha           = None    # attenuation


        # initialize data variables
        self.all_data       = None
        self.num_cursor     = None
        self.time           = None
        self.temp           = None
        self.temp2          = None
        self.field          = None
        self.field2         = None
        self.dt             = None
        self.dv0            = None
        self.amplitude0     = None
        self.peak_location0 = None
        self.dv             = None
        self.amplitude      = None
        self.peak_location  = None



    def import_data (self):
        all_data = open(self.filename, 'r')
        self.all_data = np.fromfile ( all_data, dtype=np.dtype('float64') )

        self.num_cursor = int(self.all_data[6])
        self.time   = self.all_data[0::(3*self.num_cursor+8)]
        self.temp   = self.all_data[1::(3*self.num_cursor+8)]
        self.temp2  = self.all_data[2::(3*self.num_cursor+8)]
        self.field  = self.all_data[3::(3*self.num_cursor+8)]
        self.field2 = self.all_data[4::(3*self.num_cursor+8)]
        self.dt     = self.all_data[5::(3*self.num_cursor+8)]

        self.dv0            = self.all_data[7::(3*self.num_cursor+8)]
        self.amplitude0     = self.all_data[8::(3*self.num_cursor+8)]
        self.peak_location0 = self.all_data[9::(3*self.num_cursor+8)]

        self.dv            = np.zeros((self.num_cursor-1, len(self.time)))
        self.amplitude     = np.zeros((self.num_cursor-1, len(self.time)))
        self.peak_location = np.zeros((self.num_cursor-1, len(self.time)))
        for nb in np.arange(self.num_cursor-1):
            self.dv[nb, :]            = self.all_data[10+3*nb::(3*self.num_cursor+8)]
            self.amplitude[nb, :]     = self.all_data[11+3*nb::(3*self.num_cursor+8)]
            self.peak_location[nb, :] = self.all_data[12+3*nb::(3*self.num_cursor+8)]

        if self.sweep_type == 'temperature':
            B  = round(np.mean(self.field), 2)
            self.text = 'B = ' + str(B) + ' T'
        elif self.sweep_type == 'field':
            T  = round(np.mean(self.temp), 2)
            self.text = 'T = ' + str(T) + ' K'

        self.good_index_list = np.arange(self.num_cursor-1)

        return (0)

    def mask_data (self):
        maskT = (self.temp>=self.Tmin) & (self.temp<=self.Tmax)
        maskB = (self.field>=self.Bmin) & (self.field<=self.Bmax)
        mask = maskT & maskB

        self.time   = self.time  [mask]
        self.temp   = self.temp  [mask]
        self.temp2  = self.temp2 [mask]
        self.field  = self.field [mask]
        self.field2 = self.field2[mask]
        self.dt     = self.dt    [mask]

        self.dv0            = self.dv0           [mask]
        self.amplitude0     = self.amplitude0    [mask]
        self.peak_location0 = self.peak_location0[mask]

        self.dv            = np.zeros((self.num_cursor-1, len(self.time)))
        self.amplitude     = np.zeros((self.num_cursor-1, len(self.time)))
        self.peak_location = np.zeros((self.num_cursor-1, len(self.time)))
        for nb in np.arange(self.num_cursor-1):
            self.dv[nb, :]            = self.all_data[10+3*nb::(3*self.num_cursor+8)][mask]
            self.amplitude[nb, :]     = self.all_data[11+3*nb::(3*self.num_cursor+8)][mask]
            self.peak_location[nb, :] = self.all_data[12+3*nb::(3*self.num_cursor+8)][mask]


    def find_good_echoes (self, threshhold=10):
        self.good_index_list = []
        maskT = (self.temp>=self.Tmin) & (self.temp<=self.Tmax)
        maskB = (self.field>=self.Bmin) & (self.field<=self.Bmax)
        mask = maskT & maskB
        for idx in np.arange(self.num_cursor-1):
            gradient = np.gradient(self.peak_location[idx])[mask]
            if max(abs(gradient)) < threshhold:
                self.good_index_list.append(idx)

        # self.dv            = self.dv           [self.good_index_list]
        # self.peak_location = self.peak_location[self.good_index_list]
        # self.amplitude     = self.amplitude    [self.good_index_list]
        # self.num_cursor = len(self.dv)+1
        return (self.good_index_list)


    def exponential_decay (self, t, alpha, A):
        return A*np.exp(-.5*alpha*t)


    def calculate_attenuation (self, good_echoes=None, threshold=10):
        if good_echoes is None:
            self.find_good_echoes(threshhold=threshold)
            good_echoes = self.good_index_list
            print (good_echoes)
        self.alpha = np.zeros(len(self.temp))

        for idx in np.arange(len(self.temp)):
            t = [0] + list( (self.peak_location[good_echoes][:,idx] - self.peak_location0[idx])*self.dt[0] )
            amp = [self.amplitude0[idx]] + list(self.amplitude[good_echoes][:,idx])

            popt, _ = curve_fit(self.exponential_decay, t, amp, p0=[1/5e-6,amp[0]])
            self.alpha[idx] = popt[0]
        print (good_echoes)

        return (self.alpha)


    def plot_stuff (self):
        if self.sweep_type == 'temperature':
            xdata = self.temp
            xlabel = 'T (K)'
        elif self.sweep_type == 'field':
            xdata = self.field
            xlabel = 'B (T)'
        else:
            print ('you have not specified a valid sweep type')
            print ('valid sweep types are "temperature" or "field"')
        # plot delta v/v
        plt.figure()
        for idx in np.arange(self.num_cursor-1):
            if idx in self.good_index_list:
                plt.plot(xdata, self.dv[idx]*100, label=str(idx))
            else:
                plt.plot(xdata, self.dv[idx]*100, '--', label=str(idx)+' bad')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel('$\\Delta v/v$ (%)')
        plt.title(self.text)

        # plot peak location
        plt.figure()
        for idx in np.arange(self.num_cursor-1):
            if idx in self.good_index_list:
                plt.plot(xdata, self.peak_location[idx], label=str(idx))
            else:
                plt.plot(xdata, self.peak_location[idx], '--', label=str(idx)+' bad')
                # plt.scatter(xdata, self.peak_location[idx], label=str(idx)+' bad')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel('peak location')
        plt.title(self.text)

        # plot attenuation
        plt.figure()
        plt.plot(xdata, self.alpha)
        plt.xlabel('T (K)')
        plt.ylabel('attenuation')
        plt.title(self.text)





################################################################################################
################################################################################################


if __name__ == "__main__":

    path = 'C:\\Users\\Doodleous\\Dropbox\\Mn3Ge2104D\\Temperature\\data_7_1_2021\\run_4_54_51_pm_199MHz\\peak199MHz.bin'
    path = 'C:\\Users\\Doodleous\\Dropbox\\Mn3Ge2104D\\Field\\data_7_2_2021\\run_12_50_36_pm_199MHz\\peak199MHz.bin'
    folder = 'C:\\Data\\Florian\\Mn3Ge_2104D\\temperature_sweeps'

    filenames = np.array( [ folder+'\\'+i for i in os.listdir(folder) ] )

    ################################################################################################################################
    # temperature sweeps
    ################################################################################################################################

    plt.figure(dpi=200)
    lower_limits = np.zeros(len(filenames))
    lower_limits[:2] = np.array([368.9, 368.94])
    idx = 0
    for name in filenames[:-5]:
        dat = AnalyzeData(filename=name, sweep_type='temperature')
        dat.import_data()
        # plt.plot(dat.temp[dat.temp>lower_limits[idx]], dat.dv[0][dat.temp>lower_limits[idx]]*100, label=dat.text)
        dv_v = dat.dv[0][dat.temp>lower_limits[idx]]
        c66 = (139.702 - 44.414) / 2
        deltaC = 2*dv_v * c66
        plt.plot(dat.temp[dat.temp>lower_limits[idx]], deltaC, label=dat.text)
        idx+=1

    plt.legend(fontsize=10, loc=(0.7,0.05), frameon=False, fancybox=True, edgecolor='black')
    plt.xlabel('T (K)', fontsize=13)
    # plt.ylabel('$\\Delta v/v$ (%)', fontsize=13)
    plt.ylabel('$\\Delta c$ (GPa)', fontsize=13)

    plt.tick_params(axis='both', direction='in', labelsize=10, bottom='True', top='True', left='True', right='True', length=4, width=1, which='major')



    plt.figure(dpi=200)
    idx = 0
    for name in filenames[:-3]:
        dat = AnalyzeData(filename=name, sweep_type='temperature', Tmin=370)
        dat.import_data()
        dat.calculate_attenuation(threshold=10, good_echoes=dat.good_index_list)
        plt.plot(dat.temp[dat.temp>lower_limits[idx]], dat.alpha[dat.temp>lower_limits[idx]], label=dat.text)
        idx+=1

    plt.legend(fontsize=10, loc=(0.7,0.05), frameon=False, fancybox=True, edgecolor='black')
    plt.xlabel('T (K)', fontsize=13)
    plt.ylabel('attenuation', fontsize=13)

    plt.tick_params(axis='both', direction='in', labelsize=10, bottom='True', top='True', left='True', right='True', length=4, width=1, which='major')



    # plt.figure(dpi=200)
    # lower_limits = np.array([368.88, 368.88, 367, 364, 300])
    # lower_limits = np.array([368.88, 367, 364, 300])
    # idx=0
    # for name in filenames[-4:]:
    #     dat = AnalyzeData(filename=name, sweep_type='temperature')
    #     dat.import_data()
    #     # plt.plot(dat.temp[dat.temp>lower_limits[idx]], dat.dv[index[idx]][dat.temp>lower_limits[idx]]*100, label=dat.text)
    #     dv_v = dat.dv[0][dat.temp>lower_limits[idx]]
    #     c66 = (139.702 - 44.414) / 2
    #     deltaC = 2*dv_v * c66
    #     plt.plot(dat.temp[dat.temp>lower_limits[idx]], deltaC, label=dat.text)
    #     idx+=1

    # plt.legend(fontsize=10, loc=(0.7,0.05), frameon=False, fancybox=True, edgecolor='black')
    # plt.xlabel('T (K)', fontsize=13)
    # # plt.ylabel('$\\Delta v/v$ (%)', fontsize=13)
    # plt.ylabel('$\\Delta c$ (GPa)', fontsize=13)

    # plt.tick_params(axis='both', direction='in', labelsize=10, bottom='True', top='True', left='True', right='True', length=4, width=1, which='major')



    # plt.figure(dpi=200)
    # idx = 0
    # for name in filenames[-4:]:
    #     dat = AnalyzeData(filename=name, sweep_type='temperature', Tmin=370)
    #     dat.import_data()
    #     dat.calculate_attenuation(threshold=10, good_echoes=dat.good_index_list)
    #     plt.plot(dat.temp[dat.temp>lower_limits[idx]], dat.alpha[dat.temp>lower_limits[idx]], label=dat.text)
    #     idx+=1

    # plt.legend(fontsize=10, loc=(0.7,0.05), frameon=False, fancybox=True, edgecolor='black')
    # plt.xlabel('T (K)', fontsize=13)
    # plt.ylabel('attenuation', fontsize=13)

    # plt.tick_params(axis='both', direction='in', labelsize=10, bottom='True', top='True', left='True', right='True', length=4, width=1, which='major')


    ################################################################################################################################
    # field sweeps
    ################################################################################################################################

    # folder = 'C:\\Data\\Florian\\Mn3Ge_2104D\\field_sweeps\\around_Tc'
    # # folder = 'C:\\Data\\Florian\\Mn3Ge_2104D\\field_sweeps'
    # filenames = np.array( [ folder+'\\'+i for i in os.listdir(folder) ] )

    # plt.figure(dpi=200)

    # idx = 0
    # for name in filenames:#[filenames[1]]:
    #     dat = AnalyzeData(filename=name, sweep_type='field')
    #     dat.import_data()
    #     plt.plot(dat.field, dat.dv[1]*100, label=dat.text)
    #     # plt.plot(dat.field, dat.temp, label=dat.text)
    #     idx+=1

    # plt.legend(fontsize=10, loc=(0.7,0.05), frameon=False, fancybox=True, edgecolor='black')
    # plt.xlabel('B (T)', fontsize=13)
    # # plt.ylabel('$\\Delta v/v$ (%)', fontsize=13)
    # plt.ylabel('T (K)', fontsize=13)

    # plt.tick_params(axis='both', direction='in', labelsize=10, bottom='True', top='True', left='True', right='True', length=4, width=1, which='major')



    # plt.figure(dpi=200)
    # idx = 0
    # for name in filenames:
    #     dat = AnalyzeData(filename=name, sweep_type='field')
    #     dat.import_data()
    #     # if idx == 0:
    #         # dat.calculate_attenuation(good_echoes=dat.good_index_list)
    #     # else:
    #         # dat.calculate_attenuation()
    #     dat.calculate_attenuation(good_echoes=dat.good_index_list)
    #     plt.plot(dat.field, dat.alpha, label=dat.text)
    #     idx+=1

    # plt.legend(fontsize=10, loc=(0.7,0.05), frameon=False, fancybox=True, edgecolor='black')
    # # plt.xlabel('T (K)', fontsize=13)
    # plt.xlabel('B (T)', fontsize=13)
    # plt.ylabel('attenuation', fontsize=13)

    # plt.tick_params(axis='both', direction='in', labelsize=10, bottom='True', top='True', left='True', right='True', length=4, width=1, which='major')



    # file = 'C:\\Users\\Doodleous\\Dropbox\\Mn3Ge2104D\\Field\\data_7_7_2021\\run_4_02_24_pm_199MHz\\peak199MHz.bin'

    # dat = AnalyzeData(filename=file, sweep_type='field')
    # dat = AnalyzeData(filename=filenames[-1], sweep_type='temperature', Tmin=370)
    # dat.import_data()
    # dat.calculate_attenuation(threshold=10)#, good_echoes=dat.good_index_list)
    # print (len(dat.temp))
    # dat.plot_stuff()




    plt.show()