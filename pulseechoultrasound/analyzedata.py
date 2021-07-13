import numpy as np
import matplotlib.pyplot as plt
import os




class AnalyzeData:


    def __init__ (self, filename, sweep_type):
        """
        filename: path of the data file
        sweep_type: string, can be "temperature" or "field"
        """
        self.filename = filename

        self.sweep_type = sweep_type
        self.text = None # text for plots etc. giving the average field, temp (for temp, field sweep respecitvely)


        # initialize data variables
        self.num_cursor     = None
        self.time           = None
        self.temp           = None
        self.temp2          = None
        self.field          = None
        self.field2         = None
        self.dt             = None
        self.dv0            = None
        self.voltage_ratio0 = None
        self.peak_location0 = None
        self.dv             = None
        self.voltage_ratio  = None
        self.peak_location  = None



    def import_data (self):
        all_data = open(self.filename, 'r')
        all_data = np.fromfile ( all_data, dtype=np.dtype('float64') )

        self.num_cursor = int(all_data[6])
        self.time   = all_data[0::(3*self.num_cursor+8)]
        self.temp   = all_data[1::(3*self.num_cursor+8)]
        self.temp2  = all_data[2::(3*self.num_cursor+8)]
        self.field  = all_data[3::(3*self.num_cursor+8)]
        self.field2 = all_data[4::(3*self.num_cursor+8)]
        self.dt     = all_data[5::(3*self.num_cursor+8)]

        self.dv0            = all_data[7::(3*self.num_cursor+8)]
        self.voltage_ratio0 = all_data[8::(3*self.num_cursor+8)]
        self.peak_location0 = all_data[9::(3*self.num_cursor+8)]

        self.dv            = np.zeros((self.num_cursor-1, len(self.time)))
        self.voltage_ratio = np.zeros((self.num_cursor-1, len(self.time)))
        self.peak_location = np.zeros((self.num_cursor-1, len(self.time)))
        for nb in np.arange(self.num_cursor-1):
            self.dv[nb, :]            = all_data[10+3*nb::(3*self.num_cursor+8)]
            self.voltage_ratio[nb, :] = all_data[11+3*nb::(3*self.num_cursor+8)]
            self.peak_location[nb, :] = all_data[12+3*nb::(3*self.num_cursor+8)]

        if self.sweep_type == 'temperature':
            B  = round(np.mean(self.field), 2)
            self.text = 'B = ' + str(B) + ' T'
        elif self.sweep_type == 'field':
            T  = round(np.mean(self.temp), 2)
            self.text = 'T = ' + str(T) + ' K'


        return (0)


    def choose_good_echoes (self, threshold=1, reference_index=0):
        good_index_list = []
        for idx in np.arange(self.num_cursor-1):
            if idx == reference_index:
                good_index_list.append(idx)




################################################################################################
################################################################################################


if __name__ == "__main__":

    path = 'C:\\Users\\Doodleous\\Dropbox\\Mn3Ge2104D\\Temperature\\data_7_1_2021\\run_4_54_51_pm_199MHz\\peak199MHz.bin'
    folder = 'C:\\Data\\Florian\\Code\\Mn3Ge_2104D\\temperature_sweeps'

    filenames = np.array( [ folder+'\\'+i for i in os.listdir(folder) ] )


    plt.figure(dpi=200)
    lower_limits = np.zeros(len(folder))
    lower_limits[:2] = np.array([368.9, 368.94])
    idx = 0
    for name in filenames:
        dat = AnalyzeData(filename=name, sweep_type='temperature')
        dat.import_data()
        plt.plot(dat.temp[dat.temp>lower_limits[idx]], dat.dv[0][dat.temp>lower_limits[idx]]*100, label=dat.text)
        idx+=1

    plt.legend(fontsize=10, loc=(0.7,0.05), frameon=False, fancybox=True, edgecolor='black')
    plt.xlabel('T (K)', fontsize=13)
    plt.ylabel('$\\Delta v/v$ (%)', fontsize=13)

    plt.tick_params(axis='both', direction='in', labelsize=10, bottom='True', top='True', left='True', right='True', length=4, width=1, which='major')

    plt.show()