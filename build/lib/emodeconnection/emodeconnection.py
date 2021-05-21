###########################################################
###########################################################
## EMode - Python interface, by EMode Photonix LLC
###########################################################
## Copyright (c) 2021 EMode Photonix LLC
###########################################################
## NOTES:
## - strings are UTF-8
## - numbers are doubles with IEEE 754 binary64
###########################################################
###########################################################

import os, socket, struct, pickle, time
from subprocess import Popen, PIPE
import numpy as np
import scipy.io as sio

class EMode:
    def __init__(self, sim='emode'):
        '''
        Initialize defaults and connects to EMode.
        '''
        self.ext = ".eph"
        self.exit_flag = False
        self.dsim = sim
        self.DL = 2048
        self.HOST = '127.0.0.1'
        self.LHOST = '67.205.182.231'
        self.LPORT = '64000'
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.HOST, 0))
        self.PORT_SERVER = int(self.s.getsockname()[1])
        self.s.listen(1)
        proc = Popen(['EMode.exe', self.LHOST, self.LPORT, str(self.PORT_SERVER)], stderr=None)
        self.conn, self.addr = self.s.accept()
        time.sleep(0.2) # wait for EMode to recv
        self.conn.sendall(b"connected!")
        self.call("EM_init")  
        return
    
    def call(self, function, **kwargs):
        '''
        Send a command to EMode.
        '''
        sendset = []
        if (isinstance(function, str)):
            sendset.append(function.encode('utf-8'))
        else:
            raise TypeError("input parameter 'function' must be a string")
        
        for kw in kwargs:
            sendset.append(kw.encode('utf-8'))
            if (isinstance(kwargs[kw], str)):
                if ((len(kwargs[kw]) % 8) == 0):
                    kwargs[kw] = ' '+kwargs[kw]
                sendset.append(kwargs[kw].encode('utf-8'))
            elif (isinstance(kwargs[kw], list)):
                sendset.append(struct.pack('@%dd' % int(len(kwargs[kw])), *kwargs[kw]))
            elif (isinstance(kwargs[kw], (int, float, np.integer, np.float))):
                sendset.append(struct.pack('@1d', kwargs[kw]))
            else:
                raise TypeError("type not recognized in '**kwargs' as str, list, intrger, or float")
        
        if ('sim' not in kwargs):
            sendset.append('sim'.encode('utf-8'))
            sendset.append(self.dsim.encode('utf-8'))
        
        sendstr = b':::::'.join(sendset)
        try:
            self.conn.sendall(sendstr)
            RV = self.conn.recv(self.DL)
        except:
            # Exited due to license checkout
            self.conn.close()
            self.exit_flag = True
        
        if (self.exit_flag):
            raise RuntimeError("License checkout error!")
        
        return RV.decode("utf-8")

    def get(self, variable):
        '''
        Return data from simulation file.
        '''
        if (not isinstance(variable, str)):
            raise TypeError("input parameter 'variable' must be a string")
        
        fl = open(self.dsim+self.ext, 'rb')
        f = pickle.load(fl)
        fl.close()
        if (variable in list(f.keys())):
            data = f[variable]
        else:
            print("Data does not exist.")
            return
        
        return data
    
    def inspect(self):
        '''
        Return list of keys from available data in simulation file.
        '''
        fl = open(self.dsim+self.ext, 'rb')
        f = pickle.load(fl)
        fl.close()
        fkeys = list(f.keys())
        fkeys.remove("EMode_simulation_file")
        return fkeys
    
    def close(self, **kwargs):
        '''
        Send saving options to EMode and close the connection.
        '''
        self.call("EM_close", **kwargs)
        self.conn.sendall(b"exit")
        self.conn.close()
        print("Exited EMode")
        return

def open_file(sim):
    '''
    Opens an EMode simulation file with either .eph or .mat extension.
    '''
    ext = '.eph'
    mat = '.mat'
    found = False
    for file in os.listdir():
        if ((file == sim+ext) or ((file == sim) and (sim.endswith(ext)))):
            found = True
            if (sim.endswith(ext)):
                sim = sim.replace(ext,'')
            fl = open(sim+ext, 'rb')
            f = pickle.load(fl)
            fl.close()
        elif ((file == sim+mat) or ((file == sim) and (sim.endswith(mat)))):
            found = True
            f = sio.loadmat(sim+mat)
    
    if (not found):
        print("ERROR: file not found!")
        return "ERROR"
    
    return f

def get(variable, sim='emode'):
    '''
    Return data from simulation file.
    '''
    if (not isinstance(variable, str)):
        raise TypeError("input parameter 'variable' must be a string")
    
    if (not isinstance(sim, str)):
        raise TypeError("input parameter 'sim' must be a string")
    
    f = open_file(sim=sim)
    
    if (variable in list(f.keys())):
        data = f[variable]
    else:
        print("Data does not exist.")
        return
    
    return data

def inspect(sim='emode'):
    '''
    Return list of keys from available data in simulation file.
    '''
    if (not isinstance(sim, str)):
        raise TypeError("input parameter 'sim' must be a string")
    
    f = open_file(sim=sim)
    
    fkeys = list(f.keys())
    fkeys.remove("EMode_simulation_file")
    return fkeys
