###########################################################
###########################################################
## EMode - Python interface, by EMode Photonix LLC
###########################################################
## Copyright (c) 2022 EMode Photonix LLC
###########################################################

import os, socket, json, pickle, time, atexit
from subprocess import Popen
import numpy as np
import scipy.io as sio

class EMode:
    def __init__(self, sim='emode', open_existing=False, new_name=False, priority='pN', roaming=False, verbose=False):
        '''
        Initialize defaults and connects to EMode.
        '''
        atexit.register(self.close)
        try:
            sim = str(sim)
        except:
            raise TypeError("input parameter 'sim' must be a string")
            return
        try:
            priority = str(priority)
        except:
            raise TypeError("input parameter 'priority' must be a string")
            return
        self.dsim = sim
        self.ext = ".eph"
        self.exit_flag = False
        self.DL = 2048
        self.HOST = '127.0.0.1'
        self.LHOST = 'lm.emodephotonix.com'
        self.LPORT = '64000'
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.HOST, 0))
        self.PORT_SERVER = int(self.s.getsockname()[1])
        self.s.listen(1)
        cmd_lst = ['EMode.exe', self.LHOST, self.LPORT, str(self.PORT_SERVER)]
        if (verbose == True):
            cmd_lst.append('-v')
        if (priority != 'pN'):
            priority = priority.strip('-')
            cmd_lst.append('-'+priority)
        if roaming:
            cmd_lst.append('-r')
        proc = Popen(cmd_lst, stderr=None)
        self.conn, self.addr = self.s.accept()
        time.sleep(0.1) # wait for EMode to recv
        self.conn.sendall(b"connected with Python!")
        if (open_existing):
            RV = self.call("EM_open", sim=sim, new_name=new_name)
        else:
            RV = self.call("EM_init", sim=sim)
        if (RV == 'ERROR'):
            raise RuntimeError("internal EMode error")
        self.dsim = RV[len("sim:"):]
        return
    
    def call(self, function, **kwargs):
        '''
        Send a command to EMode.
        '''
        sendset = {}
        if (isinstance(function, str)):
            sendset['function'] = function
        else:
            raise TypeError("input parameter 'function' must be a string")
        
        for kw in kwargs:
            data = kwargs[kw]
            if (type(data).__module__ == np.__name__):
                data = np.squeeze(data).tolist()
            
            if (isinstance(data, list)):
                if (len(data) == 1):
                    data = data[0]
            
            sendset[kw] = data
        
        if ('sim' not in kwargs):
            sendset['sim'] = self.dsim
        
        try:
            sendstr = json.dumps(sendset)
        except TypeError:
            raise TypeError("EMode function inputs must have type string, int/float, or list")
        
        try:
            self.conn.sendall(bytes(sendstr, encoding="utf-8"))
            recvstr = self.conn.recv(self.DL)
        except:
            # Exited due to license checkout
            self.conn.close()
            self.exit_flag = True
        
        if (self.exit_flag):
            raise RuntimeError("License checkout error!")
        
        recvjson = recvstr.decode("utf-8")
        recvset = json.loads(recvjson)
        
        return recvset['RV']

    def get(self, variable):
        '''
        Return data from simulation file.
        '''
        if (not isinstance(variable, str)):
            raise TypeError("input parameter 'variable' must be a string")
        
        RV = self.call("EM_save", sim=self.dsim)
        
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
        RV = self.call("EM_save", sim=self.dsim)
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
        if (self.conn.fileno() == -1): return
        self.call("EM_close", **kwargs)
        sendjson = json.dumps({'function': 'exit'})
        self.conn.sendall(bytes(sendjson, encoding="utf-8"))
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
