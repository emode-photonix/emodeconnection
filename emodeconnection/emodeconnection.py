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

import socket, struct, klepto
from subprocess import Popen, PIPE
import numpy as np

class EMode:
    def __init__(self, sim="emode"):
        '''
        Initialize defaults and connects to EMode.
        '''
        self.ext = ".eph"
        self.sim = sim
        self.DL = 1024
        self.HOST = '127.0.0.1'
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.HOST, 0))
        self.PORT_SERVER = int(self.s.getsockname()[1])
        self.s.listen(1)
        proc = Popen(['EMode.exe', str(self.PORT_SERVER)])
        self.conn, self.addr = self.s.accept()
        self.conn.sendall(b"connected!")
        self.call("EM_init", sim = self.sim)
    
    def call(self, function, **kwargs):
        '''
        Send a command to EMode.
        '''
        sendset = []
        if (isinstance(function, str)):
            sendset.append(function.encode('utf_8'))
        else:
            raise TypeError("input parameter 'function' must be a string")
        
        for kw in kwargs:
            sendset.append(kw.encode('utf_8'))
            if (isinstance(kwargs[kw], str)):
                sendset.append(kwargs[kw].encode('utf_8'))
            elif (isinstance(kwargs[kw], list)):
                sendset.append(struct.pack('@%dd' % int(len(kwargs[kw])), *kwargs[kw]))
            elif (isinstance(kwargs[kw], (int, float, np.int64, np.float64))):
                sendset.append(struct.pack('@1d', kwargs[kw]))
            else:
                raise TypeError("type not recognized in '**kwargs' as str, list, int, float, np.int64, or np.float64")
        
        sendstr = b':'.join(sendset)
        self.conn.sendall(sendstr)
        RV = self.conn.recv(self.DL)
        if (RV.decode("utf_8").split(":")[0] == "sim"):
            self.dsim = RV.decode("utf_8").split(":")[1]
        
        return RV.decode("utf_8")

    def get(self, variable):
        '''
        Return data from simulation file.
        '''
        if (not isinstance(variable, str)):
            raise TypeError("input parameter 'variable' must be a string")
        
        f = klepto.archives.file_archive(self.dsim+self.ext)
        f.load()
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
        f = klepto.archives.file_archive(self.dsim+self.ext)
        f.load()
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
