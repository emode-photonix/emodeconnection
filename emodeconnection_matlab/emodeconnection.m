%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EMode - Matlab/Octave interface, by EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c) 2021 EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NOTES:
%% - strings are UTF-8
%% - numbers are doubles with IEEE 754 binary64
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef emodeconnection
    properties
        dsim
        ext
        exit_flag
        DL
        HOST
        LHOST
        LPORT
        PORT_SERVER
        s
        conn
    end
    methods
        function obj = emodeconnection(sim, open_existing, new_name)
            % Initialize defaults and connect to EMode.
            
            if nargin == 0
                sim = 'emode';
                open_existing = false;
                new_name = false;
            elseif nargin == 1
                open_existing = false;
                new_name = false;
            elseif nargin == 2
                new_name = false;
            end
            
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
            
            if isOctave
                pkg load sockets;
            end
            
            try
                sim = num2str(sim);
            catch
                error('Input parameter "sim" must be a string.');
                return
            end
            
            obj.dsim = sim;
            obj.ext = '.mat';
            obj.exit_flag = false;
            obj.DL = 2048;
            obj.HOST = '127.0.0.1';
            obj.LHOST = '67.205.182.231';
            obj.LPORT = '64000';
            
            obj.PORT_SERVER = 4000;
            while (1)
                try
                    if isOctave
                        obj.s = socket(AF_INET, SOCK_STREAM);
                        bind(obj.s, obj.PORT_SERVER);
                        a = listen(obj.s, 0);
                    else % Matlab
                        obj.s = tcpserver(obj.HOST, obj.PORT_SERVER);
                    end
                    break
                catch
                    obj.PORT_SERVER = obj.PORT_SERVER + 1;
                end
            end
            
            % Open EMode
            if isOctave
                system(sprintf('EMode.exe %s %s %s', obj.LHOST, obj.LPORT, num2str(obj.PORT_SERVER)), false, 'async');
            else % Matlab
                system(sprintf('EMode.exe %s %s %s &', obj.LHOST, obj.LPORT, num2str(obj.PORT_SERVER)));
            end
            
            obj.conn = accept(obj.s);
            pause(0.2); % wait for EMode to recv
            
            if isOctave
                send(obj.conn, uint8('connected with Octave!'));
            else % Matlab
                send(obj.conn, uint8('connected with Matlab!'));
            end
            
            if open_existing
                RV = obj.call('EM_open', 'sim', sim, 'new_name', new_name);
            else
                RV = obj.call('EM_init', 'sim', sim);
            end
            obj.dsim = RV(length('sim:')+1:end);
        end
        
        function RV = call(obj, func_name, varargin)
            % Send a command to EMode.
            
            sendset = {};
            if (ischar(func_name))
                sendset = [sendset, uint8(func_name)];
            else
                error('Input parameter "function" must be a string.');
            end
            
            sim_flag = true;
            
            if (mod(length(varargin), 2))
                error('Incorrect number of inputs!\nAn even number of inputs is required following the function name.');
                return
            end
            
            for kk = 1:length(varargin)/2
                kw = varargin{kk*2-1};
                kv = varargin{kk*2};
                sendset = [sendset, uint8(kw)];
                if (ischar(kv))
                    if (mod(length(kv), 8) == 0)
                        kv = sprintf(' %s', kv);
                    end
                    sendset = [sendset, uint8(kv)];
                elseif (isa(kv, 'numeric') || isa(kv, 'integer') || isa(kv, 'logical'))
                    if isa(kv, 'logical')
                        kv = double(kv);
                    end
                    for ll = 1:length(kv)
                        tc = typecast(kv(ll), 'uint8');
                        if (ll == 1)
                            tc_set = tc;
                        else
                            tc_set = [tc_set tc];
                        end
                    end
                    sendset = [sendset, tc_set];
                else
                    error('Input type not recognized numeric or integer.');
                end
                
                if (strcmp(kw, 'sim'))
                    sim_flag = false;
                end
            end
            
            if (sim_flag)
                sendset = [sendset, uint8('sim')];
                sendset = [sendset, uint8(obj.dsim)];
            end
            
            sendstr = sendset{1};
            for mm = 2:length(sendset)
                sendstr = [sendstr uint8(':::::') sendset{mm}];
            end
            
            try
                send(obj.conn, sendstr);
                RV = recv(obj.conn, obj.DL);
            catch
                % Exited due to license checkout
                disconnect(obj.conn);
                obj.exit_flag = true;
            end
            
            if (obj.exit_flag)
                error('License checkout error!');
            end
            
            RV = char(RV);
        end

        function data = get(obj, variable)
            % Return data from simulation file.
            
            if (~ischar(variable))
                error('Input parameter 'variable' must be a string.');
            end
            
            obj.call('EM_save_mat');
            
            load(sprintf('%s%s', obj.dsim, obj.ext), variable);
            
            if (exist(variable, 'var') == 1)
                data = eval(variable);
            else
                error('Data does not exist.');
                data = 0;
            end
        end
    
        function fkeys = inspect(obj)
            % Return list of keys from available data in simulation file.
            
            obj.call('EM_save_mat');
            
            fkeys = who('-file',sprintf('%s%s', obj.dsim, obj.ext));
        end
        
        function close(obj, varargin)
            % Send saving options to EMode and close the connection.
            
            if (length(varargin) > 0)
                obj.call('EM_close', varargin{:});
            else
                obj.call('EM_close', 'save', true, 'ftype', 'mat');
            end
            send(obj.conn, uint8('exit'));
            disconnect(obj.conn);
            disp('Exited EMode');
        end
    end
    methods (Static = true)
        function f = open_file(sim)
            % Return an EMode simulation file name with .mat extension.
            
            if nargin == 0
                sim = 'emode';
            end
            
            mat = '.mat';
            
            if (strfind(sim, mat) == length(sim)-length(mat)+1)
                sim = sim(1:end-length(mat));
            end
            
            try
                f = sprintf('%s%s', sim, mat);
            catch
                error('File not found!');
                f = 0;
            end
        end

        function data = get_(variable, sim)
            % Return data from simulation file.
            
            if nargin == 1
                sim = 'emode';
            end
            
            if (~ischar(variable))
                error('Input parameter 'variable' must be a string.');
            end
            
            if (~ischar(sim))
                error('Input parameter 'sim' must be a string.');
            end
            
            mat = '.mat';
            
            if (strfind(sim, mat) == length(sim)-length(mat)+1)
                sim = sim(1:end-length(mat));
            end
            
            try
                data = load(sprintf('%s%s', sim, mat), variable).(variable);
            catch
                error('Data does not exist.');
                data = 0;
            end
        end

        function fkeys = inspect_(sim)
            % Return list of keys from available data in simulation file.
            
            if nargin == 0
                sim = 'emode';
            end
            
            if (~ischar(sim))
                error('Input parameter 'sim' must be a string.');
            end
            
            mat = '.mat';
            
            if (strfind(sim, mat) == length(sim)-length(mat)+1)
                sim = sim(1:end-length(mat));
            end
            
            try
                fkeys = who('-file',sprintf('%s%s', sim, mat));
            catch
                error('File does not exist.');
                fkeys = 0;
            end
        end
    end
end