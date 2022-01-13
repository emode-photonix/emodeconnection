%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EMode - Matlab/Octave interface, by EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c) 2021 EMode Photonix LLC
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
        function obj = emodeconnection(sim, open_existing, new_name, priority, roaming, verbose)
            % Initialize defaults and connect to EMode.
            
            if nargin == 0
                sim = 'emode';
                open_existing = false;
                new_name = false;
                priority = 'pN';
                roaming = false;
                verbose = false;
            elseif nargin == 1
                open_existing = false;
                new_name = false;
                priority = 'pN';
                roaming = false;
                verbose = false;
            elseif nargin == 2
                new_name = false;
                priority = 'pN';
                roaming = false;
                verbose = false;
            elseif nargin == 3
                priority = 'pN';
                roaming = false;
                verbose = false;
            elseif nargin == 4
                roaming = false;
                verbose = false;
            elseif nargin == 5
                verbose = false;
            end
            
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
            
            if isOctave
                pkg load sockets;
                ov = OCTAVE_VERSION;
                if str2num(ov(1)) < 7
                    [usrpkg, syspkg] = pkg('list');
                    nojsonstuff = true;
                    for kk = 1:length(usrpkg)
                        if strcmp(usrpkg{kk}.name, 'jsonstuff')
                            % found jsonstuff
                            nojsonstuff = false;
                        end
                    end
                    if nojsonstuff
                        % install jsonstuff
                        pkg install https://github.com/apjanke/octave-jsonstuff/releases/download/v0.3.3/jsonstuff-0.3.3.tar.gz
                    end
                    pkg load jsonstuff
                end
            end
            
            try
                sim = num2str(sim);
            catch
                error('Input parameter "sim" must be a string.');
                return
            end
            
            try
                priority = num2str(priority);
            catch
                error('Input parameter "priority" must be a string.');
                return
            end
            
            obj.dsim = sim;
            obj.ext = '.mat';
            obj.exit_flag = false;
            obj.DL = 2048;
            obj.HOST = '127.0.0.1';
            obj.LHOST = 'lm.emodephotonix.com';
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
            
            EM_cmd_str = 'EMode.exe %s %s %s';
            
            if verbose == true
                EM_cmd_str = strcat(EM_cmd_str, ' -v');
            end
            
            if ~strcmp(priority, 'pN')
                priority = erase(priority, '-');
                EM_cmd_str = strcat(EM_cmd_str, ' -', priority);
            end
            
            if roaming
                EM_cmd_str = strcat(EM_cmd_str, ' -r');
            end
            
            % Open EMode
            if isOctave
                % system(sprintf(EM_cmd_str, path, obj.LHOST, obj.LPORT, num2str(obj.PORT_SERVER)), false, 'async');
                popen(sprintf(EM_cmd_str, path, obj.LHOST, obj.LPORT, num2str(obj.PORT_SERVER)), "r");
            else % Matlab
                EM_cmd_str = strcat(EM_cmd_str, ' &')
                system(sprintf(EM_cmd_str, path, obj.LHOST, obj.LPORT, num2str(obj.PORT_SERVER)));
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
            
            if strcmp(RV, 'ERROR')
                error('internal EMode error');
                return
            end
            
            obj.dsim = RV(length('sim:')+1:end);
        end
        
        function RV = call(obj, func_name, varargin)
            % Send a command to EMode.
            s = struct();
            if (ischar(func_name))
                s.('function') = func_name;
            else
                error('Input parameter "function" must be a string.');
            end
            
            if (mod(length(varargin), 2))
                error('Incorrect number of inputs!\nAn even number of inputs is required following the function name.');
                return
            end
            
            sim_flag = true;
            for kk = 1:length(varargin)/2
                kw = varargin{kk*2-1};
                kv = varargin{kk*2};
                s.(kw) = kv;
                if (strcmp(kw, 'sim'))
                    sim_flag = false;
                end
            end
            
            if (sim_flag)
                s.('sim') = obj.dsim;
            end
            
            try
                send(obj.conn, uint8(jsonencode(s)));
                recvstr = recv(obj.conn, obj.DL);
            catch
                % Exited due to license checkout
                disconnect(obj.conn);
                obj.exit_flag = true;
            end
            
            if (obj.exit_flag)
                error('License checkout error!');
            end
            
            recvset = jsondecode(recvstr);
            RV = recvset.('RV');
        end
        
        function data = get(obj, variable)
            % Return data from simulation file.
            
            if (~ischar(variable))
                error('Input parameter "variable" must be a string.');
            end
            
            obj.call('EM_save_mat');
            
            fvariables = who('-file', sprintf('%s%s', obj.dsim, obj.ext), variable);
            
            if (ismember(variable, fvariables))
                T = load(sprintf('%s%s', obj.dsim, obj.ext), variable);
                data = T.(variable);
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
                error('Input parameter "variable" must be a string.');
            end
            
            if (~ischar(sim))
                error('Input parameter "sim" must be a string.');
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
                error('Input parameter "sim" must be a string.');
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
