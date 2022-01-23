%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EMode - MATLAB/Octave interface, by EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c) 2022 EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef emodeconnection
    properties
        dsim
        ext
        exit_flag
        DL
        s
    end
    methods
        function obj = emodeconnection(sim, verbose, roaming, open_existing, new_name, priority)
            % Initialize defaults and connects to EMode.
            
            if nargin == 0
                sim = 'emode';
                verbose = false;
                roaming = false;
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 1
                verbose = false;
                roaming = false;
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 2
                roaming = false;
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 3
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 4
                new_name = false;
                priority = 'pN';
            elseif nargin == 5
                priority = 'pN';
            end
            
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
            
            if isOctave
                pkg load sockets;
                ov = OCTAVE_VERSION;
                if str2num(ov(1)) < 7
                    [usrpkg, ~] = pkg('list');
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
                    pkg load instrument-control
                end
            else
                if verLessThan('matlab', '9.1')
                    error('EMode only supports MATLAB version R2016b or higher.');
                end
            end
            
            try
                sim = num2str(sim);
            catch
                error('Input parameter "sim" must be a string.');
            end
            
            try
                priority = num2str(priority);
            catch
                error('Input parameter "priority" must be a string.');
            end
            
            obj.dsim = sim;
            obj.ext = '.mat';
            obj.exit_flag = false;
            obj.DL = 2048;
            HOST = '127.0.0.1';
            PORT_SERVER = 0;
            
            port_path = fullfile(getenv('APPDATA'), 'EMode', 'port.txt');
            if exist(port_path, 'file') == 2
                delete(port_path);
            end
            
            EM_cmd_str = 'EMode.exe run';
            
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
                system(EM_cmd_str, false, 'async');
            else % MATLAB
                EM_cmd_str = strcat(EM_cmd_str, ' &');
                system(EM_cmd_str);
            end
            
            % Read EMode port
            t1 = now;
            waiting = true;
            wait_time = 20; % [seconds]
            while waiting
                try
                    file = fopen(port_path, 'r');
                    PORT_SERVER = str2num(fscanf(file, '%s'));
                    fclose(file);
                catch
                    % continue
                end
                if (PORT_SERVER ~= 0)
                    break
                elseif (now - t1) > wait_time
                    waiting = false;
                end
                pause(0.05);
            end
            
            if ~waiting
                error("EMode connection error!");
            end
            
            pause(0.1) % wait for EMode to open
            obj.s = tcpclient(HOST, PORT_SERVER, "Timeout", 60);
            if isOctave
                write(obj.s, native2unicode('connected with Octave!', 'UTF-8'));
            else % MATLAB
                write(obj.s, native2unicode('connected with MATLAB!', 'UTF-8'));
            end
            pause(0.1); % wait for EMode to recv
            
            if open_existing
                RV = obj.call('EM_open', 'sim', sim, 'new_name', new_name);
            else
                RV = obj.call('EM_init', 'sim', sim);
            end
            
            if strcmp(RV, 'ERROR')
                error('internal EMode error');
            end
            
            obj.dsim = RV(length('sim:')+1:end);
        end
        
        function RV = call(obj, func_name, varargin)
            % Send a command to EMode.
            st = struct();
            if (ischar(func_name))
                st.('function') = func_name;
            else
                error('Input parameter "function" must be a string.');
            end
            
            if (mod(length(varargin), 2))
                error('Incorrect number of inputs!\nAn even number of inputs is required following the function name.');
            end
            
            sim_flag = true;
            for kk = 1:length(varargin)/2
                kw = varargin{kk*2-1};
                kv = varargin{kk*2};
                st.(kw) = kv;
                if (strcmp(kw, 'sim'))
                    sim_flag = false;
                end
            end
            
            if (sim_flag)
                st.('sim') = obj.dsim;
            end
            
            try
                sendstr = jsonencode(st);
            catch
                error('EMode function inputs must have type string, int/float, or list');
            end
            
            try
                write(obj.s, native2unicode(sendstr, 'UTF-8'));
                while true
                    if obj.s.NumBytesAvailable > 0
                        break
                    end
                end
                pause(0.1);
                recvstr = read(obj.s, obj.s.NumBytesAvailable);
            catch
                % Exited due to license checkout
                clear obj.s;
                obj.exit_flag = true;
            end
            
            if (obj.exit_flag)
                error('License checkout error!');
            end
            
            recvset = jsondecode(char(recvstr));
            RV = recvset.('RV');
        end
        
        function data = get(obj, variable)
            % Return data from simulation file.
            
            if (~ischar(variable))
                error('Input parameter "variable" must be a string.');
            end
            
            obj.call('EM_save', 'sim', obj.dsim, 'ftype', obj.ext(2:end));
            
            fvariables = who('-file', sprintf('%s%s', obj.dsim, obj.ext), variable);
            
            for kk = 1:100
                if (ismember(variable, fvariables))
                    break
                end
                pause(0.1); % wait for file to write
            end
            
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
            obj.call('EM_save', 'sim', obj.dsim, 'ftype', obj.ext(2:end));
            fkeys = who('-file',sprintf('%s%s', obj.dsim, obj.ext));
        end
        
        function close(obj, varargin)
            % Send saving options to EMode and close the connection.
            try
                if (length(varargin) > 0)
                    obj.call('EM_close', varargin{:});
                else
                    obj.call('EM_close', 'save', true, 'ftype', obj.ext(2:end));
                end
                s = struct();
                s.('function') = 'exit';
                sendstr = jsonencode(s);
                write(obj.s, native2unicode(sendstr, 'UTF-8'));
                pause(0.25);
            catch
                % continue
            end
            clear obj.s;
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
                data_load = load(sprintf('%s%s', sim, mat), variable);
                data = data_load.(variable);
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
