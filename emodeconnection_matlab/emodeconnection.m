%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EMode - MATLAB/Octave interface, by EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c) 2023 EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef emodeconnection
    properties
        isOctave
        endian
        dsim
        ext
        exit_flag
        s
    end
    methods
        function obj = emodeconnection(simulation_name, save_path, verbose, roaming, open_existing, new_name, priority)
            % Initialize defaults and connects to EMode.
            
            if nargin == 0
                simulation_name = 'emode';
                save_path = '.';
                verbose = false;
                roaming = false;
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 1
                save_path = '.';
                verbose = false;
                roaming = false;
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 2
                verbose = false;
                roaming = false;
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 3
                roaming = false;
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 4
                open_existing = false;
                new_name = false;
                priority = 'pN';
            elseif nargin == 5
                new_name = false;
                priority = 'pN';
            elseif nargin == 6
                priority = 'pN';
            end
            
            obj.isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
            [~,~,obj.endian] = computer;
            
            if obj.isOctave
                pkg load sockets;
                pkg load instrument-control;
                ov = OCTAVE_VERSION;
                if str2num(ov(1)) < 7
                    error('EMode only supports Octave version 7.x or higher.');
                end
            else
                if verLessThan('matlab', '9.1')
                    error('EMode only supports MATLAB version R2016b or higher.');
                end
            end
            
            try
                simulation_name = num2str(simulation_name);
            catch
                error('Input parameter "simulation_name" must be a string.');
            end
            
            try
                priority = num2str(priority);
            catch
                error('Input parameter "priority" must be a string.');
            end
            
            obj.dsim = simulation_name;
            obj.ext = '.mat';
            obj.exit_flag = false;
            HOST = '127.0.0.1';
            PORT_SERVER = 0;
            
            if obj.isOctave
                now_utc = gmtime(time());
                port_file_ext = strcat(strftime("%Y%m%d%H%M%S", now_utc), num2str(now_utc.usec));
            else % MATLAB
                port_file_ext = datestr(datetime('now', 'TimeZone', 'Europe/London'), 'yyyymmddHHMMSSFFF');
            end
            port_path = fullfile(getenv('APPDATA'), 'EMode', strcat('port_', port_file_ext, '.txt'));
            
            EM_cmd_str = strcat('EMode.exe run', {' '}, port_file_ext);
            EM_cmd_str = EM_cmd_str{1,1};
            
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
            if obj.isOctave
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
            if obj.isOctave
                write(obj.s, native2unicode('connected with Octave!', 'UTF-8'));
            else % MATLAB
                write(obj.s, native2unicode('connected with MATLAB!', 'UTF-8'));
            end
            pause(0.1); % wait for EMode to recv
            
            if open_existing
                RV = obj.call('EM_open', 'sim', simulation_name, 'save_path', save_path, 'new_name', new_name);
            else
                RV = obj.call('EM_init', 'sim', simulation_name, 'save_path', save_path);
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
                if (strcmp(kw, 'sim') || strcmp(kw, 'simulation_name'))
                    sim_flag = false;
                end
            end
            
            if (sim_flag)
                st.('simulation_name') = obj.dsim;
            end
            
            try
                sendstr = jsonencode(st);
            catch
                error('EMode function inputs must have type string, int/float, or list');
            end
            
            try
                msg = native2unicode(sendstr, 'UTF-8');
                msg_L = uint32([length(msg)]);
                if (obj.endian == 'L')
                    msg_L = swapbytes(msg_L);
                end
                write(obj.s, msg_L, 'uint32');
                write(obj.s, msg);
                
                while true
                    if obj.s.NumBytesAvailable > 0
                        break
                    end
                end
                pause(0.1);
                msglen = read(obj.s, 4);
                msglen = typecast(uint8(msglen), 'uint32');
                if (obj.endian == 'L')
                    msglen = swapbytes(msglen);
                end
                recvstr = read(obj.s, msglen);
            catch
                % Exited due to license checkout
                clear obj.s;
                obj.exit_flag = true;
            end
            
            if (obj.exit_flag)
                error('License checkout error!');
            end
            
            recvjson = char(recvstr);
            try
                RV = jsondecode(recvjson);
            catch
                RV = recvjson;
            end
        end
        
        function data = get(obj, variable)
            % Return data from simulation file.
            
            if (~ischar(variable))
                error('Input parameter "variable" must be a string.');
            end
            
            data = obj.call('EM_get', 'key', variable, 'sim', obj.dsim);
            
            if isstruct(data)
                fnames = fieldnames(data);
                fnamecell = strfind(fnames, '__ndarray__');
                
                for mm = 1:length(fnamecell)
                    if fnamecell{mm} > 0
                        nd_logic = true;
                        nd_fname = fnames{mm};
                    end
                end
                
                if nd_logic
                    dtype = data.dtype;
                    dshape = data.shape;
                    if length(dshape) == 1
                        dshape = [1 dshape];
                    end
                    
                    sdshape = size(dshape);
                    if sdshape(1) > 1
                        dshape = dshape.';
                    end
                    
                    if obj.isOctave
                        data = base64_decode(getfield(data, nd_fname));
                    else
                        data = matlab.net.base64decode(getfield(data, nd_fname));
                    end
                    
                    if strcmp(dtype, 'complex128')
                        data = typecast(data, 'double complex');
                        if ~obj.isOctave
                            data = complex(data(1:2:end), data(2:2:end));
                        end
                    else
                        data = typecast(data, 'double');
                    end
                    
                    data = reshape(data, flip(dshape));
                end
            end
        end
    
        function fkeys = inspect(obj)
            % Return list of keys from available data in simulation file.
            obj.call('EM_save', 'simulation_name', obj.dsim, 'file_type', obj.ext(2:end));
            fkeys = who('-file',sprintf('%s%s', obj.dsim, obj.ext));
        end
        
        function close(obj, varargin)
            % Send saving options to EMode and close the connection.
            try
                if (length(varargin) > 0)
                    obj.call('EM_close', varargin{:});
                else
                    obj.call('EM_close', 'save', true, 'file_type', obj.ext(2:end));
                end
                s = struct();
                s.('function') = 'exit';
                sendstr = jsonencode(s);
                msg = native2unicode(sendstr, 'UTF-8');
                msg_L = uint32([length(msg)]);
                [~,~,endian] = computer;
                if (endian == 'L')
                    msg_L = swapbytes(msg_L);
                end
                write(obj.s, msg_L, 'uint32');
                write(obj.s, msg);
                pause(1.0);
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
                f = 0;
                error('File not found!');
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
                fkeys = 0;
                error('File does not exist.');
            end
        end
    end
end
