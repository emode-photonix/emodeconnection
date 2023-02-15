%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EMode - MATLAB interface, by EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c) 2023 EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef emodeconnection
    properties
        endian, dsim, ext, exit_flag, s;
        sim, simulation_name, save_path, verbose, roaming, open_existing, new_name, priority;
    end
    methods
        function obj = emodeconnection(namedArgs)
            % Initialize defaults and connects to EMode.
            
            arguments
                namedArgs.sim = 'emode';
                namedArgs.simulation_name = 'emode';
                namedArgs.save_path = '.';
                namedArgs.verbose = false;
                namedArgs.roaming = false;
                namedArgs.open_existing = false;
                namedArgs.new_name = false;
                namedArgs.priority = 'pN';
            end
            
            obj.sim = namedArgs.sim;
            obj.simulation_name = namedArgs.simulation_name;
            obj.save_path = namedArgs.save_path;
            obj.verbose = namedArgs.verbose;
            obj.roaming = namedArgs.roaming;
            obj.open_existing = namedArgs.open_existing;
            obj.new_name = namedArgs.new_name;
            obj.priority = namedArgs.priority;
            
            if strcmp(obj.simulation_name, 'emode') && ~strcmp(obj.sim, 'emode')
                obj.simulation_name = obj.sim;
            end
            
            [~,~,obj.endian] = computer;
            
            if verLessThan('matlab', '9.1')
                error('EMode only supports MATLAB version R2016b or higher.');
            end
            
            try
                obj.simulation_name = num2str(obj.simulation_name);
            catch
                error('Input parameter "simulation_name" must be a string.');
            end
            
            try
                obj.priority = num2str(obj.priority);
            catch
                error('Input parameter "priority" must be a string.');
            end
            
            obj.dsim = obj.simulation_name;
            obj.ext = '.mat';
            obj.exit_flag = false;
            HOST = '127.0.0.1';
            PORT_SERVER = 0;
            
            port_file_ext = string(datetime('now', 'TimeZone', 'Europe/London'), 'yyyyMMddHHmmssSSS');

            port_path = fullfile(getenv('APPDATA'), 'EMode', strcat('port_', port_file_ext, '.txt'));
            
            EM_cmd_str = strcat('EMode.exe run', {' '}, port_file_ext);
            EM_cmd_str = EM_cmd_str{1,1};
            
            if obj.verbose == true
                EM_cmd_str = strcat(EM_cmd_str, ' -v');
            end
            
            if ~strcmp(obj.priority, 'pN')
                obj.priority = erase(obj.priority, '-');
                EM_cmd_str = strcat(EM_cmd_str, ' -', obj.priority);
            end
            
            if obj.roaming
                EM_cmd_str = strcat(EM_cmd_str, ' -r');
            end
            
            % Open EMode
            EM_cmd_str = strcat(EM_cmd_str, ' &');
            system(EM_cmd_str);
            
            % Read EMode port
            t1 = datetime('now');
            waiting = true;
            wait_time = seconds(20); % [seconds]
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
                elseif (datetime('now') - t1) > wait_time
                    waiting = false;
                end
                pause(0.05);
            end
            
            if ~waiting
                error("EMode connection error!");
            end
            
            pause(0.1) % wait for EMode to open
            obj.s = tcpclient(HOST, PORT_SERVER, "Timeout", 60);
            write(obj.s, native2unicode('connected with MATLAB!', 'UTF-8'));
            pause(0.1); % wait for EMode to recv
            
            if obj.open_existing
                RV = obj.call('EM_open', 'simulation_name', obj.simulation_name, 'save_path', obj.save_path, 'new_name', obj.new_name);
            else
                RV = obj.call('EM_init', 'simulation_name', obj.simulation_name, 'save_path', obj.save_path);
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
            
            if (length(varargin) == 1) && strcmp(func_name, 'EM_get')
                varargin = {'key'; varargin{1}};
            elseif (mod(length(varargin), 2))
                error('Incorrect number of inputs! An even number of inputs is required following the function name.');
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
            RV = obj.convert_data(RV);
        end
        
        function data = convert_data(~, raw_data)
            if isstruct(raw_data)
                fnames = fieldnames(raw_data);
                fnamecell = strfind(fnames, '__ndarray__');
                
                for mm = 1:length(fnamecell)
                    if fnamecell{mm} > 0
                        nd_logic = true;
                        nd_fname = fnames{mm};
                    end
                end
                
                if nd_logic
                    dtype = raw_data.dtype;
                    dshape = raw_data.shape;
                    if length(dshape) == 1
                        dshape = [1 dshape];
                    end
                    
                    sdshape = size(dshape);
                    if sdshape(1) > 1
                        dshape = dshape.';
                    end
                    
                    data_bytes = matlab.net.base64decode(raw_data.(nd_fname));
                    data_ = typecast(data_bytes, 'double');

                    if strcmp(dtype, 'complex128')
                        data_ = complex(data_(1:2:end), data_(2:2:end));
                    end
                    
                    data = reshape(data_, flip(dshape));
                else
                    data = raw_data;
                end
            else
                data = raw_data;
            end
        end
        
        function close(obj, varargin)
            % Send saving options to EMode and close the connection.
            try
                if isempty(varargin)
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
        
        function varargout = subsref(obj, s)
            if ismethod(obj, s(1).subs)
                [varargout{1:nargout}] = builtin('subsref',obj,s);
                return
            end
            
            if isempty(s(2).subs)
                [varargout{1:nargout}] = obj.call(strcat('EM_',s(1).subs));
            else
                [varargout{1:nargout}] = obj.call(strcat('EM_',s(1).subs), s(2).subs{:});
            end
        end
    end
    methods (Static = true)
        function f = open_file(simulation_name)
            % Return an EMode simulation file name with .mat extension.
            
            if nargin == 0
                simulation_name = 'emode';
            end
            
            mat = '.mat';
            
            if (strfind(simulation_name, mat) == length(simulation_name)-length(mat)+1)
                simulation_name = simulation_name(1:end-length(mat));
            end
            
            try
                f = sprintf('%s%s', simulation_name, mat);
            catch
                f = 0;
                error('File not found!');
            end
        end

        function data = get_(variable, simulation_name)
            % Return data from simulation file.
            
            if nargin == 1
                simulation_name = 'emode';
            end
            
            if (~ischar(variable))
                error('Input parameter "variable" must be a string.');
            end
            
            if (~ischar(simulation_name))
                error('Input parameter "simulation_name" must be a string.');
            end
            
            mat = '.mat';
            
            if (strfind(simulation_name, mat) == length(simulation_name)-length(mat)+1)
                simulation_name = simulation_name(1:end-length(mat));
            end
            
            try
                data_load = load(sprintf('%s%s', simulation_name, mat), variable);
                data = data_load.(variable);
            catch
                error('Data does not exist.');
                data = 0;
            end
        end

        function fkeys = inspect_(simulation_name)
            % Return list of keys from available data in simulation file.
            
            if nargin == 0
                simulation_name = 'emode';
            end
            
            if (~ischar(simulation_name))
                error('Input parameter "simulation_name" must be a string.');
            end
            
            mat = '.mat';
            
            if (strfind(simulation_name, mat) == length(simulation_name)-length(mat)+1)
                simulation_name = simulation_name(1:end-length(mat));
            end
            
            try
                fkeys = who('-file',sprintf('%s%s', simulation_name, mat));
            catch
                fkeys = 0;
                error('File does not exist.');
            end
        end
    end
end
