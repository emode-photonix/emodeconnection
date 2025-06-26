%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EMode - MATLAB interface, by EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c) EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef emodeconnection < handle
    properties
        endian, dsim, ext, exit_flag, s;
        sim, simulation_name, license_type, save_path;
        verbose, roaming, open_existing, new_name, priority;
        MaterialSpec = struct(...
            'data_type', 'MaterialSpec', ...
            'material', [], ...
            'theta', [], ...
            'phi', [], ...
            'x', [], ...
            'loss', [] ...
        );
        MaterialProperties = struct(...
            'data_type', 'MaterialProperties', ...
            'n', [], ...
            'eps', [], ...
            'mu', [], ...
            'd', [] ...
        );
        print_timer, print_path;
        stop_thread = false;
        lastLine = '';
        Nlines = 0;
    end
    methods
        function obj = emodeconnection(namedArgs)
            % Initialize defaults and connects to EMode.
            
            arguments
                namedArgs.sim = 'emode';
                namedArgs.simulation_name = 'emode';
                namedArgs.license_type = 'default';
                namedArgs.save_path = '.';
                namedArgs.verbose = false;
                namedArgs.roaming = false;
                namedArgs.open_existing = false;
                namedArgs.new_name = false;
                namedArgs.priority = 'pN';
            end
            
            obj.sim = namedArgs.sim;
            obj.simulation_name = namedArgs.simulation_name;
            obj.license_type = namedArgs.license_type;
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
                obj.license_type = num2str(obj.license_type);
            catch
                error('Input parameter "license_type" must be a string.');
            end
            
            if ~ismember(lower(obj.license_type), {'default', '2d', '3d'})
                error("Input for 'license_type' is not an accepted value")
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

            port_path = fullfile(getenv('LOCALAPPDATA'), 'emode', strcat('port_', port_file_ext, '.txt'));
            
            EM_cmd_str = strcat('EMode.exe run', {' '}, port_file_ext);
            EM_cmd_str = EM_cmd_str{1,1};
            
            if ~strcmp(obj.license_type, 'default')
                EM_cmd_str = strcat(EM_cmd_str, ' -', obj.license_type);
            end

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
            
            % Run printing process
            obj.print_path = fullfile(getenv('LOCALAPPDATA'), 'emode', strcat('emode_output_', port_file_ext, '.txt'));
            obj.start_printing();
            
            % Open EMode
            java.lang.Runtime.getRuntime.exec(EM_cmd_str);
            
            % Read EMode port
            t1 = datetime('now');
            waiting = true;
            wait_time = seconds(60); % [seconds]
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
            write(obj.s, native2unicode('connected with MATLAB', 'UTF-8'));
            pause(0.1); % wait for EMode to recv
            
            if obj.open_existing
                RV = obj.call('EM_open', 'simulation_name', obj.simulation_name, 'save_path', obj.save_path, 'new_simulation_name', obj.new_name);
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
                msg = uint8(native2unicode(sendstr, 'UTF-8'));
                msg_L = uint32([length(msg)]);
                if (obj.endian == 'L')
                    msg_L_unint8 = typecast(msg_L, 'uint8');
                    msg_L = msg_L_unint8(end:-1:1);
                end

                write(obj.s, [msg_L, msg]);
                
                while obj.s.Connected
                
                    if obj.s.NumBytesAvailable > 0
                        break
                    else
                        pause(0.01);
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
        
        function data = convert_data(obj, raw_data)
            if isstruct(raw_data)
                fnames = fieldnames(raw_data);

                if isfield(raw_data, 'x__data_type__')
                    dataTypeValue = raw_data.x__data_type__;
                    if ischar(dataTypeValue) && contains(dataTypeValue, 'Error')
                        errorIdentifier = ['PythonDataConverter:PythonError:', strrep(dataTypeValue, ' ', '')];
                        error(errorIdentifier, raw_data.msg);
                    end
                end
                
                fnamecell = strfind(fnames, '__ndarray__');
                nd_logic = false;
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
        
                    switch dtype
                        case {'int8', 'int16', 'int32', 'int64', ...
                              'uint8', 'uint16', 'uint32', 'uint64'}
                            data_ = typecast(data_bytes, dtype);
                        case {'float16', 'float32', 'float64'}
                            data_ = typecast(data_bytes, 'double');
                        case {'complex64', 'complex128'}
                            data_ = typecast(data_bytes, 'double');
                            data_ = complex(data_(1:2:end), data_(2:2:end));
                        case 'bool'
                            data_ = logical(typecast(data_bytes, 'uint8'));
                        otherwise
                            % Generic case for any other data type
                            data_ = typecast(data_bytes, 'double');
                    end
        
                    data = reshape(data_, flip(dshape));
                else
                    data = struct();
                    for ii = 1:numel(fnames)
                        data.(fnames{ii}) = obj.convert_data(raw_data.(fnames{ii}));
                    end
                end
            else
                data = raw_data;
            end
        end
        
        function close(obj, varargin)
            % Send saving options to EMode and close the connection.
            try
                if ~isempty(varargin)
                    obj.call('EM_close', varargin{:});
                else
                    obj.call('EM_close', 'save', true, 'file_type', obj.ext(2:end));
                end
                st = struct();
                st.('function') = 'exit';
                sendstr = jsonencode(st);

                msg = uint8(native2unicode(sendstr, 'UTF-8'));
                msg_L = uint32([length(msg)]);
                if (obj.endian == 'L')
                    msg_L_unint8 = typecast(msg_L, 'uint8');
                    msg_L = msg_L_unint8(end:-1:1);
                end

                write(obj.s, [msg_L, msg]);

                pause(0.5);
            catch
                % continue
            end
            clear obj.s;
            obj.stop_printing();
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
        
        function print_output(obj)
            % Loop until the file exists
            if ~exist(obj.print_path, 'file') || obj.stop_thread
                return;
            end
            
            try
                while true
                    lines = custom_read(obj.print_path);
                    lines2 = custom_read(obj.print_path);
                    if isequal(lines, lines2)
                        break
                    end
                end
            catch ME
                fprintf(ME.message);
                return;
            end
            
            if isequal(length(lines), 0)
                return;
            end
            
            Nnewlines = max(0, length(lines) - obj.Nlines);
            obj.Nlines = obj.Nlines + Nnewlines;
            
            for kk = max(1,length(lines)-Nnewlines):length(lines)
                line = lines{kk};
                if ~strcmp(line, obj.lastLine)
                    pline = regexprep(line,'[\n\r]+','');
                    if startsWith(line, obj.lastLine) && ~endsWith(obj.lastLine, '\n')
                        fprintf('%s', pline(length(obj.lastLine) + 1:end));
                    else
                        fprintf('\n%s', pline);
                    end
                end
                obj.lastLine = line;
            end
        end
        
        function start_printing(obj)
            obj.print_timer = timer('ExecutionMode', 'fixedRate', ...
                               'Period', 0.02, ...
                               'TasksToExecute', Inf, ...
                               'TimerFcn', @(~,~)obj.print_output());
            start(obj.print_timer);
        end
        
        function stop_printing(obj)
            obj.stop_thread = true;
            fprintf('\n');
            if ~isempty(obj.print_timer)
                stop(obj.print_timer);
                delete(obj.print_timer);
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

        function EModeLogin()
            system('EMode');
        end
    end
end

function lines = custom_read(print_path)
    % Open the file for reading
    try
        fid = fopen(print_path, 'r');
        % Check if file was opened successfully
        if fid == -1
            error(['Error opening file: ' print_path]);
        end
    catch ME
        % Handle other potential errors during opening
        error(['Error: ' ME.message]);
    end
    
    % Read lines
    lines = cell(0);
    while true
        line = fgets(fid);
        if isequal(line, -1)
            break
        end
        lines{end+1} = line;
    end
    
    fclose(fid); % Close the file
    % Return the lines as a cell array
end
