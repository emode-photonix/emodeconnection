%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EMode - MATLAB interface, by EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c) EMode Photonix LLC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef emodeconnection < handle

    properties(Access = private)
        proc                      % .NET System.Diagnostics.Process handle
        sock   tcpclient          % MATLAB TCP socket
        endian char               % Host endianness ("L" or "B")
        lhOut     event.listener  % stdout listener
        lhErr     event.listener  % stderr listener
   
        print_timer timer
        stop_thread logical = false
        lastLine  char = ''
        Nlines    double = 0
    end

    properties
        dsim                % active simulation name inside EMode
        ext  char = '.mat'
        exit_flag logical = false
        stdoutLog string = ""   % captured stdout
        stderrLog string = ""   % captured stderr
        
        % --- canonical MATLAB versions of EMode typed structs ------------
        MaterialSpec = struct( ...
            'x__data_type__', 'MaterialSpec', ...
            'material',   [], ...
            'theta',      [], ...
            'phi',        [], ...
            'x',          [], ...
            'loss',       []  ...
        );
        MaterialProperties = struct( ...
            'x__data_type__', 'MaterialProperties', ...
            'n',   [], ...
            'eps', [], ...
            'mu',  [], ...
            'd',   []  ...
        );
    end

    %% ---------------------------------------------------------------------
    % Constructor
    %% ---------------------------------------------------------------------
    methods
        function obj = emodeconnection(namedArgs)
            arguments
                namedArgs.sim                 string = "emode"
                namedArgs.simulation_name     string = "emode"
                namedArgs.license_type        string = "default"
                namedArgs.save_path           string = "."
                namedArgs.verbose             logical = false
                namedArgs.roaming             logical = false
                namedArgs.open_existing       logical = false
                namedArgs.new_name            string  = ""
                namedArgs.priority            string = "pN"   % pN / pL / pH
            end
            assert(~verLessThan('matlab','9.1'),"EMode requires R2016b+");
            [~, ~, obj.endian] = computer;

            % ---- launch EMode -------------------------------------------------
            token     = datestr(now,'yyyymmddHHMMSSFFF');
            port_file = fullfile(getenv('LOCALAPPDATA'),'emode',sprintf('port_%s.txt',token));
            cmd = "run " + token;
            if namedArgs.license_type ~= "default", cmd = cmd + " -"+namedArgs.license_type; end
            if namedArgs.verbose,                cmd = cmd + " -v"; end
            if namedArgs.priority ~= "pN",       cmd = cmd + " -"+erase(namedArgs.priority,'-'); end
            if namedArgs.roaming,                cmd = cmd + " -r"; end
            obj.proc = obj.startProcess('EMode.exe',cmd);

            % ---- wait for server port + connect -------------------------------
            port = obj.waitForPortFile(port_file);
            obj.sock = obj.openSocket('127.0.0.1',port);
            write(obj.sock,native2unicode('connected with MATLA','UTF-8'));

            % ---- init / open sim ---------------------------------------------
            if namedArgs.open_existing
                rv = obj.call('EM_open','simulation_name',namedArgs.simulation_name,'save_path',namedArgs.save_path,'new_simulation_name',namedArgs.new_name);
            else
                rv = obj.call('EM_init','simulation_name',namedArgs.simulation_name,'save_path',namedArgs.save_path);
            end
            if strcmp(rv,'ERROR'), error('Internal EMode error'); end
            obj.dsim = extractAfter(rv,'sim:');
        end
        function delete(obj); obj.cleanup(); end
    end

    %% ---------------------------------------------------------------------
    % Public API (call, close, subsref)
    %% ---------------------------------------------------------------------
    methods
        function rv = call(obj, func_name, varargin)
            st = struct('function',func_name);
            
            % ---- legacy shorthand for EM_get ---------------------------
            if strcmp(func_name, 'EM_get') && numel(varargin) == 1
                varargin = {'key', varargin{1}};
            end
            
            if mod(numel(varargin),2), error('Arguments must be name/value pairs'); end
            for k = 1:2:numel(varargin); st.(varargin{k}) = varargin{k+1}; end
            if ~isfield(st,'simulation_name'), st.simulation_name = obj.dsim; end

            try
                sendstr = jsonencode(st);
                sendstr = regexprep(sendstr, '\"x__data_type__\":', '\"__data_type__\":');
                % this is an ugly hack
                sendstr = regexprep(sendstr, '\[\]','null');
                obj.sendMessage(sendstr)
                recv = obj.recvMessage();
            catch ME
                obj.cleanup(); rethrow(ME);
            end

            try
                parsed = jsondecode(recv);
            catch
                rv = recv;
                return;
            end
                
            % Convert / propagate Python side errors
            
            rv = obj.convert_data(parsed);
        end

        function close(obj,varargin)
            if nargin>1
                obj.call('EM_close',varargin{:});
            else
                obj.call('EM_close','save',true,'file_type',obj.ext(2:end));
            end
            obj.sendMessage(jsonencode(struct('function','exit')));
            pause(0.3);
            obj.cleanup();
        end

        function varargout = subsref(obj,s)
            % 1) let built‑in handle property access or standard behaviour
            if numel(s)==1
                [varargout{1:nargout}] = builtin('subsref',obj,s); return; end

            % 2) if first token is a real method, fall back to builtin
            if ismethod(obj,s(1).subs)
                [varargout{1:nargout}] = builtin('subsref',obj,s); return; end

            % 3) treat as EMode RPC: obj.<name>(args...)
            if isempty(s(2).subs)
                [varargout{1:nargout}] = obj.call("EM_"+s(1).subs);
            else
                [varargout{1:nargout}] = obj.call("EM_"+s(1).subs, s(2).subs{:});
            end
        end
    end

    %% ---------------------------------------------------------------------
    % Internals
    %% ---------------------------------------------------------------------
    methods(Access = private)
        function proc = startProcess(obj,exe,args)
            NET.addAssembly('System'); import System.Diagnostics.*
            psi = ProcessStartInfo(exe,char(args)); psi.CreateNoWindow=true; psi.UseShellExecute=false;
            psi.RedirectStandardOutput=true; psi.RedirectStandardError=true;
            proc = Process(); proc.StartInfo=psi; if ~proc.Start(), error('Failed to start %s',exe); end
            obj.lhOut = addlistener(proc,'OutputDataReceived',@(~,ev)obj.handleStdout(ev)); proc.BeginOutputReadLine();
            obj.lhErr = addlistener(proc,'ErrorDataReceived', @(~,ev)obj.handleStderr(ev)); proc.BeginErrorReadLine();
        end
        function handleStdout(obj,ev)
            if ~isempty(ev.Data)
                line = char(ev.Data);
                fprintf('%s\n',line);
                obj.stdoutLog = obj.stdoutLog + string(line) + newline;
            end
        end
        function handleStderr(obj,ev)
            if ~isempty(ev.Data)
                line = char(ev.Data);
                fprintf('%s\n',line);
                obj.stderrLog = obj.stderrLog + string(line) + newline;
            end
        end

        % ---------------- wait helpers ------------------------------------
        function port = waitForPortFile(~,file)
            back = 0.05; while true
                if exist(file,'file')
                    t = str2double(strtrim(fileread(file))); if ~isnan(t)&&t>0, port=t; return; end
                end
                pause(back); back=min(0.5,back*1.4);
            end
        end
        
        function s = openSocket(obj,host,port)
            while true
                try
                    s = tcpclient(host,port,"ConnectTimeout", 30, 'Timeout', 10); return;   % success
                catch
                    if obj.proc.HasExited, error('EMode exited before socket opened.'); end
                    pause(0.2);
                end
            end
        end

        function sendMessage(obj,str)
            m = uint8(native2unicode(str,'UTF-8')); len = uint32(numel(m)); if obj.endian=='L', len=swapbytes(len); end
            write(obj.sock,[typecast(len,'uint8') m]);
        end
        function str = recvMessage(obj)
            % Robust receive: tolerate timeouts but abort if the EMode
            % process terminates before a complete message is received.
            while true
                % -------- check external process first ------------------
                if obj.proc.HasExited
                    error('EMode exited before sending a response.');
                end
                % -------- attempt to read 4‑byte length prefix ----------
                try
                    lenB = read(obj.sock,4);
                catch ME
                    if emodeconnection.isTimeoutErr(ME)
                        pause(0.1); continue;  % benign timeout – retry
                    else
                        rethrow(ME);          % genuine socket failure
                    end
                end
                if isempty(lenB)
                    pause(0.05); continue;     % nothing yet – retry
                end
                len = typecast(uint8(lenB),'uint32'); if obj.endian=='L', len = swapbytes(len); end
                % -------- read payload ----------------------------------
                payload = uint8.empty;
                while numel(payload) < len
                    if obj.proc.HasExited
                        error('EMode exited mid‑transmission.');
                    end
                    try
                        payload = [payload read(obj.sock, len - numel(payload))]; %#ok<AGROW>
                    catch ME
                        if emodeconnection.isTimeoutErr(ME)
                            pause(0.05); continue;
                        else
                            rethrow(ME);
                        end
                    end
                end
                str = char(payload);
                return;
            end
        end
        % function tf = obj.isTimeoutErr(~,ME)
            % id = lower(ME.identifier); msg = lower(ME.message);
            % tf = contains(id,'timeout') || contains(msg,'timed out');
        % end
        
        % ---------------- full NumPy ndarray → MATLAB conversion ----------
        % function data = convert_data(~, raw)
            % if isstruct(raw)
                % fn = fieldnames(raw);
                % % propagate Python‑side exceptions if present -------------
                % if isfield(raw,'x__data_type__') && contains(raw.x__data_type__,'Error')
                    % errorIdentifier = ['EMode:PythonError:' strrep(raw.x__data_type__,' ','')];
                    % error(errorIdentifier, raw.msg);
                % end

                % nd_idx = find(contains(fn,'__ndarray__'),1);
                % if ~isempty(nd_idx)   % base64‑encoded ndarray  ----------
                    % dtype  = raw.dtype;
                    % shape  = raw.shape; if numel(shape)==1, shape=[1 shape]; end
                    % if size(shape,1) > 1, shape = shape.'; end
                    % bytes  = matlab.net.base64decode(raw.(fn{nd_idx}));
                    % switch dtype
                        % case {'int8','int16','int32','int64', ...
                              % 'uint8','uint16','uint32','uint64'}
                            % arr = typecast(bytes, dtype);
                        % case {'float16','float32','float64','double'}
                            % arr = typecast(bytes,'double');
                        % case {'complex64','complex128'}
                            % d   = typecast(bytes,'double');
                            % arr = complex(d(1:2:end), d(2:2:end));
                        % case 'bool'
                            % arr = logical(typecast(bytes,'uint8'));
                        % otherwise
                            % arr = typecast(bytes,'double');
                    % end
                    % data = reshape(arr, fliplr(shape));
                % else                % recurse down struct  --------------
                    % data = struct();
                    % for i = 1:numel(fn)
                        % data.(fn{i}) = convert_data([], raw.(fn{i})); %#ok<AGROW>
                    % end
                % end
            % elseif iscell(raw)
                % data = cellfun(@(x)convert_data([],x), raw, 'UniformOutput',false);
            % else
                % data = raw;         % primitive ➜ as‑is
            % end
        % end
        
        function data = convert_data(obj, raw_data)
            % Recursive decoder for EMode → MATLAB data, mirroring the
            % original Python client behaviour.
            %   • Detects Python-side exceptions via x__data_type__ == *Error*
            %   • Converts base‑64 NumPy ndarrays to MATLAB arrays, with
            %     shape reversed so that column‑major MATLAB matches the
            %     row‑major Python view.
            %   • Recreates nested structs so users can chain dots, e.g.
            %       em.get('shape_core').metadata.scattering_sum
            if isstruct(raw_data)
                fnames = fieldnames(raw_data);

                % ---------- propagate Python exceptions ----------------
                if isfield(raw_data, 'x__data_type__')
                    dtVal = raw_data.x__data_type__;
                    if (ischar(dtVal) || isstring(dtVal)) && contains(dtVal, 'Error')
                        errorIdentifier = ['EMode:PythonError:' strrep(dtVal, ' ', '')];
                        if isfield(raw_data, 'msg')
                            error(errorIdentifier, raw_data.msg);
                        else
                            error(errorIdentifier, raw_data);
                        end
                    end
                end

                % ---------- detect encoded ndarray ---------------------
                nd_logic = false; nd_fname = '';
                for mm = 1:numel(fnames)
                    if contains(fnames{mm}, '__ndarray__')
                        nd_logic = true; nd_fname = fnames{mm}; break; end
                end

                if nd_logic
                    dtype  = raw_data.dtype;
                    dshape = raw_data.shape;
                    if numel(dshape) == 1, dshape = [1 dshape]; end
                    if size(dshape,1) > 1, dshape = dshape.'; end

                    data_bytes = matlab.net.base64decode(raw_data.(nd_fname));
                    switch dtype
                        case {'int8','int16','int32','int64', ...
                              'uint8','uint16','uint32','uint64'}
                            data_ = typecast(data_bytes, dtype);
                        case {'float16','float32','float64'}
                            data_ = typecast(data_bytes, 'double');
                        case {'complex64','complex128'}
                            d = typecast(data_bytes, 'double');
                            data_ = complex(d(1:2:end), d(2:2:end));
                        case 'bool'
                            data_ = logical(typecast(data_bytes,'uint8'));
                        otherwise
                            data_ = typecast(data_bytes,'double');
                    end
                    data = reshape(data_, flip(dshape));
                else
                    % ---------- recurse into subfields ----------------
                    data = struct();
                    for ii = 1:numel(fnames)
                        data.(fnames{ii}) = obj.convert_data(raw_data.(fnames{ii}));
                    end
                end
            elseif iscell(raw_data)
                data = cellfun(@(x)obj.convert_data(x), raw_data, 'UniformOutput', false);
            else
                data = raw_data;
            end
        end
    %% ---------------------------------------------------------------------
    % Cleanup & destructor
    %% ---------------------------------------------------------------------
%    methods(Access = private)
        function cleanup(obj)
            if ~isempty(obj.sock) && isvalid(obj.sock) && strcmp(obj.sock.Status,'open'); clear obj.sock; end
            if ~isempty(obj.lhOut); delete(obj.lhOut); end
            if ~isempty(obj.lhErr); delete(obj.lhErr); end
            if ~isempty(obj.proc) && isvalid(obj.proc) && ~obj.proc.HasExited; obj.proc.Kill(); end
        end
    end
    methods(Access = private, Static = true)
        function tf = isTimeoutErr(ME)
            id  = lower(ME.identifier);
            msg = lower(ME.message);
            tf = contains(id,'timeout') || contains(msg,'timed out');
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

