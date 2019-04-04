clear
clc
%control port
controlSerialPort=6;
%data port
dataSerialPort=5;


%cfg name
%configurationFileName = 'C:\Users\haoyu.zz\Desktop\test.cfg';
configurationFileName = 'mmw_pplcount_demo_default.cfg';
%%initial
syncPatternUINT64 = typecast(uint16([hex2dec('0102'),hex2dec('0304'),hex2dec('0506'),hex2dec('0708')]),'uint64');
syncPatternUINT8 = typecast(uint16([hex2dec('0102'),hex2dec('0304'),hex2dec('0506'),hex2dec('0708')]),'uint8');

frameHeaderStructType = struct(...
    'sync',             {'uint64', 8}, ... % See syncPatternUINT64 below
    'version',          {'uint32', 4}, ...
    'platform',         {'uint32', 4}, ...
    'timestamp',        {'uint32', 4}, ... % 600MHz clocks
    'packetLength',     {'uint32', 4}, ... % In bytes, including header
    'frameNumber',      {'uint32', 4}, ... % Starting from 1
    'subframeNumber',   {'uint32', 4}, ...
    'chirpMargin',      {'uint32', 4}, ... % Chirp Processing margin, in ms
    'frameMargin',      {'uint32', 4}, ... % Frame Processing margin, in ms
    'uartSentTime' ,    {'uint32', 4}, ... % Time spent to send data, in ms
    'trackProcessTime', {'uint32', 4}, ... % Tracking Processing time, in ms
    'numTLVs' ,         {'uint16', 2}, ... % Number of TLVs in thins frame
    'checksum',         {'uint16', 2});    % Header checksum

tlvHeaderStruct = struct(...
    'type',             {'uint32', 4}, ... % TLV object Type
    'length',           {'uint32', 4});    % TLV object Length, in bytes, including TLV header 

% Point Cloud TLV object consists of an array of points. 
% Each point has a structure defined below
pointStruct = struct(...
    'range',            {'float', 4}, ... % Range, in m
    'angle',            {'float', 4}, ... % Angel, in rad
    'doppler',          {'float', 4}, ... % Doplper, in m/s
    'snr',              {'float', 4});    % SNR, ratio
% Target List TLV object consists of an array of targets. 
% Each target has a structure define below
targetStruct = struct(...
    'tid',              {'uint32', 4}, ... % Track ID
    'posX',             {'float', 4}, ... % Target position in X dimension, m
    'posY',             {'float', 4}, ... % Target position in Y dimension, m
    'velX',             {'float', 4}, ... % Target velocity in X dimension, m/s
    'velY',             {'float', 4}, ... % Target velocity in Y dimension, m/s
    'accX',             {'float', 4}, ... % Target acceleration in X dimension, m/s2
    'accY',             {'float', 4}, ... % Target acceleration in Y dimension, m/s
    'EC',               {'float', 9*4}, ... % Tracking error covariance matrix, [3x3], in range/angle/doppler coordinates
    'G',                {'float', 4});    % Gating function gain

frameHeaderLengthInBytes = lengthFromStruct(frameHeaderStructType);
tlvHeaderLengthInBytes = lengthFromStruct(tlvHeaderStruct);
pointLengthInBytes = lengthFromStruct(pointStruct);
targetLengthInBytes = lengthFromStruct(targetStruct);
indexLengthInBytes = 1;

exitRequest = 0;
lostSync = 0;
gotHeader = 0;
outOfSyncBytes = 0;
runningSlow = 0;
maxBytesAvailable = 0;
point3D = [];

%%send config
hControlSerialPort = configureControlPort(controlSerialPort);
%configurationFileName = 'mmw_pcdemo_default.cfg';   
cliCfg = readCfg(configurationFileName);
%Params = parseCfg(cliCfg);
    %Send CLI configuration
     fprintf('Sending configuration from %s file to IWR16xx ...\n', configurationFileName);
     for k=1:length(cliCfg)
            fprintf(hControlSerialPort, cliCfg{k});
            fprintf('%s\n', cliCfg{k});
            echo = fgetl(hControlSerialPort); % Get an echo of a command
            done = fgetl(hControlSerialPort); % Get "Done" 
%           prompt = fread(hControlSerialPort, size(mmwDemoCliPrompt,2)); % Get the prompt back 
     end
        fclose(hControlSerialPort);
        delete(hControlSerialPort);
        
hDataSerialPort = configureDataSport(dataSerialPort, 65536);

j=11;
savedata=[];
savetime=[];
%% main function
while(isvalid(hDataSerialPort))
   
    
     while(lostSync == 0 && isvalid(hDataSerialPort))
        j=j+1
        frameStart = tic;
       
        bytesAvailable = get(hDataSerialPort,'BytesAvailable');
        if(bytesAvailable > maxBytesAvailable)
            maxBytesAvailable = bytesAvailable;
        end
       
        if(gotHeader == 0)
            %Read the header first
            [rxHeader, byteCount] = fread(hDataSerialPort, frameHeaderLengthInBytes, 'uint8');
        end
        
        magicBytes = typecast(uint8(rxHeader(1:8)), 'uint64');
        if(magicBytes ~= syncPatternUINT64)
            reason = 'No SYNC pattern';
            lostSync = 1;
            break;
        end
        if(byteCount ~= frameHeaderLengthInBytes)
            reason = 'Header Size is wrong';
            lostSync = 1;
            break;
        end        
        if(validateChecksum(rxHeader) ~= 0)
            reason = 'Header Checksum is wrong';
            lostSync = 1;
            break; 
        end
        
        frameHeader = readToStruct(frameHeaderStructType, rxHeader);
        
        if(gotHeader == 1)
            if(frameHeader.frameNumber > targetFrameNum)
                targetFrameNum = frameHeader.frameNumber;
                disp(['Found sync at frame ',num2str(targetFrameNum),'(',num2str(frameNum),'), after ', num2str(1000*toc(lostSyncTime),3), 'ms']);
                gotHeader = 0;
            else
                reason = 'Old Frame';
                gotHeader = 0;
                lostSync = 1;
                break;
            end
        end
        
        % We have a valid header
        targetFrameNum = frameHeader.frameNumber;
       
        dataLength = frameHeader.packetLength - frameHeaderLengthInBytes;
          
        numInputPoints = 0;
        numTargets = 0;
        mIndex = [];

        if(dataLength > 0)
            %Read all packet
            [rxData, byteCount] = fread(hDataSerialPort, double(dataLength), 'uint8');
            if(byteCount ~= double(dataLength))
                reason = 'Data Size is wrong'; 
                lostSync = 1;
                break;  
            end
            offset = 0;
            
            % TLV Parsing
            for nTlv = 1:frameHeader.numTLVs
                tlvType = typecast(uint8(rxData(offset+1:offset+4)), 'uint32');
                tlvLength = typecast(uint8(rxData(offset+5:offset+8)), 'uint32');
                if(tlvLength + offset > dataLength)
                    reason = 'TLV Size is wrong';
                    lostSync = 1;
                    break;                    
                end
                offset = offset + tlvHeaderLengthInBytes;
                valueLength = tlvLength - tlvHeaderLengthInBytes;
                switch(tlvType)
                     case 6
                        % Point Cloud TLV
                        numInputPoints = valueLength/pointLengthInBytes;
                        if(numInputPoints > 0)                        
                            % Get Point Cloud from the sensor
                            p = typecast(uint8(rxData(offset+1: offset+valueLength)),'single');
                            pointCloud = reshape(p,4, numInputPoints);    
%                           pointCloud(2,:) = pointCloud(2,:)*pi/180;
                            posAll = [pointCloud(1,:).*sin(pointCloud(2,:)); pointCloud(1,:).*cos(pointCloud(2,:))];
                            snrAll = pointCloud(4,:);
%                             % Remove out of Range, Behind the Walls, out of FOV points
%                             inRangeInd = (pointCloud(1,:) > 1) & (pointCloud(1,:) < 6) & ...
%                                 (pointCloud(2,:) > -50*pi/180) &  (pointCloud(2,:) < 50*pi/180) & ...
%                                 (posAll(1,:) > scene.areaBox(1)) & (posAll(1,:) < (scene.areaBox(1) + scene.areaBox(3))) & ...
%                                 (posAll(2,:) > scene.areaBox(2)) & (posAll(2,:) < (scene.areaBox(2) + scene.areaBox(4)));
%                             pointCloudInRange = pointCloud(:,inRangeInd);
%                             posInRange = posAll(:,inRangeInd);

                            numOutputPoints = size(pointCloud,2);                          
                        end                        
                        offset = offset + valueLength;
                                            
                    case 7
                        % Target List TLV
                        numTargets = valueLength/targetLengthInBytes;                        
                        TID = zeros(1,numTargets);
                        S = zeros(6, numTargets);
                        EC = zeros(9, numTargets);
                        G = zeros(1,numTargets);                        
                        for n=1:numTargets
                            TID(n)  = typecast(uint8(rxData(offset+1:offset+4)),'uint32');      %1x4=4bytes
                            S(:,n)  = typecast(uint8(rxData(offset+5:offset+28)),'single');     %6x4=24bytes
                            EC(:,n) = typecast(uint8(rxData(offset+29:offset+64)),'single');    %9x4=36bytes
                            G(n)    = typecast(uint8(rxData(offset+65:offset+68)),'single');    %1x4=4bytes
                            offset = offset + 68;
                        end
                        
                    case 8
                        % Target Index TLV
                        numIndices = valueLength/indexLengthInBytes;
                        mIndex = typecast(uint8(rxData(offset+1:offset+numIndices)),'uint8');
                        offset = offset + valueLength;
                end
            end
        end
    if(numInputPoints == 0)
            numOutputPoints = 0;
            pointCloud = single(zeros(4,0));
            posAll = [];
            posInRange = [];  
    end
        if(numTargets == 0)
            TID = [];
            S = [];
            EC = [];
            G = [];
        end

        savedata=[savedata,zeros(4,1),pointCloud];             
        if(mod(j,1200)==0)
            num=num2str(j);
            name=[num '.mat'];
              time= datestr(now,'yyyymmddTHHMMSS');
            save(name,'savedata','time')
            savedata=[];
        end
     end
end
            
function config = readCfg(filename)
    config = cell(1,100);
    fid = fopen(filename, 'r');
    if fid == -1
        fprintf('File %s not found!\n', filename);
        return;
    else
        fprintf('Opening configuration file %s ...\n', filename);
    end
    tline = fgetl(fid);
    k=1;
    while ischar(tline)
        config{k} = tline;
        tline = fgetl(fid);
        k = k + 1;
    end
    config = config(1:k-1);
    fclose(fid);
end
        
function [sphandle] = configureDataSport(comPortNum, bufferSize)
    if ~isempty(instrfind('Type','serial'))
        disp('Serial port(s) already open. Re-initializing...');
        delete(instrfind('Type','serial'));  % delete open serial ports.
    end
    comPortString = ['COM' num2str(comPortNum)];
    sphandle = serial(comPortString,'BaudRate',921600);
    set(sphandle,'Terminator', '');
    set(sphandle,'InputBufferSize', bufferSize);
    set(sphandle,'Timeout',10);
    set(sphandle,'ErrorFcn',@dispError);
    fopen(sphandle);
end

function [sphandle] = configureControlPort(comPortNum)
    %if ~isempty(instrfind('Type','serial'))
    %    disp('Serial port(s) already open. Re-initializing...');
    %    delete(instrfind('Type','serial'));  % delete open serial ports.
    %end
    comPortString = ['COM' num2str(comPortNum)];
    sphandle = serial(comPortString,'BaudRate',115200);
    set(sphandle,'Parity','none')    
    set(sphandle,'Terminator','LF')        
    fopen(sphandle);
end


function length = lengthFromStruct(S)
    fieldName = fieldnames(S);
    length = 0;
    for n = 1:numel(fieldName)
        [~, fieldLength] = S.(fieldName{n});
        length = length + fieldLength;
    end
end

function CS = validateChecksum(header)
    h = typecast(uint8(header),'uint16');
    a = uint32(sum(h));
    b = uint16(sum(typecast(a,'uint16')));
    CS = uint16(bitcmp(b));
end

function [R] = readToStruct(S, ByteArray)
    fieldName = fieldnames(S);
    offset = 0;
    for n = 1:numel(fieldName)
        [fieldType, fieldLength] = S.(fieldName{n});
        R.(fieldName{n}) = typecast(uint8(ByteArray(offset+1:offset+fieldLength)), fieldType);
        offset = offset + fieldLength;
    end
end