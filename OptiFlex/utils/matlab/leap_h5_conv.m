% This function converts LEAP H5 data to PNG frames.

% (c) Si-yang @ PSI, 2019
% (c) LEAP and all related codes @ Talmo Pereira (talmo@princeton.edu).

% Function Input: dataFile  (str): *.h5 image-data with LEAP definition.
%                 outPath   (str): output path to store converted data.

function leap_h5_conv(dataFile, outPath)
  %% Get required information from inputs
  % Get dataset basic information
    [~, dataName, ~] = fileparts(dataFile);
    dataInfo = h5info(dataFile, '/box');  % H5 file with LEAP definition
    frameCount = dataInfo.Dataspace.Size(4);
    frameNum = strcat('%0', num2str(ceil(log10(frameCount)) + 1), 'd');

  % Prepare output leaf directory
    outDir = fullfile(outPath, dataName);
    mkdir(outDir);

  %% Main processing loop
    for i = 1 : frameCount
        frameName = strcat('frm_', num2str(i, frameNum), '.png');
        framePath = fullfile(outDir, frameName);
        frame = h5readframes(dataFile, 'box', i);  % function form LEAP
        imwrite(frame, framePath);
    end
end
