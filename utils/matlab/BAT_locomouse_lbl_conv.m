% This SCRIPT converts all manual labelled file form LocoMouse to general machine learning labelling files in defined directory.

% (c) Si-yang @ PSI, 2019
% (c) LocoMouse and all related codes @ Megan R. Carey Lab.


% ---------------------------------------------- Parameter Area ----------------------------------------------
  data_set_dir = '.\data_set';

  
% ---------------------------------------------- Execution Area ---------------------------------------------- 
  code_dir = pwd;
  rpt_str = 'Label file for dataset: [%s] is successfully converted! (%d of %d)\n';
% Get all names in defined directory
  dir_list = dir(data_set_dir);
  is_sub = [dir_list(:).isdir];    % Check sub-folders in defined directory, returns a logical
% Create file paths
  file_path_name = {dir_list(is_sub).name}';
  file_path_name(ismember(file_path_name, {'.', '..'})) = [];    % Remove general path {.} and {..} in names
  file_path = fullfile(data_set_dir, file_path_name);    % Builds a full sub-folder specification
  n = size(file_path, 1);
% Process labels
  for i = 1:n
      cd(char(file_path(i)));
      lbl = dir('*.mat');
      cd(code_dir);
      locomouse_lbl_conv(fullfile(char(file_path(i)), lbl.name));
      fprintf(rpt_str, char(file_path_name(i)), i, n);
  end
% Clean up extra variables
  vars = {'code_dir', 'rpt_str','dir_list', 'is_sub', 'file_path_name', 'file_path', 'n', 'i', 'lbl'};
  clear(vars{:});
  clear vars;
% Goto target folder
  cd(data_set_dir);
