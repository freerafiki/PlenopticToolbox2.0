fid = fopen('Plantball/FocusedViews_7x7/Other/LF.txt');
finished = false;
%numberofimages
read_the_number = false;
structure_shape = [13, 13, 689, 1061, 3];
%create the LF
created_LF_structure = false;
while ~finished 
    if read_the_number && ~created_LF_structure
        LF = zeros(structure_shape);
        created_LF_structure = true;
    end
    tline = fgets(fid);
    if tline == -1
        finished = true;
        if ~read_the_number
            fprintf('ERROR');
        end
    else
        info = strsplit(tline);
        if ~read_the_number
            structure_shape = [str2double(info(1)), str2double(info(2)), str2double(info(3)), str2double(info(4)), str2double(info(5))];
            read_the_number = true;
        else
            image_path = info(1);
            img = imread(image_path{1});
            position_s = str2double(info(2)) + 1;
            position_t = str2double(info(3)) + 1;
            LF(position_s, position_t, :, :, :) = double(img) ./255;
        end
    end
end
save('LF')
            
        
        

