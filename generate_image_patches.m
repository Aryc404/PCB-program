function R_X = generate_image_patches(I, N_X, N_b)
% Generate random image patches randomly rotated by multiples of 90 degrees
% I: image data, N_X: number of pixels, N_b: number of image patches
R_X = zeros(2*N_X, N_b); % For ON and OFF
n_x = sqrt(N_X);
im_size = size(I);
buff = 20; % Buffer size that exclude image patches close to boundaries

for i = 1 : N_b
    while true
        im_x = buff-1 + randi(im_size(1) - n_x - 2*buff);
        im_y = buff-1 + randi(im_size(2) - n_x - 2*buff);
        im_z = randi(im_size(3));
        
        im = I(im_x+(1:n_x), im_y+(1:n_x), im_z);
        im = rot90(im, randi(4)-1);
        im = im(:);
        if any(isnan(im))
            continue;
        end
        r_x = im * 70;
        pos_r = r_x > 0;
        R_X(1:N_X, i) = r_x .* pos_r; % ON responses
        R_X(N_X+1:end, i) = -r_x .* ~pos_r; % OFF responses
        break;
    end
end
end
