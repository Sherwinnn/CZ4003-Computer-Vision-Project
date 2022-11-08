%% 3.1 (a)
img = imread('macritchie.jpg');
img = rgb2gray(img);

figure
imshow(img)

%% 3.1 (b)
sobel_vertical = [-1 0 1; -2 0 2; -1 0 1;];
sobel_horizontal = [-1 -2 -1; 0 0 0; 1 2 1;];

sobel_v_img = conv2(img, sobel_vertical);
figure
imshow(uint8(sobel_v_img))

sobel_h_img = conv2(img, sobel_horizontal); 
figure
imshow(uint8(sobel_h_img)) 

%%

figure
imshow(uint8(abs(sobel_v_img)))

figure
imshow(uint8(abs(sobel_h_img)))

%% 3.1 (c)

E = sobel_v_img.^2 + sobel_h_img.^2;
imshow(uint8(E));

%%
E = sqrt(E);
imshow(uint8(E));

%% 3.1 (d)

t1 = 10000;
Et1 = E>t1;
imshow(Et1)
figure

t2 = 30000;
Et2 = E>t2;
imshow(Et2)
figure

t3 = 50000;
Et3 = E>t3;
imshow(Et3)
figure

t4 = 80000;
Et4 = E>t4;
imshow(Et4)


%% 3.1 (e) i

tl = 0.04;
th = 0.1;

sigma1 = 1.0;
E = edge(img, 'canny', [tl th], sigma1);
figure
imshow(E)

sigma2 = 2.0;
E = edge(img, 'canny', [tl th], sigma2);
figure
imshow(E)

sigma3 = 3.0;
E = edge(img, 'canny', [tl th], sigma3);
figure
imshow(E)

sigma4 = 4.0;
E = edge(img, 'canny', [tl th], sigma4);
figure
imshow(E)

sigma5 = 5.0;
E = edge(img, 'canny', [tl th], sigma5);
figure
imshow(E)
%% 3.1 (e) ii

th = 0.1;
sigma = 1.0;

tl1 = 0.01;
E1 = edge(img, 'canny' ,[tl1 th], sigma);
figure
imshow(E1);

tl2 = 0.02;
E2 = edge(img, 'canny' ,[tl2 th], sigma);
figure
imshow(E2);

tl3 = 0.04;
E3 = edge(img, 'canny' ,[tl3 th], sigma);
figure
imshow(E3);

tl4 = 0.08;
E4 = edge(img,'canny',[tl4 th],sigma);
figure
imshow(E4);

%% 3.2 (a)

P = imread('macritchie.jpg');
I = rgb2gray(P);
tl = 0.04; th = 0.1; sigma = 1.0;

E = edge(I,'canny',[tl th],sigma);
imshow(E);

%% 3.2 (b)
[H, xp] = radon(E);
imshow(uint8(H));


%%
[radius, theta] = find(H>=max(max(H)));
radius; theta;

%%
help radon

%% 3.2 (c)
imagesc(uint8(H));
colormap('default');

%% 3.2 (d)
theta = 103;
radius = xp(157);
[A, B] = pol2cart(theta*pi/180, radius);
B = -B;


%%
[y_size, x_size] = size(I);
x_centre = x_size/2;
y_centre = y_size/2;
C = A*(A+x_centre) + B*(B+y_centre);

x_centre;
y_centre;

%% 3.3 (e)
xl = 0;
yl = (C - A * xl) / B;

xr = 357;
yr = (C - A * xr) / B;

yl;
yr;
%% 3.3 (f)
imshow(I)
line([xl xr], [yl yr])

%% 3.3 (b)
l = imread('corridorl.jpg'); 
l = rgb2gray(l);
figure
imshow(l);

r = imread('corridorr.jpg');
r = rgb2gray(r);
figure
imshow(r);

%% 3.3 (c)
D = Dmap(l, r);
figure
imshow(D,[-15 15]);

res = imread('corridor_disp.jpg');
figure
imshow(res);

%% 3.3 (d)
l = imread('triclopsi2l.jpg'); 
l = rgb2gray(l);
figure
imshow(l);

r = imread('triclopsi2r.jpg');
r = rgb2gray(r);
figure
imshow(r);

D = Dmap(l, r);
figure
imshow(D,[-15 15]);

res = imread('triclopsid.jpg');
figure
imshow(res);



%% 3.3 (a)

function res_map = Dmap(I_l, I_r)
[x,y] = size(I_l);
temp_x = 11; temp_y = 11;
center_x = floor(temp_x/2); center_y = floor(temp_y/2);
search_lim = 14;
res_map = ones(x-temp_x + 1, y-temp_y + 1);

for i = 1+center_x: x-center_x
    for j = 1+center_y: y-center_y
        curIpatch_r = I_l(i-center_x : i+center_x, j-center_y : j+center_y);
        curIpatch_l = rot90(curIpatch_r, 2);
        min_diff = inf; min_coor = j;
        for k = max(1+center_y, j-search_lim) : j
            T = I_r(i-center_x : i+center_x, k-center_y : k+center_y);
            curIpatch_r = rot90(T,2);

            c_1 = conv2(T,curIpatch_r); c_2 = conv2(T,curIpatch_l);
            ssd = c_1(temp_x, temp_y)-2*c_2(temp_x, temp_y);
            if ssd < min_diff
                min_diff = ssd;
                min_coor = k;
            end
        end
        res_map(i-center_x, j-center_y) = j-min_coor;
    end
end
end

