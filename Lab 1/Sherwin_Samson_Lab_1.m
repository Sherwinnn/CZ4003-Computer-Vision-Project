Pc = imread('mrt-train.jpg');
whos Pc;
%%
P = rgb2gray(Pc);
imshow(P)
%%
minP = min(P(:));
maxP = max(P(:));
%%
sub = imsubtract(P,double(minP));
P2 = immultiply(sub,255/191);
minP2 = min(P2(:));
maxP2 = max(P2(:));
imshow(P2);
%% 2.2 (a)
imhist(P,10);
%% Histogram with 256 bins
imhist(P,256);
%% (b) Histogram equalization with 255 bins
P3 = histeq(P,255);
imhist(P3,10);
%%
imhist(P3,256);
%%
P4 = histeq(P3,255);
imhist(P4,10);
%%
imhist(P4,256);
%% 2.3 (a) sigma = 1.0, X & Y = 5
sigma1 = 1.0;
size1 = -2:2;
[X1,Y1] = meshgrid(size1);
H1 = ( 1 / ( 2 * pi * sigma1.^2) ) * exp( -((X1).^2 + (Y1).^2)/(2*sigma1.^2));
mesh(H1);
%% 2.3 (a) sigma = 2.0, X & Y = 5
sigma2 = 2.0;
size2 = -2:2;
[X2,Y2] = meshgrid(size2);
H2 = ( 1 / ( 2 * pi * sigma2.^2) ) * exp( -((X2).^2 + (Y2).^2)/(2*sigma2.^2));
mesh(H2);
%% 2.3 (b)
Pic1 = imread("ntu-gn.jpg");
figure
imshow(Pic1);
%% 2.3 (c) (i) ntu-gn.jpg with Filter H1
Pic1_Filter1 = uint8(conv2(Pic1,H1));
imshow(Pic1_Filter1);
%% 2.3 (c) (ii) ntu-gn.jpg with Filter H2
Pic1_Filter2 = uint8(conv2(Pic1,H2));
imshow(Pic1_Filter2);
%% 2.3 (d)
Pic2 = imread("ntu-sp.jpg");
imshow(Pic2);
%% 2.3 (e) ntu-sp.jpg with Filter H1
Pic2_Filter1 = uint8(conv2(Pic2,H1));
imshow(Pic2_Filter1);
%% 2.3 (e) ntu-sp.jpg with Filter H2
Pic2_Filter2 = uint8(conv2(Pic2,H2));
imshow(Pic2_Filter2);
%% 2.4 (i) Gaussian noise
Pic1 = imread("ntu-gn.jpg");
figure
imshow(Pic1);
Pic1_3x3 = uint8(medfilt2(Pic1,[3,3]));
figure
imshow(Pic1_3x3);
Pic1_5x5 = uint8(medfilt2(Pic1,[5,5]));
figure
imshow(Pic1_5x5);
%% 2.4 (ii) Speckle noise
Pic2 = imread("ntu-sp.jpg");
figure
imshow(Pic2);
Pic2_3x3 = uint8(medfilt2(Pic2,[3,3]));
figure
imshow(Pic2_3x3);
Pic2_5x5 = uint8(medfilt2(Pic2,[5,5]));
figure
imshow(Pic2_5x5);
%% 2.5 (a)
Pic3 = imread("pck-int.jpg");
figure
imshow(Pic3);
%% 2.5 (b)
Pic3ft = fft2(Pic3);
Pic3S = abs(Pic3ft).^2/ length(Pic3);
figure
imagesc(fftshift(Pic3S.^0.1)); %10th root
colormap('default')
%% 2.5 (c)
imagesc(Pic3S.^0.1);
colormap('default');
% Location of peaks a1,b1 and a2,b2 = 249,17 and 9,241
%% 2.5 (d) Set neighbourhood 5x5 elements = 0
a1 = 249;b1 = 17;
a2 = 9; b2 = 241;
Pic3ft(b1-2 : b1+2, a1-2 : a1+2) = 0;
Pic3ft(b2-2 : b2+2, a2-2 : a2+2) = 0;
newS = abs(Pic3ft).^2 / length(Pic3);
figure
imagesc(fftshift(newS.^0.1));
colormap('default');
%% 2.5 (e) Inverse Fourier Transform
Pic3Inv = uint8(ifft2(Pic3ft));
figure
imshowpair(Pic3,Pic3Inv, 'montage')
title('Initial Image vs Removed Peaks Image')
%% Improve
a1 = 249;b1 = 17;
a2 = 9; b2 = 241;

Pic3ft(b1, :) = 0;
Pic3ft(b2, :) = 0;
Pic3ft(:, a1) = 0;
Pic3ft(:, a2) = 0;
Pic3ft(b2-2 : b2+2, a2-2 : a2+2) = 0;
Pic3ft(b1-2 : b1+2, a1-2 : a1+2) = 0;

S = abs(Pic3ft).^2 / length(Pic3);
figure
title('Interference Lines identified along the axes ')

imagesc(fftshift(log10(S)))
newPckPeak = uint8(ifft2(Pic3ft));
figure
imshowpair(Pic3Inv,real(newPckPeak), 'montage')
title('Initial Removed Peak Image VS Removed Peak & Lines')
%%
pckMinP = double(min(newPck(:)));
pckMaxP = double(max(newPck(:)));
newPckContrast = uint8(255*(double(newPck) - pckMinP) / (pckMaxP - pckMinP));
figure
imshowpair(real(newPckPeak),real(newPckContrast), 'montage')
title('Removed Peak & Lines VS Removed Peak & Lines + Contrast Stretched')

figure
imshowpair(Pic3Inv,real(newPckContrast), 'montage')
title('Initial Removed Peak Image VS After Improvements')
%% 2.5 (f)
Pic4 = imread('primatecaged.jpg');
Pic4 = rgb2gray(Pic4);
figure
imshow(Pic4);
%%
Pic4ft = fft2(Pic4);
Pic4S = abs(Pic4ft).^2/ length(Pic4);
figure
imagesc(fftshift(Pic4S.^0.1)); %10th root
colormap('default')
%% Find Location of peaks
imagesc(Pic4S.^0.1);
colormap('default');
% p1,q1 = 253,11; p2,q2 = 6,247;
% p3,q3 = 248,21; p4,q4 = 10,236;
%% Set neighbourhood 5x5 elements = 0
p1 = 253; q1 = 11; p2 = 6; q2 = 247;
p3 = 248; q3 = 21; p4 = 10; q4 = 236;
Pic4ft(p1-2 : p1+2, q1-2 : q1+2) = 0;
Pic4ft(p2-2 : p2+2, q2-2 : q2+2) = 0;
Pic4ft(p3-2 : p3+2, q3-2 : q3+2) = 0;
Pic4ft(p4-2 : p4+2, q4-2 : q4+2) = 0;
newPic4S = abs(Pic4ft).^2 / length(Pic4);
figure
imagesc(fftshift(newPic4S.^0.1));
colormap('default');
%% Inverse Fourier Transform
Pic4Inv = uint8(ifft2(Pic4ft));
figure
imshow(Pic4Inv)
%% Improve
p1 = 253; q1 = 11; p2 = 6; q2 = 247;
p3 = 248; q3 = 21; p4 = 10; q4 = 236;

Pic4ft(q1, :) = 0;
Pic4ft(q2, :) = 0;
Pic4ft(q3, :) = 0;
Pic4ft(q4, :) = 0;
Pic4ft(:, p1) = 0;
Pic4ft(:, p2) = 0;
Pic4ft(:, p3) = 0;
Pic4ft(:, p4) = 0;

Pic4ft(248 : 254, 1 : 8.5) = 0;
Pic4ft(3 : 10, 2: 6) = 0;
Pic4ft(248:252 , 250: 252) = 0;
Pic4ft(2:7.5 , 249.5: 255) = 0;


S = abs(Pic4ft).^2 / length(Pic4);
imagesc(real((log10(S))))
imagesc(real(fftshift(log10(S))))
newPrimatePeak = uint8(ifft2(Pic4ft));
title('New Peaks & Interference lines identified ')
figure
imshowpair(real(Pic4Inv),real(newPrimatePeak), 'montage')
title('Initial Removed Peak Image VS Removed Peak & Lines')
%%
primateMinP = min(double(newPrimate(:)));
primateMaxP = max(double(newPrimate(:)));
newPrimateContrast = uint8(255*(double(newPrimate) - pckMinP) / (pckMaxP - pckMinP));
figure
imshowpair(real(Pic4Inv),real(newPrimateContrast), 'montage')
title('Initial Removed Peak Image VS After Improvements')
%% 2.6 (a)
Book_Pic = imread('book.jpg');
imshow(Book_Pic);
%% 2.6 (b)
[X,Y] = ginput(4);
xTarget = [0 210 210 0];
yTarget = [0 0 297 297];
%% 2.6 (c)
A = [ X(1) Y(1) 1 0 0 0 -xTarget(1)*X(1) -xTarget(1)*Y(1);
   0 0 0 X(1) Y(1) 1 -yTarget(1)*X(1) -yTarget(1)*Y(1);
   X(2) Y(2) 1 0 0 0 -xTarget(2)*X(2) -xTarget(2)*Y(2);
   0 0 0 X(2) Y(2) 1 -yTarget(2)*X(2) -yTarget(2)*Y(2);
   X(3) Y(3) 1 0 0 0 -xTarget(3)*X(3) -xTarget(3)*Y(3);
   0 0 0 X(3) Y(3) 1 -yTarget(3)*X(3) -yTarget(3)*Y(3);
   X(4) Y(4) 1 0 0 0 -xTarget(4)*X(4) -xTarget(4)*Y(4);
   0 0 0 X(4) Y(4) 1 -yTarget(4)*X(4) -yTarget(4)*Y(4);];
v = [xTarget(1);
  yTarget(1);
  xTarget(2);
  yTarget(2);
  xTarget(3);
  yTarget(3);
  xTarget(4);
  yTarget(4);];
u = A \ v;
U = reshape([u;1],3,3)';
w = U*[X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:))
%% 2.6 (d) Warp the Image
T = maketform('projective', U');
P2 = imtransform(Book_Pic, T, 'XData', [0 210], 'YData', [0 297]);
%% 2.6 (e) Display the image:
imshow(P2)
