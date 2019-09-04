dirs=dir('C:/Users/LucasX/Desktop/Masks/*.png');
for n=1:numel(dirs)
     strname=strcat('C:/Users/LucasX/Desktop/Masks/',dirs(n).name);
     img=imread(strname);
     [x,map]=rgb2ind(img,256);
     newname=strcat('C:/Users/LucasX/Desktop/Masks/',dirs(n).name);
     imwrite(x,map,newname,'png');
end
