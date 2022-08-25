fp = fopen('all.txt','rt');
i=0;
while not(feof(fp))
    i=i+1
    file = fgetl(fp);
    info = niftiinfo(file);
    I{i} =double(niftiread(file));
    
end
fclose(fp);

maxI=65535;


X = linspace(0,65535,65535);

for i=1:length(I)
    NX{i} = hist(double(I{i}(I{i}(:)>0)),X);
end

for i=1:length(I)
    CNX{i} = cumsum(NX{i});
    CNX{i} = CNX{i}/max(CNX{i});
end


options.MaxFunEvals=5e4
for i=1:length(I)
    disp(i)
    cnxD{i} = fminsearch(@(x) nanmean((CNX{1}-interp1([-1000 x(1)*X+x(2)],[0 CNX{i}],X)).^2),[1 0],options);
    cCNX{i} = interp1(cnxD{i}(1)*X+cnxD{i}(2),CNX{i},X);
end

figure(1)
clf
for i=1:length(I)
    plot(X,cCNX{i},':')
    hold on
end
hold off
title('CDF After')


figure(2)
clf
for i=1:length(I)
    plot(X,CNX{i},':')
    hold on
end
hold off
title('CDF Before')

figure(3); 
clf
imagesc(I{1}(:,:,floor(end/2)))
colorbar
title('One slice')
caxis([0 5e4])

% AFTER
M = I{1}(:,:,floor(end/2));
II=M;
for i=1:length(I)
    M(:,:,1,i)=imresize(cnxD{i}(1)*I{i}(:,:,floor(end/2))+cnxD{i}(2),size(II));
end

for i=1:50:length(I)
    figure, 
    montage(M(:,:,1,i:min(length(I),i+50)), 'DisplayRange'  ,[0 500])
    title('example after')
end


% BEFORE

M = I{1}(:,:,floor(end/2));
II=M;
for i=1:length(I)
    M(:,:,1,i)=imresize(I{i}(:,:,floor(end/2)),size(II));
end

for i=1:50:length(I)
    figure, 
    montage(M(:,:,1,i:min(length(I),i+50)), 'DisplayRange'  ,[0 500])
    title('example before')
end

fp = fopen('all.txt','rt');
i=0;
while not(feof(fp))
    i=i+1
    file = fgetl(fp);
    info = niftiinfo(file);    
    II = cnxD{i}(1)*I{i}+cnxD{i}(2);
    info.Datatype='double';
    info.MultiplicativeScaling=1;
    info.raw.scl_slope = 1;
    [P,F,E] = fileparts(file);
    niftiwrite(double(II),[P filesep F '_scaled' E], info);
end
fclose(fp);