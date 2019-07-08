load('label.mat');
path = pwd;
cd('.\data');
file=dir('*.png'); 
[k len]=size(file);
for i=1:k  
    name=file(i).name;  
    A=imread(name);
    B(i,:)=reshape(A,1,[]);
    if label(i) == 0;
        label_set(:,i) = [1 0 0 0 0 0 0 0 0 0];
    elseif label(i) == 1;
        label_set(:,i) = [0 1 0 0 0 0 0 0 0 0];
    elseif label(i) == 2;
        label_set(:,i) = [0 0 1 0 0 0 0 0 0 0];
    elseif label(i) == 3;
        label_set(:,i) = [0 0 0 1 0 0 0 0 0 0];
    elseif label(i) == 4;
        label_set(:,i) = [0 0 0 0 1 0 0 0 0 0];
    elseif label(i) == 5;
        label_set(:,i) = [0 0 0 0 0 1 0 0 0 0];
    elseif label(i) == 6;
        label_set(:,i) = [0 0 0 0 0 0 1 0 0 0];
    elseif label(i) == 7;
        label_set(:,i) = [0 0 0 0 0 0 0 1 0 0];
    elseif label(i) == 8;
        label_set(:,i) = [0 0 0 0 0 0 0 0 1 0];
    else label(i) == 9;
        label_set(:,i) = [0 0 0 0 0 0 0 0 0 1];
    end
  
end
cd(path);
[pc,score,latent] = pca(double(B));
p_d_ratio=cumsum(latent)./sum(latent); 
for i = 1:784
    if (p_d_ratio(i) >=0.9)
        ration_number = i;
        break;
    end
    i=i+1;
end
B=score(:,1:ration_number);
    for c=1:k
        if c<=4000;
            train_feature(c,:) = B(c,:);
            train_label(:,c) = label_set(:,c);
        else
            test_feature(c-4000,:) = B(c,:);
            test_label(:,c-4000) = label_set(:,c);
        end
    end
    P = train_feature';
    T = train_label;
    net = newff(P,T,50,{'tansig','purelin'},'traincgp');
    net.trainParam.lr=0.2;
    net.trainParam.goal=0.01;
    net.trainParam.epochs=1000;
    net=train(net,P,T);
    save ('.\ann_model.mat','net');
    an=sim(net,test_feature');
    plotconfusion(test_label, an);
    clear;


    

    
    
    
    
    
    
    
    
    
    
    
    
    