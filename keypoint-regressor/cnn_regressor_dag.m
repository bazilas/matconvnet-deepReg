function [net, info] = cnn_regressor_dag(varargin)

%Dataset
opts.datas='LSP';

%LSP params for augmentation
opts.patchHei=120;
opts.patchWi=80;

%Camera
opts.cam=1;

%augmentation
opts.aug=0;
opts.NoAug=0;

opts.expDir = sprintf('/data/vb/Temp/%s-baseline%d',opts.datas,opts.cam) ;
opts.imdbPath = fullfile(opts.expDir, sprintf('imdb%d.mat',opts.cam));

opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.derOutputs= {'objective', 1} ;
opts.train.learningRate = [0.001*ones(1, 17) 0.0005*ones(1, 50) 0.002*ones(1, 500)  0.03*ones(1, 130) 0.01*ones(1, 100)] ;
opts.train.momentum=0.9;
opts.useBnorm = false ;
opts.train.prefetch = true ;

%GPU
opts.train.gpus = [];

%architecture parameters
opts.initNet=''; %pre-trained network
opts.outNode=28;%predicted values
opts.inNode=3;%R, G, B channels
opts.lossFunc='tukeyloss'; %loss functions: tukeyloss / l2loss
opts.errMetric = 'mse-combo';
opts.train.thrs=0;
opts.train.refine=false; %cascade model (not integrated in the demo)

%convert the error in pixels using the scbox
opts.train.scbox=opts.patchWi*ones(opts.outNode,1);
opts.train.scbox(2:2:end)=opts.patchHei;

%cross validation
opts.cvset=[];
opts.cvidx=[];

%Dataset Path, OSX / Ubuntu
opts.DataMatTrain=sprintf('/mnt/ramdisk/vb/%s/%s_imdbsT%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);
opts.DataMatVal=sprintf('/mnt/ramdisk/vb/%s/%s_imdbsV%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);

%load network
net = [];

%parse settings
[opts, trainParams] = vl_argparse(opts, varargin); %main settings
[opts.train, boptsParams]= vl_argparse(opts.train, trainParams); %train settings
net=boptsParams{1}.net; %network
clear trainParams boptsParams;

useGpu = numel(opts.train.gpus) > 0 ;

%Paths OSX / Ubuntu
opts.train.expDir = opts.expDir ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
else
    if isempty(opts.cvset)
        imdb = getDACHImdb(opts) ; %normal training
    else
        %imdb = getCVImdb(opts) ; %cross validation
    end
    
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

fn = getBatchDagNNWrapper(useGpu) ;
  
info = cnn_train_dag_regressor(net, imdb, fn, opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, useGpu)
% -------------------------------------------------------------------------

[im, lab] = getBatch(imdb, batch);

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', lab} ;
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,batch) ;


% --------------------------------------------------------------------
function imdb = getDACHImdb(opts)
% --------------------------------------------------------------------

load(opts.DataMatTrain); %training data
imdbTr=imdb;
clear imdb;

%permute the indices
s = RandStream('mt19937ar','Seed',0);
idx=randperm(s,(length(imdbTr)));

trN=0;
for i=1:length(imdbTr)
    data(:,:,:,trN+i)=imdbTr((idx(i))).im;
    %labels{1,trN+i}=imdbTr((idx(i))).cl; %classification
    labels{1,trN+i}=imdbTr((idx(i))).y; %y or y3D
    sets(1,trN+i)=imdbTr((idx(i))).set;
    fr(trN+i)=imdbTr((idx(i))).fr;
end

dataMean = mean(single(data), 4); %only with training data

load(opts.DataMatVal); %validation data
imdbV=imdb;
clear imdb;

trN=size(data,4);
for i=1:length(imdbV)
    data(:,:,:,trN+i)=imdbV((i)).im;
    %labels{1,trN+i}=imdbV((i)).cl; %classification
    labels{1,trN+i}=imdbV((i)).y; %y or y3D
    sets(1,trN+i)=imdbV((i)).set;
    fr(trN+i)=imdbV((i)).fr;
end

data = single(data);
data = bsxfun(@minus, data, dataMean); %subtract mean

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
%imdb.images.data_std = dataStd;
imdb.images.labels = labels ;
imdb.images.set = sets;
imdb.frames=fr;
imdb.idx=idx;
imdb.meta.sets = {'train', 'val', 'test'} ;