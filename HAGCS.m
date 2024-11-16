%% HAGCS
function[fmin,bestnest,curve]=HAGCS(n,dim,t_max,fobj,fun,xstfmin)
pa=0.25;
FE=0;
%% Define simple boundaries
n_min=5;
n_max=n;
Lb=-100*ones(1,dim); 
Ub=100*ones(1,dim);

curve=[];
if dim==10
MaxFE=200000;
end
if dim==20
MaxFE=1000000;
end
%% population initialization
for i=1:n
    nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb));
    fitness(i)=feval(fobj,nest(i,:)',fun);
end
[fmin,K]=min(fitness) ;
[fmax,~]=max(fitness);
bestnest=nest(K,:);
FE=FE+n;
ET=Et(n,fmin,fmax,fitness);
Fmin=fmin;
Fmax=fmax;
t=0;
wnest=[];
wfitness=[];

while FE<MaxFE
    t=t+1;
    % Fitness-distance balance with functional weight
    SortedIndex=fitnessDistanceBalanceWithWeightGaussian(nest, fitness);
    if ET>rand*10
        new_nest=get_cuckoos(nest,bestnest,Lb,Ub); 
    else
        % Adaptive global search
        new_nest=adaptive_global_search(nest,Lb,Ub,t,t_max,SortedIndex);
    end
     [fmin,fmax,bestnest,nest,fitness]=get_best_nest(nest,new_nest,fitness,fobj,fun);
      FE=FE+n; 
      SortedIndex=fitnessDistanceBalanceWithWeightGaussian(nest, fitness);
          if fmin-xstfmin>0.01
              Fmin=fmin;
              Fmax=fmax;
          end
          ET=Et(n,Fmin,Fmax,fitness);
      %% Hierarchical nest replacement strategy
              if ET>1
                  for i=1:round(n*0.1)
                      K=rand(size(nest))>pa;
                      stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
                      S=stepsize.*K;
                      new_nest(i,:)=nest(i,:)+S(i,:);
                      s=new_nest(i,:);
                      new_nest(i,:)=simplebounds(s,Lb,Ub);
                  end
                  p1=randperm(round(n*0.1),1);
                  for i=round(n*0.1)+1:round(0.6*n)
                      K1=rand(1,dim)>pa;
                      r1=SortedIndex(p1);
                      S=K1.*(rand.*(nest(r1,:)-nest(i,:)));
                      new_nest(i,:)=nest(i,:)+S;
                      s=new_nest(i,:);
                      new_nest(i,:)=simplebounds(s,Lb,Ub);
                  end
                  p3=randperm(round(n*0.2),1);
                  for i=round(0.6*n)+1:n
                      K2=rand(1,dim)>pa;
                      r3=SortedIndex(p3);
                      S=K2.*(rand*(nest(r3,:)-nest(i,:)));
                      new_nest(i,:)=nest(i,:)+S;
                      s=new_nest(i,:);
                      new_nest(i,:)=simplebounds(s,Lb,Ub);
                  end
              else
                  new_nest=empty_nests(nest,Lb,Ub,pa);
              end
      [fnew,fmax,best,nest,fitness]=get_best_nest(nest,new_nest,fitness,fobj,fun);
      FE=FE+n;
    if fmin-xstfmin>0.01
        Fmin=fmin;
        Fmax=fmax;
    end
    ET=Et(n,Fmin,Fmax,fitness);
    %% Adaptive Population Size Strategy for Serrated Lines
      a=0.9;
      W=0.1;
      fl=(n_max-(((n_max-n_min)/MaxFE)*FE));
      fs=sawtooth(W*t);
      pn=round(a*fl+(1-a)*fl*fs);
      if pn>n_min
      if pn<n
          rn=n-pn;
          if n-rn<n_min
             rn =n-n_min;
          end
          n=n-rn;
          for r=1:rn
              [sortf,index]=sort(fitness,'ascend');
              wnest=[wnest;nest(index(end),:)];
              nest(index(end),:)=[];
              wfitness=[wfitness;fitness(index(end))];
              fitness(index(end))=[];
          end
      end
      end
      if pn>n
          rn=pn-n;
          n=n+rn;
          for r=1:rn
              q=length(wnest);
              p=randperm(q,1);
              nest=[nest;wnest(p,:)];
              fitness=[fitness wfitness(p)];
          end
      end
      
    if fnew<fmin 
        fmin=fnew; 
        bestnest=best;   
    end
    curve=[curve fmin];
end
end




%%  ---------------List of subfunctions------------------
%% Entropy of solution distribution
function [Et]=Et(n,fmin,fmax,fitness)
Dist=linspace(fmin,fmax,n+1);
for i=1:n
    S(i,:)=[Dist(i),Dist(i+1)];
end
s=zeros(1,n);
p=zeros(1,n);
for i=1:n
    for j=1:n
        if fitness(i)>=S(j,1)&&fitness(i)<=S(j,2)
            s(j)=s(j)+1;
        end
    end
end

for i=1:n
    p(i)=s(i)/n;
end

for i=1:n
    if p(i)~=0
        P(i)=p(i).*log(p(i)); 
    else
        P(i)=0;
    end
end
Et=-sum(P);
end


%% Lévy flight Random Walk
function nest=get_cuckoos(nest,best,Lb,Ub)
n=size(nest,1);
beta=3/2;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

for j=1:n
    s=nest(j,:);
    %% Lévy flight
    u=randn(size(s))*sigma;
    v=randn(size(s));
    step=u./abs(v).^(1/beta);
    stepsize=0.01*step.*(s-best);
    s=s+stepsize.*randn(size(s));
   nest(j,:)=simplebounds(s,Lb,Ub);
end
end

%% Adaptive global search
function nest=adaptive_global_search(nest,Lb,Ub,t,t_max,SortedIndex)
n=size(nest,1);
for j=1:n
    A=SortedIndex(1);
    C=SortedIndex(end);
    c=nest(C,:);
    s=nest(A,:);
    a=2.*cos((pi/3)*(1+t/(2*t_max)));
    stepsize=(s-c);
    s=nest(j,:)+a.*stepsize;
    nest(j,:)=simplebounds(s,Lb,Ub);
end
end


%%  Find the best nest
function [fmin,fmax,best,nest,fitness]=get_best_nest(nest,newnest,fitness,fobj,fun)
for j=1:size(nest,1)
    fnew=feval(fobj,newnest(j,:)',fun);
    if fnew<=fitness(j)
       fitness(j)=fnew;
       nest(j,:)=newnest(j,:);
    end
end
[fmin,K]=min(fitness) ;
best=nest(K,:);
[fmax,~]=max(fitness);
end

%% Abandon nest, replace old solutions with new ones
function new_nest=empty_nests(nest,Lb,Ub,pa)
n=size(nest,1);
K=rand(size(nest))>pa;

stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
new_nest=nest+stepsize.*K;
for j=1:size(new_nest,1)
    s=new_nest(j,:);
  new_nest(j,:)=simplebounds(s,Lb,Ub);  
end
end

%% Simple boundary
function s=simplebounds(s,Lb,Ub)
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);

  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  
  s=ns_tmp;
end
