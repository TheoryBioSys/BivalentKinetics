%********************************************************************************
%***************** Copyright (C) 2021-2022, Richard Spinney.*********************
%********************************************************************************
% //                                                                           //
% //    This program is free software: you can redistribute it and/or modify   //
% //    it under the terms of the GNU General Public License as published by   //
% //    the Free Software Foundation, either version 3 of the License, or      //
% //    (at your option) any later version.                                    //
% //                                                                           //
% //    This program is distributed in the hope that it will be useful,        //
% //    but WITHOUT ANY WARRANTY; without even the implied warranty of         //
% //    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
% //    GNU General Public License for more details.                           //
% //                                                                           //
% //    You should have received a copy of the GNU General Public License      //
% //    along with this program.  If not, see <http://www.gnu.org/licenses/>.  //
% //                                                                           //
% ///////////////////////////////////////////////////////////////////////////////

%%%%%%%% PURPOSE %%%%%%%%%%%%%%

% This code samples the stationary distribution of a lattice of receptor
% sites interacting with a popoulation of bivalent molecules using a
% metropolis hastings MCMC scheme.

% It runs in three modes
% 1. Find stationary distribution of cross bound occupancy for n1=n2=4 (square) lattice
% 2. Find stationary distribution of cross bound occupancy for n1=4,n2=2 (square with regular holes) lattice
% 3. Find probability of percolation (any occupancy) for n1=n2=4 (square) lattice
% These are controlled with the lattice42 and percolation flags

%%%%%%% GENERAL STRATEGY %%%%%%%%

%   accept/reject sampling using known free energy differences like Ising
%   etc. Difference is that since neighbouring sites may need to be matched
%   we accept/reject a *pair* of neighbouring sites.
%   We are also greedy - we ignore proposals
%   that we know have probability of 0 and propose only possible exchanges.
%   Correct sampling distibution guaranteed by symmetric nature of proposal
%   distribution at all points in the Metropolis algorithm.

%   i.e. we use acceptance probability for proposed transition xold -> xnew as
%   follows

%   PA(xnew|xold) = min (1, (P(xnew)/P(xold)) * (q(xold|xnew)/q(xnew|xold)) )

%   where P(.) is the Gibbs distribution such that (P(xnew)/P(xold)) is
%   exponetiated free energy difference, and q is proposal distribution.
%   We also have (q(xold|xnew)/q(xnew|xold)) = 1 achieved here by using flat proposal 
%   distributions (i.e. uniform across proposed states) independent of
%   state, in conjunction with micro-reversibility

%%%%%%% BINDING STATES %%%%%%%%%%

%   1 == vacant
%   2 == singly bound
%   3 == cross bound horizontal - left - pointing "right"
%   4 == cross bound horizontal - right - pointing "left"
%   5 == cross bound vertical - bottom - pointing "up"
%   6 == cross bound vertical - top - pointing "down"


%%%%%% WARNING %%%%%%%%%%%%

%   The system suffers from dynamical slowing around c0 = k1 (in a
%   range of ~3-5 orders of magnitude e.g. c0 ~ 10^-11 - 10^-7 for
%   k1=10^-9). As such long burn in times can be required causing the
%   simulation to be slow to get faithful statistics. 
%   General advice is to start with smaller lattices/burn times and 
%   increase both whilst testing for convergence.
%   Data used in the paper utilised extremely long runs to achieve good
%   statistics.

clear all;
close all;

has_parallel = license('test','Distrib_Computing_Toolbox');

if (has_parallel)
    M=8; %guess at number of hardware threads
else
    M=0;
end

% are we sampling percolation or general occupancy statistics?
percolation = 0;

% use n1=4, n2=2 lattice variant? otherwise default to n1=n2=4
lattice42 = 0;

if (percolation && lattice42)
    disp("cannot test percolation for 4-2 lattice, use 4-4 only");
    return;
end

%total number of accepted/rejected exchanges per parameter value
tau=4.5e8;
%burn in time before sampling
min_t=0.9*tau;%tau=1e8;
%sampling frequency
del_t = 1000;

%lattice size
L=300;

%common parameter values
k1 = 10^(-9);
cc = 10^(-6);

%start, finish, and delta exponent for c0
ex1=-15; %starting exponent
ex2=-2; %end exponent
dex=0.1; %delta exponent
dex=1.0;

if (percolation)
    %start and end occupancies (approx.)
    pa = 0.5; 
    pb = 0.6;

    %convert occupancies to concentrations using approx. theory
    c01=(8*cc + 2*k1 - 2*sqrt(k1^2 + 4*cc*k1*(-2 + pa)*(-1 + pa) +    16*(cc^2)*(-1 + pa)^2) - (8*cc + k1)*pa)/(-1 + pa);
    c02=(8*cc + 2*k1 - 2*sqrt(k1^2 + 4*cc*k1*(-2 + pb)*(-1 + pb) +    16*(cc^2)*(-1 + pb)^2) - (8*cc + k1)*pb)/(-1 + pb);

    %to exponents
    ex1 = log10(c01);
    ex2 = log10(c02);
    dex = (ex2-ex1)/22; %arbitrary choice
end

%neighbour indices
Xp=ones(L,1);
Xm=ones(L,1);

for i=1:L
    Xp(i)=i+1;
    Xm(i)=i-1;
end

Xp(L)=1; %periodic bc
Xm(1)=L;

num_loops = 1 +  ceil((ex2-ex1)/dex);
Clist = ones(num_loops,1); % concentrations
Pcross = ones(num_loops,1); % prob of cross bound occupancy - empirical
Pocc = ones(num_loops,1);  %prob of occupancy - empirical/sampled
Poccref = ones(num_loops,1); %prob of occupncy - analytical
Pperc = ones(num_loops,1); %prob of percolation
Pcrossref = ones(num_loops,1); %prob of cross bound occupancy - analytical

%parallel for loop - defaults to regular for loop if M=0, i.e. when no
%parallelism available.
parfor (count = 1:num_loops,M) 
    
    exponent = ex1 + dex * (count-1);
    
    % exponent to concentration 
    c0 = 10^exponent;
    p1=exp(log(c0/k1)); %singly bound baton measure (propto)
    p12=exp(log((c0*cc)/(k1*k1))); %cross bound baton measure (propto)
    
    %percolation statistics
    perc_attempt = 0;
    perc_success = 0;

    %distribution statistics
    run_count = 0;
    Pcross(count)=0;
    Pocc(count)=0;
    
    %lattice
    X=ones(L,L); 
    
    %number of proposed configurations to attempt to swap
    attempts=0;
    
    for t=1:tau   

        if (M==0) && (mod(t,0.01*tau)==0)
            fprintf('index = %i of %i, c0 = %s index progress: %f percent complete.\r',count,num_loops,c0,100*t/tau);
            fprintf('average configuration attempts per timestep= %f \r',attempts/t);
        end
        
        %initial values - stop matlab from complaining
        independent = [1,1];   
        Pold=1;
        horiz =0;
        s1=0;
        s2=0;
        pos1=[1,1];
        pos2=pos1;
        Pnew=1;
            
        while 1 %loop until we find a valid configuration to propose
    
            attempts=attempts+1;
            
            r=rand(3,1);
            %three random numbers
            % 1. xpos
            % 2. ypos
            % 3. orientation of proposed site pair
            
            pos1 = [ceil(L*r(1)),ceil(L*r(2))]; %random lattice site
            
            if (lattice42) %1/4 of sites are removed
                if (mod(pos1(1),2)==0) &&  (mod(pos1(2),2)==0) %hit a null lattice point
                    continue;
                end
            end
            
            pos2 = pos1; %position of paired site
            independent = [1,1]; %record of whether site is (not) bound to another
            
            if (r(3)<0.5) %vertical
                horiz = 0; %flag not horizontal
                pos2=[pos1(1),Xp(pos1(2))]; %extend "up" (doesn't matter up or down since sampling all points)
                
                if (lattice42)
                    if (mod(pos2(1),2)==0) &&  (mod(pos2(2),2)==0) % extended into a null lattice point
                        continue; % cant use this
                    end
                end
                
                if (X(pos1(1),pos1(2))==5)&&(X(pos2(1),pos2(2))==6) %if we have selected a cross bound molecule (vertical)
                    independent = [0,0]; %neither site is independent to any other
                    Pold = p12; %old free energy
                    break % we can use this config - break
                end
            else %horizontal
                horiz = 1; %flag horizontal
                pos2=[Xp(pos1(1)),pos1(2)]; %extend "across"
                
                if (lattice42)
                    if (mod(pos2(1),2)==0) &&  (mod(pos2(2),2)==0) % extended into a null lattice point
                        continue; % cant use this
                    end
                end
                
                if (X(pos1(1),pos1(2))==3)&&(X(pos2(1),pos2(2))==4) %if we have selected a cross bound molecule (horizontal)
                    independent = [0,0]; %neither site is independent to any other
                    Pold = p12; %old free energy
                    break % we can use this config - break
                end
            end  
            
            if (X(pos1(1),pos1(2))<3)&&(X(pos2(1),pos2(2))<3) %neither site is implicated in cross binding
                independent = [1,1]; %both sites are independent and can be freely replaced
                Pold=1;
                if (X(pos1(1),pos1(2))==2) %site 1 is occupied
                    Pold = Pold * p1; %adjust fe
                end
                if (X(pos2(1),pos2(2))==2) %site 2 is occupied
                    Pold = Pold * p1; %adjust fe
                end
                
                break % we can use this config - break
            elseif (X(pos1(1),pos1(2))<3) % site 1 can be freely replaced
                independent = [1,0]; %record independent status
                if (X(pos1(1),pos1(2))==2) % site is occupied
                    Pold =  p1; %adjust fe
                end
                break % we can use this config - break
            elseif (X(pos2(1),pos2(2))<3) % site 2 can be freely replaved
                independent = [0,1]; % record independent status
                if (X(pos2(1),pos2(2))==2) % site is occupied
                    Pold =  p1; %adjust fe
                end
                break % we can use this config - break
            end 
        end
           
        id = sum(independent);
        
        if (id==0)||(id==2) 
        %   both are unrelated to other sites or we have selected a cross bound 
        %   molecule so can propose exchanging both.
            
            config =  ceil(5*rand(1)); % generate proposed new config
            % we can propose 5 configurations with P>0.
            
            % 1. both vacant
            % 2. [single bound, vacant]
            % 3. [vacant, single bound]
            % 4. [single bound, single bound]
            % 5. mutually cross bound
            
            if config == 1 % replace whatever is there with 2 vacancies
                s1 = 1; %new states
                s2 = 1;
                Pnew = 1; %new prob (propto)
            elseif config == 2 %etc
                s1 = 2;
                s2 = 1;
                Pnew = p1;
            elseif config == 3
                s1 = 1;
                s2 = 2;
                Pnew = p1;
            elseif config == 4
                s1 = 2;
                s2 = 2;
                Pnew = p1 * p1;
            elseif config == 5
                Pnew = p12;
                if horiz ==1 %states depend on whether we have selected horizontal/vertical sites
                    s1 = 3;
                    s2 = 4;
                else
                    s1 = 5;
                    s2 = 6;
                end
            end
            
            BoltzFactor=Pnew/Pold; %boltzmann factor
            
            % note full accpetance prob is given by ratio
            % (Pnew/Pold)*(q(old|new)/q(new|old)) but the latter term is 1.
            % since, specifically, here q(new|old) = q(old|new) = 0.2
            
            % It is symmetric in this way as all 5 proposed configurations
            % would lead to this same branch-point where
            % id = sum(independent)= 0 | 2
            % i.e. the algorithm is micro-reversible.
            
            if (rand(1)<BoltzFactor) %accept/reject, metropolis hastings
                X(pos1(1),pos1(2))=s1; 
                X(pos2(1),pos2(2))=s2;
            end
            
        else %only one of the sites is not cross bound
        
            config =  ceil(2*rand()); %can choose two new configs with P>0
            
            %1. replace possible single site with vacancy
            %2. replace possible single site with single occupancy
                        
            if config == 1
                s1 = 1;
                Pnew = 1;
            elseif config == 2
                s1 = 2;
                Pnew = p1;
            end
            
            % as before, independent of proposal distribution, but now
            % q(new|old) = q(old|new) = 0.5 and proposed new states also
            % have id = sum(independent) = 1
            
            BoltzFactor=Pnew/Pold; %boltzmann factor
            
            if (rand(1)<BoltzFactor) %accept/reject, metropolis hastings
                if (independent(1)==1) %which site are we talking about?
                    X(pos1(1),pos1(2))=s1;
                else
                    X(pos2(1),pos2(2))=s1;
                end
            end          
        end
   
        if (percolation)
            %%%%%% test for percolation

            if (mod(t,del_t)==0)&&(t>min_t)

                perc_attempt = perc_attempt+1;

                z = X>1; %binary values based on occupied or not, 1 - vacant, 2-6 - occupied to some extent
                [lw,num] = bwlabel(z,4); %create matrix of labelled connected components
                A=intersect(lw(1,:),lw(L,:)); % if any connected component index on lhs equals one on rhs we have percolated

                for ii=1:length(A)
                    if A(ii)>0 % 0 is not a connected component
                        perc_success = perc_success + 1;
                        break;
                    end
                end
                occ=sum(z,'all');%total occupied in any way
                Pocc(count) = Pocc(count)+(occ)/(L*L);
                            
            end
            run_count = run_count +1;
            %%%%%%% end test for perclation
        else

            %statistics
            if (mod(t,del_t)==0)&&(t>min_t)
                c=0;b=0;u=0;tot=0;
                for i=1:L
                    for j=1:L

                        if (lattice42)
                            if (mod(i,2)==0)&&(mod(j,2)==0) %null site
                                continue;
                            end
                        end

                        if (X(i,j)>2) % cross bound site
                            c=c+1;
                        elseif (X(i,j)==2) %singly occupied site
                            b=b+1;
                        else % vacant site
                            u=u+1;
                        end
                        tot=tot+1; %number of sites - manually calc'd in case of lattice42
                    end
                end

                %statistics of various occupancies
                Pcross(count)=Pcross(count)+c/tot;
                Pocc(count) = Pocc(count)+(c+b)/tot;
                run_count=run_count+1;
            end
        end        
    end

    if (percolation)
        %occupancy and percolation as sampled
        %assumes n1=n2=4 lattice
        Pcrossref(count) = 2   - ((16* c0*cc)/(c0*c0 + k1*k1 + 2*c0*(8*cc + k1))) - ((2*(c0 + k1)*sqrt(c0*c0 + k1*k1 + 2*c0*(6*cc + k1)))/( c0*c0 + k1*k1 + 2*c0*(8*cc + k1)));
        Poccref(count) = Pcrossref(count) + (c0/(c0+k1))*(1-Pcrossref(count));
        Pocc(count) =  Pocc(count)/run_count;
        Pperc(count) = perc_success/ perc_attempt;
        
    else  
        %average obtained from the sampling
        Pcross(count) = Pcross(count)/run_count;
        Pocc(count) = Pocc(count)/run_count;

        % reference solution - i.e. approximate theory
        if (lattice42)
            Pcrossref(count) = 2*(2*(c0*c0 + 6*c0*cc + 2*c0*k1 + k1*k1 - sqrt(-4*c0*cc*(8*c0*cc + (c0 + k1)*(c0+k1)) + ((c0*c0 + k1*k1 + 2*c0*(3*cc + k1))^2))))/(3*(c0*c0 + k1*k1 + 2*c0*(4*cc + k1)));
        else
            Pcrossref(count) = 2 - ((16* c0*cc)/(c0*c0 + k1*k1 + 2*c0*(8*cc + k1))) - ((2*(c0 + k1)*sqrt(c0*c0 + k1*k1 + 2*c0*(6*cc + k1)))/( c0*c0 + k1*k1 + 2*c0*(8*cc + k1)));
        end
        
    end

    % concentration value
    Clist(count)=c0;
    fprintf("Finished concentration %s\n",Clist(count));

end

%plot the result
if (percolation)
    plot(Pocc,Pperc);
    figure;
    plot(Pocc,Pperc);
else
    semilogx(Clist,Pcrossref);hold on;
    semilogx(Clist,Pcross);
end






