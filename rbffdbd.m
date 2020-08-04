load XYdatanearoptimalbd300point5700mat.mat XY
x = XY(:,1);
y = XY(:,2);
X_vec = [x,y];

C = [1:299;2:300];
C = transpose(C);
C(300,:) = [300,1];
dt=delaunayTriangulation(x,y,C);
io = dt.isInterior();
patch('faces',dt(io,:), 'vertices', dt.Points, 'FaceColor','r');

node = dt.Points;
elem = dt(io,:);
NT = size(elem,1);

Dirichlet = (1:300);
Dirichlet = transpose(Dirichlet);
%------------------------Stiffness matrix with RHS ------------------------
[A, b] = assembling2(node);

%-------------------- Dirichlet boundary conditions----------------------
NV_tot = 5700;
isBdNode = false(NV_tot,1);
isBdNode(Dirichlet) = true;
bdNode = find(isBdNode);
freeNode = find(~isBdNode);
%plot(x(bdNode),y(bdNode),'x', x(freeNode),y(freeNode),'o');

f_sol = @(X) exp(-5*(((X(:,1)-1.75).^2)+(X(:,2).^2)));

A(bdNode,:)=0; 
A(bdNode,bdNode)=speye(length(bdNode),length(bdNode));
b(bdNode) = f_sol([x(bdNode), y(bdNode)]);

%-------------------- Linear Algebra -----------------------------------
%A_full = full(A);
figure(2);
sol = pinv(A)*b;
sol(5701:11630) = [];
F = scatteredInterpolant(x,y,sol);
z = F(x,y) ;
trisurf(elem,x,y,z);
%sol = reshape(sol,NV,2*NV);
%figure;surf(sol);

sol_anal = f_sol([x, y]);
figure(4);
err = abs(sol-sol_anal)./sol_anal;
F_e = scatteredInterpolant(x,y,err);
z_e = F_e(x,y) ;
trisurf(elem,x,y,z_e);
%sol_anal = reshape(sol_anal,NV,2*NV);
%figure;surf(sol_anal);

function [A, F] = assembling2(node)
load bounadryerr5930point.mat xy
np = [node;xy];
mnp = size(np,1);
n = 100;

N = size(node,1);

F = zeros(N,1); % load vector F to hold integrals of phi's times load f(x,y)
A = zeros(N,mnp); 
e = ones(n,1);

f_sol = @(X) exp(-5*(((X(:,1)-1.75).^2)+(X(:,2).^2)));
f = @(X) 10 * f_sol(X) .* ( 10 * ( ((X(:,1)-1.75).^2)+(X(:,2).^2) ) - 2 );

phi = @(X) exp(-((3*X).^2));
L_phi = @(X) 36*phi(X) .* (9*X.^2 - 1);
for i = 1:N
    nod = node(i,:);
    [Idx, D] = knnsearch(np,nod,'k',n);
    neighbor = np(Idx,:);
    B = zeros(n,n); % zero matrix in sparse format: zeros(N) would be "dense"
    LB = zeros(n,1);
    for j = 1:n
        vec = neighbor - neighbor(j,:);
        dist = sqrt(vec(:,1).^2+vec(:,2).^2);
        B(j,:) = phi(dist);       
    end
    B = [B e;transpose(e) 0];
    LB = L_phi(D);
    LB = [transpose(LB); 0];
    %LB = transpose(LB);
    B_sol = pinv(B)*LB;
    A(i,Idx) = B_sol(1:n);
    F(i) = f(nod);   
end
F = transpose(A)*F;
A = transpose(A)*A;
%figure(1);
%surf(A);
end