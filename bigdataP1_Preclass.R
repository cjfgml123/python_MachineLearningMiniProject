print("1.행렬 A,B를 출력하시오.")
names<-list(c("  |","  |","  |"),c(" "," "))
A<-matrix(c(6,0,-1,1,-3,2),nrow=3,ncol=2,dimnames=names); B<-matrix(c(4,0,-5,2,1,-1),nrow=3,ncol=2,dimnames=names); 
A
B
print("2.덧셈행렬 A+B 와 곱셈 행렬 A*B를 출력하시오.")
A+B 
A*B
print("3. 행렬 A의 전치 행렬을 구하시오.")
t(A)
print("4.원소 행렬 A를 입력하고 이 행렬의 역행렬B를 구하시오.")
A<-matrix(c(4,0,3,0,1,0,5,-6,4),nrow=3,ncol=3); 
A 
B<-solve(A); 
B
print("5.두 행렬 A와 B의 곱을 구하시오.")
A%*%B
print("6. B와 A의 곱을 구하시오.")
B%*%A
print("A[,1]와 A[1,]의 차이점을 출력으로 보이시오.")
A[,1]
A[1,]
print("8. 행렬 A에서 하나의 행 및 열의 원소들로 구성된 행렬을 만드시오.")
A[1,,drop=FALSE]
A[,1,drop=FALSE]