### Distribution Functions and Operations

이 파트에서는 확률 분포 함수와 그것들을 이용한 연산들이 어떤 것이 있는지 알아본다.

#### Discrete and Continuous Random Variables and Distributions

확률 변수에는 이산형과 연속형 확률 변수가 있다. 우리는 이산형과 연속형 확률 변수의 정의와 그것들의 분포 함수들의 정의를 알아볼 것이다. 우선, 각 확률 변수의 정의부터 알아보도록 하자.

* 이산형 확률 변수
  
  확률 변수의 공간이 가산집합의 경우, 이 확률 변수를 이산형 확률 변수라고 한다.

* 연속형 확률 변수
  
  확률 변수 $X$의 누적 분포 함수 $F_X(x)$가 $x \in \R$에서 연속이면 이 확률 변수를 연속 확률 변수라고 한다.

그리고 이러한 확률 변수에서 확률을 도출해 내는 것이 우리의 목표이다. 그것을 위해서 다양한 해석이 가능한데, 대표적인 예시로 우리는 누적 분포 함수를 사용할 수도 있지만 다른 접근법 또한 가능하다. 그것이 바로 probability density function과 probability mass function이다. 

* Probability Mass Function
  
  이것은 이산 확률 변수로부터 확률을 얻어내는 함수의 일종이다. 정의는 다음과 같다.
  $$
  x \in D, p_X(x) = P[X = x]
  \tag{1}
  $$
  누적 분포 함수에서 단순한 뻴셈으로 이것을 유도할 수 있다. 
  $$
  F_X(x_i) = P[X \leq x_i] = \Sigma^{i}_{j=1}P[X = x_j]
  $$
  으로 생각할 수 있는데, 이러한 두 함수의 차를 구하여도, 이는 확률로써 정의할 수 있다. 결론적으로 이렇게 정의되는 것이 확률 질량 함수이다.  
  이러한 확률 질량 함수에는 다음과 같은 특징이 있다.
  $$
  0 \leq p_X(x) \leq 1, x \in D
  \tag{1}
  $$
  $$
  \Sigma_{x \in D} p_X(x) = 1
  \tag{2}
  $$

* Probability Density Function
  
  이것은 

#### Operations



#### Famous Inequality


