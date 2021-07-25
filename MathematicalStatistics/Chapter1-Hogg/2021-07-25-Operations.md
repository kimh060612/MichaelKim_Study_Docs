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
  F_X(x_i) = P[X \leq x_i] = \sum^{i}_{j=1}P[X = x_j]
  $$
  으로 생각할 수 있는데, 이러한 두 함수의 차를 구하여도, 이는 확률로써 정의할 수 있다. 결론적으로 이렇게 정의되는 것이 확률 질량 함수이다.  
  이러한 확률 질량 함수에는 다음과 같은 특징이 있다.
  $$
  0 \leq p_X(x) \leq 1, x \in D
  \tag{1}
  $$
  $$
  \sum_{x \in D} p_X(x) = 1
  \tag{2}
  $$

* Probability Density Function
  
  이산 확률 변수와는 달리, 연속 확률 변수는 위와 같이 정의를 할 수 없다. 누적 분포의 정의에 따라서 다음이 성립하는 것을 보고 연속 확률 변수를 탐구해 보자.
  $$
  P(X = x) = F_X(x) - F_X(x-) \\
  F_X(x_0-) = \lim_{x \downarrow x_0} F_X(x)
  $$
  하지만 연속 확률 변수의 정의에 따라서 $F_X(x)$는 완전 연속이기에, $P(X = x) = 0$이 항상 성립하게 된다. 따라서 이렇게 분포 함수를 정의하면 안되기에, 다음과 같이 확률 밀도 함수를 정의한다.
  $$
  F_X(x) = \int_{-\infin}^x f_X(x)dx
  $$
  미분적분학의 기본 정리에 따라서 다음이 성립한다.
  $$
  \frac{d}{dx}F_X(x) = f_X(x)
  $$

* Transformation

  이러한 확률 변수들은 변수변환이 가능하다. 예를 들어서, $Y$라는 확률 변수는 $X$로 변환할 수 있을때, $Y$의 확률 분포를 $X$의 분포로 표현할 수 있다.

  * Discrete
    
    이산 확률 변수의 변환은 간단하다. $X$가 공간 $D_X$를 가진 이산형 확률 변수라고 가정해 보자. 그렇다면 확률 변수 $Y = g(X)$일때, $Y \in \{g(x): x \in D_X\}$이다. 그렇다면 정의에 의해서, 다음이 성립한다.
    $$
    P[Y = y] = P[g(X) = y] = P[X = g^{-1}(y)]
    $$
    이때, $g(X)$가 일대일 대응인 함수와 일대일 대응이 아닌 함수로 나눌 수 있는데, 일대일 함수면 그냥 역함수를 써도 되지만, 그것이 아닐 경우에는 범위를 나누어서 일대일 대응으로 부분 함수를 만들어서 생각해야한다. 

  * Continuous

    이 경우에는 이산형처럼 PMF를 바로 사용하여 유도를 할 수 없다. 그렇기에 CDF를 활용하여서 유도를 진행한다. 변수 관계는 다음과 같다고 하자.
    $$
    F_Y(y) = P(Y \leq y) = P(g(X) \leq y) = P(X \leq g^{-1}(y)) = F_X(g^{-1}(y))
    $$
    그리고 이를 통해서 우리는 확률 변수 $Y$의 PDF를 얻을 수 있다.
    $$
    \frac{d}{dy}F_Y(y) = \frac{d}{dy}F_X(g^{-1}(y)) = f_X(g^{-1}(y))|\frac{dx}{dy}|
    $$
    이때, 왜 절대값이냐에 대한 논증은 다음과 같다.
    
    1. $g(X)$가 순 단조 증가일 경우
        
        이때 $\frac{dx}{dy} \geq 0$이므로 위의 식이 성립한다.

    2. $g(X)$가 순 단조 감소일 경우

        이 경우에는 $\frac{dx}{dy} \leq 0$이고 $F_Y(y) = 1 - F_X(g^{-1}(y))$이다.
        따라서 이 둘을 합치면 위의 식이 성립한다.

    3. $g(X)$가 무엇도 아닐 경우
        
        이 경우에는 범위를 나눠서 생각해야 한다. 범위를 나누어 일대일 대응으로 함수를 나누어 확률 밀도 함수를 구할 수 있다.

#### Operations

이 파트에서는 확률 분포 함수로 수행할 수 있는 여러가지 연산들을 알아보도록 한다.

* Expectation
  
  이는 굳이 설명을 하지 않더라도 알고 있는 사람들이 많을 것이다. 간단하게 정의만 작성해 두겠다.  
  다음이 식 1이 존재할때, 식2를 평균으로 정의한다.
  
  연속일 경우
  $$
  \int_{-\infty}^{\infty} |x|f_X(x)dx < \infty
  \tag{1}
  $$

  $$
  E[X] = \int_{-\infty}^{\infty} xf_X(x)dx
  \tag{2}
  $$ 

  이산일 경우
  $$
  \sum^{\infty}_{-\infty} |x|f_X(x) < \infty
  \tag{1}
  $$

  $$
  E[X] = \sum^{\infty}_{-\infty} xf_X(x)
  \tag{2}
  $$

* Variance
  
  이는 더더욱 간단하다. 다음을 분산이라고 정의한다.
  $$
  Var(X) = E[(X - \mu)^2]\\
  (\mu = E[X])
  $$ 
  
  이때, $Var(X) = \sigma^2$으로 표현하기도 한다.
  이는 다음과 같이 표현되기도 하며, 보통 평균으로부터 얼마나 분포가 퍼져 있는지를 나타내는 지표로 사용된다.
  $$
  E[(X - \mu)^2] = E[X^2] - (E[X])^2
  $$

* MGF (Moment Generation Function)
  
  여기서 Moment는 다음과 같이 정의된다.  
  $$
  m_n = E[X^n]
  $$
  이를 n차 moment라고 하는데, 이를 쉽게 구할 수 있게 해주는 것이 MGF, Moment Generation Function이다. 이는 다음과 같이 정의된다.
  $$
  M_X(t) = E[e^{tX}] = \int_{-\infty}^{\infty} e^{tX}f_X(x)dx
  $$

  이것을 moment generation function으로 부르는 이유는 이 함수를 $N$번 미분하고 0을 대입하면 $m_n$이 되기 때문이다. 이의 증명은 너무 쉽기 때문에 굳이 여기서는 증명하지 않겠다.
  $$
  \frac{d^n}{dt^n}M_X(t)\bigg|_{t = 0}  = E[X^n]
  $$

  그리고 눈치챈 이도 있겠지만, 이는 라플라스 변환과 굉장히 유사한 형태를 띄고 있다는 것을 알 수 있다. 그것을 활용하여 이 식의 유일성을 증명할 수 있다.

* Characteristics Function
  
  이는 방금 배운 것의 복소수 버전이다. 다음과 같이 정의된다고 생각하면 된다.
  $$
  \Phi_X(\omega) = E[e^{i\omega X}]
  $$

  이 또한 다음과 같이 $N$번 미분하였을때 $m_N$을 생성할 수 있다.
  $$
  (-i)^n\frac{d^n}{d\omega^n}\Phi_X(\omega)\bigg|_{\omega=0} = E[X^n]
  $$

  이것 또한 배운 사람은 알겠지만, 이는 푸리에 변환과 형태가 매우 유사함을 알 수 있다. 따라서 이를 통해서 우리는 이 식의 유일성을 입증할 수 있다.

#### Famous Inequality

이 파트에서는 확률 변수와 관련된 유명한 부등식을 알아볼 예정이다.

1. Markov's Inequality
  
    이 식의 정의는 다음과 같다.
    확률 변수 $X$에 대하여 음이 아닌 함수 $u(X)$가 있다고 가정하자. 이때, $E[u(X)]$가 존재하면 모든 양수 $c$에 대하여 다음이 성립한다.
    $$
    P[u(X) \geq c] \leq \frac{E[u(X)]}{c}
    $$ 

    이에 관한 자세한 증명은 [위키 피디아](https://en.wikipedia.org/wiki/Markov%27s_inequality)를 참조하기를 바란다.

2. Chebyeshev's Inequality

    이 식은 본질적으로 Markov's Inquality와 같다. 정확히는, Chebyeshev's Inequality의 일반화가 Markov's Inquality이다. 이 식의 정의는 다음과 같다.  
    확률 변수 $X$가 유한 분산 $\sigma^2$를 가졌을때, 모든 $k > 0$에 대하여 다음이 성립한다.
    $$
    P[|X - \mu| \geq k\sigma] \leq \frac{1}{k^2}
    $$
    이는 위의 Markov's Inquality를 통해서 쉽게 증명이 가능하다.
    Markov's Inequality에 $u(X) = (X - \mu)^2$이고 $c = (k\sigma)^2$이라고 할때 다음이 성립한다.
    $$
    P[(X - \mu)^2 \geq (k\sigma)^2] \leq \frac{E[(X - \mu)^2]}{(k\sigma)^2}
    \rightarrow P[|X - \mu| \geq k\sigma] \leq \frac{1}{k^2}
    $$
    
