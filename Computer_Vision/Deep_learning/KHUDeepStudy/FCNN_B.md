## Title: Neural Network Study (2) - Fully Connnected Neural Network Part. B

*본 포스트는 Hands-on Machine learning 2nd Edition, CS231n, Tensorflow 공식 document를 참조하여 작성되었음을 알립니다.*

###### Index  

1. ~~Basic of Neural Network~~  
2. ~~Definition of Fully Connected Neural Network(FCNN)~~
3. ~~Feed Forward of FCNN~~
4. ~~Gradient Descent~~
5. ~~Back Propagation of FCNN~~
6. Partice(+ 부록 Hyper parameter tuning)

<br>

#### 6. Partice(+ 부록 Hyper parameter tuning)
-------------------------------------------------------

자 실습 시간이다. 왜 실습을 Part. B로 뺐느냐? FCNN이 뭐 할게 있다고?
뭐 할게 있겠다. Tensorflow 2가 대충 어떻게 이루어 졌는지 설명하기 위해 분량 조절을 위해서 뺀것이다.
무엇보다 Part. A 쓰는데 수식을 너무 많이 써서 힘들어서 분리했다. ~~Tlqkf~~
자, 우선 실습에 들어가기에 앞서, TF 2를 애정하는 나로써는 앞으로 이 스터디 포스트에 작성될 대부분의 소스코드를 꿰뚫는 구현 체계를 먼저 설명하고 넘어가겠다.
다음 사진을 보자.

<p align="center"><img src="./assests/API.png" width="70%" height="70%"></p>

*출처: pyimagesearch blog: [링크](https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/)*

위 그림에서 필자는 대부분의 코드를 **Model Subclassing** 방식으로 구현할 것이다. 구현 하면서 설명할 터이니 잘 따라와 주기를 바란다.
여기서부터는 대학교 강의 수준의 **객체지향프로그래밍** 지식을 갖추지 않으면 읽기 힘들 수 있다. "상속", "오버라이딩"의 개념이라도 살펴보고 오자.

###### 6-1. Model Subclassing
----------------------------------------------------------
우선 준비한 소스부터 보고 시작하자.
```python

```