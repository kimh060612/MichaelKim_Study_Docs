## Title: Perspective Transformation for Lane Detection

이번 포스트에서는 Lane detection을 위해서 영상의 시점을 전환하는 방법을 다룰 것입니다.  
이를 위해서 필요한 개념은 Perspective Transformation이라는 것입니다. 이것은 다양한 2D Image Transformation 방식 중에 하나입니다.  
여기서 2D Image Transformation은 다음 표와 같은 행렬 변환을 의미합니다. 

<p align="center"><img src="./assests/AT.png" width="100%"></p> <br>

*출처: Matlab 공식 홈페이지 - Affine Transformation 문서*

대충 아시는 분들도 계실텐데, Affine Transformation은 2D 이미지 변환중에서도 그 기하학적인 성질들을 보존하는 변환입니다. 
그렇다면 Perspective Transformation은 무엇일까요? 우선 명확한 정의를 먼저 짚고 넘어갑시다. 
> ##### Perspective Transformation (Homography)
> ##### Definition: 평면 물체의 경우에는 3D 공간에서 2D image로 임의의 변환이 가능합니다. 이때 두 이미지 사이의 변환 관계를 homography로 표현할 수 있는 수단이 존재합니다. 이를 이용한 변환이 Perspective Transformation입니다.

이를 조금 더 직관적으로 해석하자면, *어떤 2D 평면의 사각형을 임의의 다른 사각형으로 mapping할 수 있는 변환*이 Homography입니다. 이는 어떻게 보면 Affine Transformation의 일반화라고 할 수 있습니다.

이를 수학적으로 표현하면 다음과 같습니다.

```math
w
\begin{bmatrix}
    x^{'} \\
    y^{'} \\
    1
\end{bmatrix}
=
\begin{bmatrix}
    h_{11} & h_{12} & h_{13} \\
    h_{21} & h_{22} & h_{23} \\
    h_{31} & h_{32} & h_{33} 
\end{bmatrix}
\begin{bmatrix}
    x \\
    y \\
    1 
\end{bmatrix}
```
이때, 