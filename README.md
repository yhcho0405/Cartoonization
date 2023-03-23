# VideoCartoonizer

1. Bilateral filter 적용 (edge를 보존하기 위해 각 채널(RGB)에 개별적으로 적용)
2. Canny edge 알고리즘 적용
3. RGB -> HSV 색 공간으로 변환
4. 세 채널(HSV)에 대한 histogram 계산
5. 각 histogram 대해 cluster center 계산
6. 각 채널의 색상 값을 가장 가까운 cluster center로 대체
7. HSV -> RGB 색 공간으로 변환
8. 2번에서 구한 edge 적용
9. 각 채널을 eroding. (edge 강조)


## 시연 영상

![choi](https://user-images.githubusercontent.com/52823519/227142102-fb6141c9-7249-4181-b5b9-6089676a6687.gif)
