# 정직하게 자신의 값 반환하는 선형함수


# 층을 쌓는 의미가 없게 만듬 y = cx (c는 상수)일 때 진행해나갈 수록 c만큼 곱해져 값이 커지기만 함
# y = cx와 y = c^3 * x 는 큰 차이 없다. (직접 테스트 해봐야겠음)

# 이상치가 존재하면 분류가 어렵다. (4 ~ 5의 값이 들어오면 True일 때 데이터에 20-True인 데이터를 학습하면? 학습이 어렵게 된다) (직접 구현해 봐야겠음)

def identity_function(x):
    return x