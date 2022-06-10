# ML_TeamProject
심장병 분류(불균형 데이터)\
캐글 데이터 셋 이용 https://www.kaggle.com/code/omarmohamedyehia/heart-disease-prediction

# 코드가 보이지 않을 경우
용량 문제로 일부 파일이 깃허브에서 로드가 되지 않습니다.\
보이지 않을 경우 아래 링크들을 참고해 주세요.\
아래 링크로도 보이지 않을 경우 https://nbviewer.org/ 에 접속하셔셔 각 코드의 링크를 직접 넣어주기길 바랍니다.


1. EDA : https://nbviewer.org/github/power-TY/ML_TeamProject/blob/main/EDA.ipynb

2. Data_Imbalancee_Handling : https://nbviewer.org/github/power-TY/ML_TeamProject/blob/main/Data_Imabalance_Handling.ipynb

3. TargetEncoding_Sampling_Logistic_XAI : https://nbviewer.org/github/power-TY/ML_TeamProject/blob/main/TargetEncoding_Sampling_Logistic_XAI.ipynb#%EB%AA%A8%EB%8D%B8-%EC%A0%81%EC%9A%A9

4. BinaryEncoding_Sampling_Logistic_XAI.ipynb : 

5. OneHotEncoding_Sampling_Logistic_XAI.ipynb : https://nbviewer.org/github/power-TY/ML_TeamProject/blob/main/OneHotEncoding_Sampling_Logistic_XAI.ipynb

6. 

# 프로젝트 선정 이유
내용추가 예정.\
내용추가 예정.

# 데이터 불균형 문제 해결 방식
1. UnderSampling : 다수 클래스에서 샘플링을 통해 관측치를 지워 불균형 문제를 해결하는 방식
   종류 : CNN[3], Tomek Links[4], ENN[5], RENN[6], ALLKNN[6], OSS[7], NCR[8], NearMiss-1, 2, 3[9]
2. OverSampling : 소수 클래스에서 데이터를 복제 및 추가하는 방식
   종류 : SMOTE[10], BorderlineSMOTE[11], ADASYN[12]
3. Combining Over-and Under-Sampling : 오버샘플링과 언더샘플링 방식을 결합한 방식
   종류 : SMOTE+Tomek[13], SMOTE+ENN[14]
4. Cost-Senstive Learning : 클래스를 오분류하는 비용에 따라서 가중치를 두는 학습 방식

# 이용한 데이터 불균형 방식
OverSampling : SMOTE, BorderlineSMOTE, ADASYN\
CombineSampling : SMOTE+ENN\
Cost-Sensitive Learning\
언더 샘플링의 경우 결과가 너무 좋지 않아 이용하지 않음.

# 참고자료
[3] Kubat, Miroslav, and Stan Matwin, “Addressing the curse of imbalanced training sets: one-sided selection,” Proc. International Conference on Machine Learning, Vol.97, pp.179-186, 1997.\
[4] I. Tomek, “Two Modifications of CNN,” IEEE Transactions on Systems, Man and Cybernetics, Vol.6, No.11, pp.769-772,1976.\
[5] Dennis L. Wilson, “Asymptotic properties of nearest neighbor rules using edited data,” IEEE Transactions on Systems, Man, and Cybernetics, Vol.3, pp.408-421, 1972.\
[6]  I. Tomek, “An experiment with the edited nearest-neighbor rule,” IEEE Transactions on systems, Man, and Cybernetics, Vol.6, No.6, pp.448-452, 1976.\
[7] Kubat, Miroslav, and Stan Matwin, “Addressing the curse of imbalanced training sets: one-sided selection,” Proc. International Conference on Machine Learning, Vol.97, pp.179-186, 1997.\
[8] J. Laurikkala, “Improving identification of difficult small classes by balancing class distribution,” Proc. Conference on Artificial Intelligence in Medicine in Europe – Artificial Intelligence in Medicine, pp.63-66, 2001.\
[9] Mani, Inderjeet and I. Zhang, “kNN approach to unbalanced data distributions: a case study involving information extraction,” Proc. workshop on learning from imbalanced datasets, Vol.126, 2003.\
[10] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, Vol.16, No.1, pp.321-357, 2002.\
[11] Han, Hui, Wen-Yuan Wang, and Bing-Huan Mao. "Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning." International conference on intelligent computing. Springer, Berlin, Heidelberg, 2005.\
[12] H. He, Y. Bai, E. A. Garcia, and S. Li, “ADASYN: Adaptive synthetic sampling approach for imbalanced learning,” Proc. IEEE International Joint Conference on Neural Networks, pp.1322-1328, 2008.\
[13] Batista, Gustavo EAPA, Ana LC Bazzan, and Maria Carolina Monard, “Balancing Training Data for Automated Annotation of Keywords: a Case Study,” Proc. Workshop on Bioinformatics, 2003.\
[14] Batista, Gustavo EAPA, Ronaldo C. Prati and Maria Carolina Monard, “A study of the behavior of several methods for balancing machine learning training data,” SIGKDD Explorations, Vol.6, No.1, pp.20-29, 2004.\
