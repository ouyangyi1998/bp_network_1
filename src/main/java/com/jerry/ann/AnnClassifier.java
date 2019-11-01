package com.jerry.ann;

import java.util.ArrayList;
import java.util.List;

public class AnnClassifier {
    private int mInputCount;//输入层数量
    private int mHiddenCount;//隐含层数量
    private int mOutputCount;//输出层数量

    private List<NetworkNode> mInputNodes;//输入层神经元集合
    private List<NetworkNode> mHiddenNodes;//隐含层神经元集合
    private List<NetworkNode> mOutputNodes;//输出层神经元集合

    private float[][] mInputHiddenWeight;//输入权值矩阵
    private float[][] mHiddenOutputWeight;//隐含权值矩阵

    private List<DataNode> trainNodes;//用于测试的数据集合

    public void setTrainNodes(List<DataNode> trainNodes) {
        this.trainNodes = trainNodes;
    }

    public AnnClassifier(int inputCount, int hiddenCount, int outputCount)//输入隐含输出层数目
    {
        trainNodes = new ArrayList<>();
        mInputCount = inputCount;
        mHiddenCount = hiddenCount;
        mOutputCount = outputCount;
        mInputNodes = new ArrayList<>();
        mHiddenNodes = new ArrayList<>();
        mOutputNodes = new ArrayList<>();
        mInputHiddenWeight = new float[inputCount][hiddenCount];
        mHiddenOutputWeight = new float[mHiddenCount][mOutputCount];

    }

    //更新权重
    private void updateWeights(float eta) {
        for (int i = 0; i < mInputCount; i++) {
            for (int j = 0; j < mHiddenCount; j++) {
                mInputHiddenWeight[i][j] -= eta
                        * mInputNodes.get(i).getmForwardOutputValue()
                        * mHiddenNodes.get(j).getmBackwardOutputValue();
            }
        }
        for (int i = 0; i < mHiddenCount; i++) {
            for (int j = 0; j < mOutputCount; j++) {
                mHiddenOutputWeight[i][j] -= eta
                        * mHiddenNodes.get(i).getmForwardOutputValue()
                        * mOutputNodes.get(j).getmBackwardOutputValue();
            }
        }
    }

    //前向传播
    private void forward(List<Float> list) {
        for (int k = 0; k < list.size(); k++)//设置前向输入层的值
        {
            mInputNodes.get(k).setmForwardInputValue(list.get(k));
        }
        for (int j = 0; j < mHiddenCount; j++)//设置隐含输入层的值
        {
            float temp = 0;
            for (int k = 0; k < mInputCount; k++) {
                temp = (temp + mInputHiddenWeight[k][j]) * mInputNodes.get(k).getmForwardOutputValue();
            }
            mHiddenNodes.get(j).setmForwardInputValue(temp);//隐含层2前向输入get
        }
        for (int j = 0; j < mOutputCount; j++) {
            float temp = 0;
            for (int k = 0; k < mHiddenCount; k++) {
                temp = (temp + mHiddenOutputWeight[k][j])
                        * mHiddenNodes.get(k).getmForwardOutputValue();

            }
            mOutputNodes.get(j).setmForwardInputValue(temp);//输出层前向输入get
        }
    }

    //反向
    private void backward(int type)//type为train的原本属性
    {
        for (int j = 0; j < mOutputCount; j++) {//1 属于 -1不属于
            float result = -1;
            if (j == type) {
                result = 1;
            }
            mOutputNodes.get(j).setmBackwardInputValue(mOutputNodes.get(j).getmForwardOutputValue() - result);
        }
        for (int j = 0; j < mHiddenCount; j++)//隐含层矩阵调节
        {
            float temp = 0;
            for (int k = 0; k < mOutputCount; k++) {
                temp = (temp + mHiddenOutputWeight[j][k]) * mOutputNodes.get(k).getmBackwardOutputValue();
            }
            mHiddenNodes.get(j).setmBackwardInputValue(temp);
        }

    }

    //训练 先进行初始化
    //1.前向传播 把train集合的数据导入前向
    //2.反向传播 把train集合的种类导入反向传播
    //3.更新权值 eta系数更新权值
    // eta:权值更新系数  n:为训练次数
    public void train(float eta, int n) {
        reset();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < trainNodes.size(); j++) {
                forward(trainNodes.get(j).getmAttribList());
                backward(trainNodes.get(j).getType());
                updateWeights(eta);
            }
            System.out.println("n= " + i);
        }
    }

    //初始化
    // 1.对于神经元集合添加节点
    // 2.对于权值矩阵的初始化
    private void reset() {
        mInputNodes.clear();
        mOutputNodes.clear();
        mHiddenNodes.clear();
        for (int i = 0; i < mInputCount; i++) {
            mInputNodes.add(new NetworkNode(NetworkNode.TYPE_INPUT));
        }
        for (int i = 0; i < mHiddenCount; i++) {
            mHiddenNodes.add(new NetworkNode(NetworkNode.TYPE_HIDDEN));
        }
        for (int i = 0; i < mOutputCount; i++) {
            mOutputNodes.add(new NetworkNode(NetworkNode.TYPE_OUTPUT));
        }
        for (int i = 0; i < mInputCount; i++) {
            for (int j = 0; j < mHiddenCount; j++) {
                mInputHiddenWeight[i][j] = (float) (Math.random() * 0.1);
            }
        }
        for (int i = 0; i < mHiddenCount; i++) {
            for (int j = 0; j < mOutputCount; j++) {
                mHiddenOutputWeight[i][j] = (float) (Math.random() * 0.1);
            }
        }
    }

    //判断花卉的类别
    public int test(DataNode dn) {
        forward(dn.getmAttribList());
        float res = 2;
        int type = 0;
        for (int i = 0; i < mOutputCount; i++) {
            if ((1 - mOutputNodes.get(i).getmForwardOutputValue()) < res) {
                res = 1 - mOutputNodes.get(i).getmForwardOutputValue();
                type = i;
            }
        }
        return type;
    }
}
