package com.jerry.ann;

import java.util.ArrayList;
import java.util.List;

public class AnnClassifier {
    private int mInputCount;
    private int mHiddenCount;
    private int mOutputCount;

    private List<NetworkNode> mInputNodes;
    private List<NetworkNode> mHiddenNodes;
    private List<NetworkNode> mOutputNodes;

    private float[][] mInputHiddenWeight;
    private float[][] mHiddenOutputWeight;

    private  List<DataNode> trainNodes;

    public void setTrainNodes(List<DataNode> trainNodes) {
        this.trainNodes = trainNodes;
    }
    public AnnClassifier(int inputCount,int hiddenCount,int outputCount)
    {
        trainNodes=new ArrayList<>();
        mInputCount=inputCount;
        mHiddenCount=hiddenCount;
        mOutputCount=outputCount;
        mInputNodes=new ArrayList<>();
        mHiddenNodes=new ArrayList<>();
        mOutputNodes=new ArrayList<>();
       mInputHiddenWeight=new float[inputCount][hiddenCount];
        mHiddenOutputWeight=new float[mHiddenCount][mOutputCount];

    }
    //更新权重
    private void updateWeights(float eta)
    {
        for (int i=0;i<mInputCount;i++)
        {
            for (int j=0;j<mHiddenCount;j++)
            {
                mInputHiddenWeight[i][j]-=eta
                        *mInputNodes.get(i).getmForwardOutputValue()
                        *mHiddenNodes.get(j).getmBackwardOutputValue();
            }
        }
        for (int i=0;i<mHiddenCount;i++)
        {
            for (int j=0;j<mOutputCount;j++)
            {
                mHiddenOutputWeight[i][j]-=eta
                        *mHiddenNodes.get(i).getmForwardOutputValue()
                        *mOutputNodes.get(j).getmBackwardOutputValue();
            }
        }
    }
    //前向传播
    private void forward(List<Float> list)
    {
        for(int k=0;k<list.size();k++)
        {
            mInputNodes.get(k).setmForwardInputValue(list.get(k));
        }
        for (int j=0;j<mHiddenCount;j++)
        {
            float temp=0;
            for (int k=0;k<mInputCount;k++)
            {
                temp+=mInputHiddenWeight[k][j]
                        *mInputNodes.get(k).getmForwardOutputValue();

            }
            mHiddenNodes.get(j).setmForwardInputValue(temp);
        }
        for(int j=0;j<mOutputCount;j++)
        {
            float temp=0;
            for(int k=0;k<mHiddenCount;k++)
            {
                temp+=mHiddenOutputWeight[k][j]
                        *mHiddenNodes.get(k).getmForwardOutputValue();

            }
            mOutputNodes.get(j).setmForwardInputValue(temp);
        }
    }
    //反向
    private void backward(int type)
    {
        for(int j=0;j<mOutputCount;j++)
        {//1 属于 -1不属于
            float result=-1;
            if(j==type)
            {
                result=1;
            }
            mOutputNodes.get(j).setmBackwardInputValue(mOutputNodes.get(j).getmForwardOutputValue()-result);
        }
        for(int j=0;j<mHiddenCount;j++)
        {
            float temp=0;
            for(int k=0;k<mOutputCount;k++)
            {
                temp+=mHiddenOutputWeight[j][k]*mOutputNodes.get(k).getmBackwardOutputValue();
            }
            mHiddenNodes.get(j).setmBackwardInputValue(temp);
        }

    }
    public void train(float eta,int n)
    {
        reset();
        for(int i=0;i<n;i++)
        {
            for (int j=0;j<trainNodes.size();j++)
            {
                forward(trainNodes.get(j).getmAttribList());
                backward(trainNodes.get(j).getType());
                updateWeights(eta);
            }
            System.out.println("n= "+i);
        }
    }
    private void reset()
    {
        mInputNodes.clear();
        mOutputNodes.clear();
        mHiddenNodes.clear();
        for(int i=0;i<mInputCount;i++)
        {
            mInputNodes.add(new NetworkNode(NetworkNode.TYPE_INPUT));
        }
        for(int i=0;i<mHiddenCount;i++)
        {
            mHiddenNodes.add(new NetworkNode(NetworkNode.TYPE_HIDDEN));
        }
        for(int i=0;i<mOutputCount;i++)
        {
            mOutputNodes.add(new NetworkNode(NetworkNode.TYPE_OUTPUT));
        }
        for(int i=0;i<mInputCount;i++)
        {
            for (int j=0;j<mHiddenCount;j++)
            {
                mInputHiddenWeight[i][j]=(float)(Math.random()*0.1);
            }
        }
        for(int i=0;i<mHiddenCount;i++)
        {
            for (int j=0;j<mOutputCount;j++)
            {
               mHiddenOutputWeight[i][j]=(float)(Math.random()*0.1);
            }
        }
    }
public  int test(DataNode dn)
{
    forward(dn.getmAttribList());
    float res=2;
    int type=0;
    for(int i=0;i<mOutputCount;i++)
    {
        if((1-mOutputNodes.get(i).getmForwardOutputValue())<res)
        {
            res=1-mOutputNodes.get(i).getmForwardOutputValue();
            type=i;
        }
    }
    return type;
}
}
