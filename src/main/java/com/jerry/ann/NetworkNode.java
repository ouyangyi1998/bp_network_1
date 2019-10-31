package com.jerry.ann;

public class NetworkNode {
    public final static int TYPE_INPUT=0;
    public final static int TYPE_HIDDEN=1;
    public final static int TYPE_OUTPUT=2;

    private int type;

    public void setType(int type) {
        this.type = type;
    }
    //节点前向输入输出值
    private float mForwardInputValue;
    private float mForwardOutputValue;
    //节点反向输入输出值
    private float mBackwardInputValue;
    private float mBackwardOutputValue;

    public NetworkNode()
    {

    }
    public NetworkNode(int type)
    {
        this.type=type;
    }
    private float forwardSigmoid(float in)
    {
        switch (type)
        {
            case TYPE_INPUT: return in;
            case TYPE_HIDDEN:
            case  TYPE_OUTPUT: return tanhS(in);
        }
        return 0;
    }
    private float tanhS(float in)
    {
        return (float)((Math.exp(in)-Math.exp(-in))/(Math.exp(in)+Math.exp(-in)));
    }
    //导数
    private float tanSDerivative(float in){
        return (float)((1-Math.pow(mForwardOutputValue,2))*in);
    }
    //误差反向传播
    private  float backwardPropagate(float in)
    {
        switch (type)
        {
            case TYPE_INPUT :return in;
            case  TYPE_HIDDEN:
            case  TYPE_OUTPUT:return tanSDerivative(in);
        }
        return 0;
    }

    public float getmForwardInputValue() {
        return mForwardInputValue;
    }

    public void setmForwardInputValue(float mForwardInputValue) {
        this.mForwardInputValue = mForwardInputValue;
        setmForwardOutputValue(mForwardInputValue);
    }

    public float getmForwardOutputValue() {
        return mForwardOutputValue;
    }

    public void setmForwardOutputValue(float mForwardOutputValue) {
        this.mForwardOutputValue = forwardSigmoid(mForwardOutputValue);
    }

    public float getmBackwardInputValue() {
        return mBackwardInputValue;
    }

    public void setmBackwardInputValue(float mBackwardInputValue) {
        this.mBackwardInputValue = mBackwardInputValue;
        setmBackwardOutputValue(mBackwardInputValue);
    }

    public float getmBackwardOutputValue() {
        return mBackwardOutputValue;
    }

    public void setmBackwardOutputValue(float mBackwardOutputValue) {
        this.mBackwardOutputValue = backwardPropagate(mBackwardOutputValue);
    }
}
